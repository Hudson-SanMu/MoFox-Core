#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""纯 inkfox 视频关键帧分析工具

仅依赖 `inkfox.video` 提供的 Rust 扩展能力：
    - extract_keyframes_from_video
    - get_system_info

功能：
    - 关键帧提取 (base64, timestamp)
    - 批量 / 逐帧 LLM 描述
    - 自动模式 (<=3 帧批量，否则逐帧)
"""

from __future__ import annotations

import os
import io
import asyncio
import base64
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import hashlib
import time

from PIL import Image

from src.common.logger import get_logger
from src.config.config import global_config, model_config
from src.llm_models.utils_model import LLMRequest
from src.common.database.sqlalchemy_models import Videos, get_db_session  # type: ignore
from sqlalchemy import select, update, insert  # type: ignore
from sqlalchemy import exc as sa_exc  # type: ignore

# 简易并发控制：同一 hash 只处理一次
_video_locks: Dict[str, asyncio.Lock] = {}
_locks_guard = asyncio.Lock()

logger = get_logger("utils_video")

from inkfox import video


class VideoAnalyzer:
    """基于 inkfox 的视频关键帧 + LLM 描述分析器"""

    def __init__(self) -> None:
        cfg = getattr(global_config, "video_analysis", object())
        self.max_frames: int = getattr(cfg, "max_frames", 20)
        self.frame_quality: int = getattr(cfg, "frame_quality", 85)
        self.max_image_size: int = getattr(cfg, "max_image_size", 600)
        self.enable_frame_timing: bool = getattr(cfg, "enable_frame_timing", True)
        self.use_simd: bool = getattr(cfg, "rust_use_simd", True)
        self.threads: int = getattr(cfg, "rust_threads", 0)
        self.ffmpeg_path: str = getattr(cfg, "ffmpeg_path", "ffmpeg")
        self.analysis_mode: str = getattr(cfg, "analysis_mode", "auto")
        self.frame_analysis_delay: float = 0.3

        # 人格与提示模板
        try:
            import cv2

            opencv_available = True
        except ImportError:
            pass

        if not RUST_VIDEO_AVAILABLE and not opencv_available:
            logger.error("❌ 没有可用的视频处理实现，视频分析器将被禁用")
            self.disabled = True
            return
        elif not RUST_VIDEO_AVAILABLE:
            logger.warning("⚠️ Rust视频处理模块不可用，将使用Python降级实现")
        elif not opencv_available:
            logger.warning("⚠️ OpenCV不可用，仅支持Rust关键帧模式")

        self.disabled = False

        # 使用专用的视频分析配置
        try:
            self.video_llm = LLMRequest(
                model_set=model_config.model_task_config.video_analysis, request_type="video_analysis"
            )
            logger.debug("✅ 使用video_analysis模型配置")
        except (AttributeError, KeyError) as e:
            # 如果video_analysis不存在，使用vlm配置
            self.video_llm = LLMRequest(model_set=model_config.model_task_config.vlm, request_type="vlm")
            logger.warning(f"video_analysis配置不可用({e})，回退使用vlm配置")

        # 从配置文件读取参数，如果配置不存在则使用默认值
        config = global_config.video_analysis

        # 使用 getattr 统一获取配置参数，如果配置不存在则使用默认值
        self.max_frames = getattr(config, "max_frames", 6)
        self.frame_quality = getattr(config, "frame_quality", 85)
        self.max_image_size = getattr(config, "max_image_size", 600)
        self.enable_frame_timing = getattr(config, "enable_frame_timing", True)

        # Rust模块相关配置
        self.rust_keyframe_threshold = getattr(config, "rust_keyframe_threshold", 2.0)
        self.rust_use_simd = getattr(config, "rust_use_simd", True)
        self.rust_block_size = getattr(config, "rust_block_size", 8192)
        self.rust_threads = getattr(config, "rust_threads", 0)
        self.ffmpeg_path = getattr(config, "ffmpeg_path", "ffmpeg")

        # 从personality配置中获取人格信息
        try:
            personality_config = global_config.personality
            self.personality_core = getattr(personality_config, "personality_core", "是一个积极向上的女大学生")
            self.personality_side = getattr(
                personality_config, "personality_side", "用一句话或几句话描述人格的侧面特点"
            )
        except AttributeError:
            # 如果没有personality配置，使用默认值
            self.personality_core = "是一个积极向上的女大学生"
            self.personality_side = "用一句话或几句话描述人格的侧面特点"

        self.batch_analysis_prompt = getattr(
            cfg,
            "batch_analysis_prompt",
            """请以第一人称视角阅读这些按时间顺序提取的关键帧。\n核心：{personality_core}\n人格：{personality_side}\n请详细描述视频(主题/人物与场景/动作与时间线/视觉风格/情绪氛围/特殊元素)。""",
        )

        # 新增的线程池配置
        self.use_multiprocessing = getattr(config, "use_multiprocessing", True)
        self.max_workers = getattr(config, "max_workers", 2)
        self.frame_extraction_mode = getattr(config, "frame_extraction_mode", "fixed_number")
        self.frame_interval_seconds = getattr(config, "frame_interval_seconds", 2.0)

        # 将配置文件中的模式映射到内部使用的模式名称
        config_mode = getattr(config, "analysis_mode", "auto")
        if config_mode == "batch_frames":
            self.analysis_mode = "batch"
        elif config_mode == "frame_by_frame":
            self.analysis_mode = "sequential"
        elif config_mode == "auto":
            self.analysis_mode = "auto"
        else:
            logger.warning(f"无效的分析模式: {config_mode}，使用默认的auto模式")
            self.analysis_mode = "auto"

        self.frame_analysis_delay = 0.3  # API调用间隔（秒）
        self.frame_interval = 1.0  # 抽帧时间间隔（秒）
        self.batch_size = 3  # 批处理时每批处理的帧数
        self.timeout = 60.0  # 分析超时时间（秒）

        if config:
            logger.debug("✅ 从配置文件读取视频分析参数")
        else:
            logger.warning("配置文件中缺少video_analysis配置，使用默认值")

        # 系统提示词
        self.system_prompt = "你是一个专业的视频内容分析助手。请仔细观察用户提供的视频关键帧，详细描述视频内容。"

        logger.debug(f"✅ 视频分析器初始化完成，分析模式: {self.analysis_mode}, 线程池: {self.use_multiprocessing}")

        # 获取Rust模块系统信息
        self._log_system_info()

    def _log_system_info(self):
        """记录系统信息"""
        if not RUST_VIDEO_AVAILABLE:
            logger.info("⚠️ Rust模块不可用，跳过系统信息获取")
            return

        try:
            system_info = rust_video.get_system_info()
            logger.debug(f"🔧 系统信息: 线程数={system_info.get('threads', '未知')}")

            # 记录CPU特性
            features = []
            if system_info.get("avx2_supported"):
                features.append("AVX2")
            if system_info.get("sse2_supported"):
                features.append("SSE2")
            if system_info.get("simd_supported"):
                features.append("SIMD")

            if features:
                logger.debug(f"🚀 CPU特性: {', '.join(features)}")
            else:
                logger.debug("⚠️ 未检测到SIMD支持")

            logger.debug(f"📦 Rust模块版本: {system_info.get('version', '未知')}")

        except Exception as e:
            logger.warning(f"获取系统信息失败: {e}")

    def _calculate_video_hash(self, video_data: bytes) -> str:
        """计算视频文件的hash值"""
        hash_obj = hashlib.sha256()
        hash_obj.update(video_data)
        return hash_obj.hexdigest()

        self._log_system()

    # ---- 系统信息 ----
    def _log_system(self) -> None:
        try:
            info = video.get_system_info()  # type: ignore[attr-defined]
            logger.info(
                f"inkfox: threads={info.get('threads')} version={info.get('version')} simd={info.get('simd_supported')}"
            )
        except Exception as e:  # pragma: no cover
            logger.debug(f"获取系统信息失败: {e}")

    # ---- 关键帧提取 ----
    async def extract_keyframes(self, video_path: str) -> List[Tuple[str, float]]:
        """提取关键帧并返回 (base64, timestamp_seconds) 列表"""
        with tempfile.TemporaryDirectory() as tmp:
            result = video.extract_keyframes_from_video(  # type: ignore[attr-defined]
                video_path=video_path,
                output_dir=tmp,
                max_keyframes=self.max_frames * 2,  # 先多抓一点再截断
                max_save=self.max_frames,
                ffmpeg_path=self.ffmpeg_path,
                use_simd=self.use_simd,
                threads=self.threads,
                verbose=False,
            )
            files = sorted(Path(tmp).glob("keyframe_*.jpg"))[: self.max_frames]
            total_ms = getattr(result, "total_time_ms", 0)
            frames: List[Tuple[str, float]] = []
            for i, f in enumerate(files):
                img = Image.open(f).convert("RGB")
                if max(img.size) > self.max_image_size:
                    scale = self.max_image_size / max(img.size)
                    img = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=self.frame_quality)
                b64 = base64.b64encode(buf.getvalue()).decode()
                ts = (i / max(1, len(files) - 1)) * (total_ms / 1000.0) if total_ms else float(i)
                frames.append((b64, ts))
            return frames

    # ---- 批量分析 ----
    async def _analyze_batch(self, frames: List[Tuple[str, float]], question: Optional[str]) -> str:
        from src.llm_models.payload_content.message import MessageBuilder, RoleType
        from src.llm_models.utils_model import RequestType
        prompt = self.batch_analysis_prompt.format(
            personality_core=self.personality_core, personality_side=self.personality_side
        )
        if question:
            prompt += f"\n用户关注: {question}"
        desc = [
            (f"第{i+1}帧 (时间: {ts:.2f}s)" if self.enable_frame_timing else f"第{i+1}帧")
            for i, (_b, ts) in enumerate(frames)
        ]
        prompt += "\n帧列表: " + ", ".join(desc)
        mb = MessageBuilder().set_role(RoleType.User).add_text_content(prompt)
        for b64, _ in frames:
            mb.add_image_content("jpeg", b64)
        message = mb.build()
        model_info, api_provider, client = self.video_llm._select_model()
        resp = await self.video_llm._execute_request(
            api_provider=api_provider,
            client=client,
            request_type=RequestType.RESPONSE,
            model_info=model_info,
            message_list=[message],
            temperature=None,
            max_tokens=None,
        )
        return resp.content or "❌ 未获得响应"

    # ---- 逐帧分析 ----
    async def _analyze_sequential(self, frames: List[Tuple[str, float]], question: Optional[str]) -> str:
        results: List[str] = []
        for i, (b64, ts) in enumerate(frames):
            prompt = f"分析第{i+1}帧" + (f" (时间: {ts:.2f}s)" if self.enable_frame_timing else "")
            if question:
                prompt += f"\n关注: {question}"
            try:
                text, _ = await self.video_llm.generate_response_for_image(
                    prompt=prompt, image_base64=b64, image_format="jpeg"
                )
                results.append(f"第{i+1}帧: {text}")
            except Exception as e:  # pragma: no cover
                results.append(f"第{i+1}帧: 失败 {e}")
            if i < len(frames) - 1:
                await asyncio.sleep(self.frame_analysis_delay)
        summary_prompt = "基于以下逐帧结果给出完整总结:\n\n" + "\n".join(results)
        try:
            final, _ = await self.video_llm.generate_response_for_image(
                prompt=summary_prompt, image_base64=frames[-1][0], image_format="jpeg"
            )
            return final
        except Exception:  # pragma: no cover
            return "\n".join(results)

    # ---- 主入口 ----
    async def analyze_video(self, video_path: str, question: Optional[str] = None) -> Tuple[bool, str]:
        if not os.path.exists(video_path):
            return False, "❌ 文件不存在"
        frames = await self.extract_keyframes(video_path)
        if not frames:
            return False, "❌ 未提取到关键帧"
        mode = self.analysis_mode
        if mode == "auto":
            mode = "batch" if len(frames) <= 20 else "sequential"
        text = await (self._analyze_batch(frames, question) if mode == "batch" else self._analyze_sequential(frames, question))
        return True, text

    async def analyze_video_from_bytes(
        self,
        video_bytes: bytes,
        filename: Optional[str] = None,
        prompt: Optional[str] = None,
        question: Optional[str] = None,
    ) -> Dict[str, str]:
        """从内存字节分析视频，兼容旧调用 (prompt / question 二选一) 返回 {"summary": str}."""
        if not video_bytes:
            return {"summary": "❌ 空视频数据"}
        # 兼容参数：prompt 优先，其次 question
        q = prompt if prompt is not None else question
        video_hash = hashlib.sha256(video_bytes).hexdigest()

        # 查缓存（第一次，未加锁）
        cached = await self._get_cached(video_hash)
        if cached:
            logger.info(f"视频缓存命中(预检查) hash={video_hash[:16]}")
            return {"summary": cached}

        # 获取锁避免重复处理
        async with _locks_guard:
            lock = _video_locks.get(video_hash)
            if lock is None:
                lock = asyncio.Lock()
                _video_locks[video_hash] = lock
        async with lock:
            # 双检缓存
            cached2 = await self._get_cached(video_hash)
            if cached2:
                logger.info(f"视频缓存命中(锁后) hash={video_hash[:16]}")
                return {"summary": cached2}

            try:
                with tempfile.NamedTemporaryFile(delete=False) as fp:
                    fp.write(video_bytes)
                    temp_path = fp.name
                try:
                    ok, summary = await self.analyze_video(temp_path, q)
                    # 写入缓存（仅成功）
                    if ok:
                        await self._save_cache(video_hash, summary, len(video_bytes))
                    return {"summary": summary}
                finally:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)

                # 保存分析结果到数据库（仅保存成功的结果）
                if success and not result.startswith("❌"):
                    metadata = {"filename": filename, "file_size": len(video_bytes), "analysis_timestamp": time.time()}
                    self._store_video_result(video_hash=video_hash, description=result, metadata=metadata)
                    logger.info("✅ 分析结果已保存到数据库")
                else:
                    logger.warning("⚠️ 分析失败，不保存到数据库以便后续重试")

                # 处理完成，通知等待者并清理资源
                video_event.set()
                async with video_lock_manager:
                    # 清理资源
                    video_locks.pop(video_hash, None)
                    video_events.pop(video_hash, None)

                return {"summary": result}

        except Exception as e:
            error_msg = f"❌ 从字节数据分析视频失败: {str(e)}"
            logger.error(error_msg)

    async def _save_cache(self, video_hash: str, summary: str, file_size: int) -> None:
        try:
            async with get_db_session() as session:  # type: ignore
                stmt = insert(Videos).values(  # type: ignore
                    video_id="",
                    video_hash=video_hash,
                    description=summary,
                    count=1,
                    timestamp=time.time(),
                    vlm_processed=True,
                    duration=None,
                    frame_count=None,
                    fps=None,
                    resolution=None,
                    file_size=file_size,
                )
                try:
                    await session.execute(stmt)
                    await session.commit()
                    logger.debug(f"视频缓存写入 success hash={video_hash}")
                except sa_exc.IntegrityError:  # 可能并发已写入
                    await session.rollback()
                    logger.debug(f"视频缓存已存在 hash={video_hash}")
        except Exception:  # pragma: no cover
                logger.debug("视频缓存写入失败")


# ---- 外部接口 ----
_INSTANCE: Optional[VideoAnalyzer] = None


def get_video_analyzer() -> VideoAnalyzer:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = VideoAnalyzer()
    return _INSTANCE


def is_video_analysis_available() -> bool:
    return True


def get_video_analysis_status() -> Dict[str, Any]:
    try:
        info = video.get_system_info()  # type: ignore[attr-defined]
    except Exception as e:  # pragma: no cover
        return {"available": False, "error": str(e)}
    inst = get_video_analyzer()
    return {
        "available": True,
        "system": info,
        "modes": ["auto", "batch", "sequential"],
        "max_frames_default": inst.max_frames,
        "implementation": "inkfox",
    }
