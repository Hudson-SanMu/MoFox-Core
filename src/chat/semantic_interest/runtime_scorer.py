"""运行时语义兴趣度评分器

在线推理时使用，提供快速的兴趣度评分
支持异步加载、超时保护、批量优化、模型预热

2024.12 优化更新：
- 新增 FastScorer 模式，绕过 sklearn 直接使用 token→weight 字典
- 全局线程池避免每次创建新的 executor
- 可选的批处理队列模式
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.common.logger import get_logger
from src.chat.semantic_interest.features_tfidf import TfidfFeatureExtractor
from src.chat.semantic_interest.model_lr import SemanticInterestModel

logger = get_logger("semantic_interest.scorer")

# 全局配置
DEFAULT_SCORE_TIMEOUT = 2.0  # 评分超时（秒），从 5.0 降低到 2.0

# 全局线程池（避免每次创建新的 executor）
_GLOBAL_EXECUTOR: ThreadPoolExecutor | None = None
_EXECUTOR_MAX_WORKERS = 4


def _get_global_executor() -> ThreadPoolExecutor:
    """获取全局线程池（单例）"""
    global _GLOBAL_EXECUTOR
    if _GLOBAL_EXECUTOR is None:
        _GLOBAL_EXECUTOR = ThreadPoolExecutor(
            max_workers=_EXECUTOR_MAX_WORKERS,
            thread_name_prefix="semantic_scorer"
        )
        logger.info(f"[评分器] 创建全局线程池，workers={_EXECUTOR_MAX_WORKERS}")
    return _GLOBAL_EXECUTOR


# 单例管理
_scorer_instances: dict[str, "SemanticInterestScorer"] = {}  # 模型路径 -> 评分器实例
_instance_lock = asyncio.Lock()  # 创建实例的锁


class SemanticInterestScorer:
    """语义兴趣度评分器
    
    加载训练好的模型，在运行时快速计算消息的语义兴趣度
    优化特性：
    - 异步加载支持（非阻塞）
    - 批量评分优化
    - 超时保护
    - 模型预热
    - 全局线程池（避免重复创建 executor）
    - 可选的 FastScorer 模式（绕过 sklearn）
    """

    def __init__(self, model_path: str | Path, use_fast_scorer: bool = True):
        """初始化评分器
        
        Args:
            model_path: 模型文件路径 (.pkl)
            use_fast_scorer: 是否使用快速评分器模式（推荐）
        """
        self.model_path = Path(model_path)
        self.vectorizer: TfidfFeatureExtractor | None = None
        self.model: SemanticInterestModel | None = None
        self.meta: dict[str, Any] = {}
        self.is_loaded = False
        
        # 快速评分器模式
        self._use_fast_scorer = use_fast_scorer
        self._fast_scorer = None  # FastScorer 实例

        # 统计信息
        self.total_scores = 0
        self.total_time = 0.0

    def load(self):
        """同步加载模型（阻塞）"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        logger.info(f"开始加载模型: {self.model_path}")
        start_time = time.time()

        try:
            bundle = joblib.load(self.model_path)

            self.vectorizer = bundle["vectorizer"]
            self.model = bundle["model"]
            self.meta = bundle.get("meta", {})

            # 如果启用快速评分器模式，创建 FastScorer
            if self._use_fast_scorer:
                from src.chat.semantic_interest.optimized_scorer import FastScorer, FastScorerConfig
                
                config = FastScorerConfig(
                    ngram_range=self.vectorizer.get_config().get("ngram_range", (2, 3)),
                    weight_prune_threshold=1e-4,
                )
                self._fast_scorer = FastScorer.from_sklearn_model(
                    self.vectorizer, self.model, config
                )
                logger.info(
                    f"[FastScorer] 已启用，词表从 {self.vectorizer.get_vocabulary_size()} "
                    f"剪枝到 {len(self._fast_scorer.token_weights)}"
                )

            self.is_loaded = True
            load_time = time.time() - start_time

            logger.info(
                f"模型加载成功，耗时: {load_time:.3f}秒, "
                f"词表大小: {self.vectorizer.get_vocabulary_size()}"  # type: ignore
            )

            if self.meta:
                logger.info(f"模型元信息: {self.meta}")

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    async def load_async(self):
        """异步加载模型（非阻塞）"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        logger.info(f"开始异步加载模型: {self.model_path}")
        start_time = time.time()

        try:
            # 在全局线程池中执行 I/O 密集型操作
            executor = _get_global_executor()
            loop = asyncio.get_running_loop()
            bundle = await loop.run_in_executor(executor, joblib.load, self.model_path)

            self.vectorizer = bundle["vectorizer"]
            self.model = bundle["model"]
            self.meta = bundle.get("meta", {})

            # 如果启用快速评分器模式，创建 FastScorer
            if self._use_fast_scorer:
                from src.chat.semantic_interest.optimized_scorer import FastScorer, FastScorerConfig
                
                config = FastScorerConfig(
                    ngram_range=self.vectorizer.get_config().get("ngram_range", (2, 3)),
                    weight_prune_threshold=1e-4,
                )
                self._fast_scorer = FastScorer.from_sklearn_model(
                    self.vectorizer, self.model, config
                )
                logger.info(
                    f"[FastScorer] 已启用，词表从 {self.vectorizer.get_vocabulary_size()} "
                    f"剪枝到 {len(self._fast_scorer.token_weights)}"
                )

            self.is_loaded = True
            load_time = time.time() - start_time

            logger.info(
                f"模型异步加载成功，耗时: {load_time:.3f}秒, "
                f"词表大小: {self.vectorizer.get_vocabulary_size()}"  # type: ignore
            )

            if self.meta:
                logger.info(f"模型元信息: {self.meta}")
            
            # 预热模型
            await self._warmup_async()

        except Exception as e:
            logger.error(f"模型异步加载失败: {e}")
            raise

    def reload(self):
        """重新加载模型（热更新）"""
        logger.info("重新加载模型...")
        self.is_loaded = False
        self.load()
    
    async def reload_async(self):
        """异步重新加载模型"""
        logger.info("异步重新加载模型...")
        self.is_loaded = False
        await self.load_async()

    def score(self, text: str) -> float:
        """计算单条消息的语义兴趣度
        
        Args:
            text: 消息文本
            
        Returns:
            兴趣分 [0.0, 1.0]，越高表示越感兴趣
        """
        if not self.is_loaded:
            raise ValueError("模型尚未加载，请先调用 load() 或 load_async() 方法")

        start_time = time.time()

        try:
            # 优先使用 FastScorer（绕过 sklearn，更快）
            if self._fast_scorer is not None:
                interest = self._fast_scorer.score(text)
            else:
                # 回退到原始 sklearn 路径
                # 向量化
                X = self.vectorizer.transform([text])

                # 预测概率
                proba = self.model.predict_proba(X)[0]

                # proba 顺序为 [-1, 0, 1]
                p_neg, p_neu, p_pos = proba

                # 兴趣分计算策略：
                # interest = P(1) + 0.5 * P(0)
                # 这样：纯正向(1)=1.0, 纯中立(0)=0.5, 纯负向(-1)=0.0
                interest = float(p_pos + 0.5 * p_neu)

            # 确保在 [0, 1] 范围内
            interest = max(0.0, min(1.0, interest))

            # 统计
            self.total_scores += 1
            self.total_time += time.time() - start_time

            return interest

        except Exception as e:
            logger.error(f"兴趣度计算失败: {e}, 消息: {text[:50]}")
            return 0.5  # 默认返回中立值

    async def score_async(self, text: str, timeout: float = DEFAULT_SCORE_TIMEOUT) -> float:
        """异步计算兴趣度（带超时保护）
        
        Args:
            text: 消息文本
            timeout: 超时时间（秒），超时返回中立值 0.5
            
        Returns:
            兴趣分 [0.0, 1.0]
        """
        # 使用全局线程池，避免每次创建新的 executor
        executor = _get_global_executor()
        loop = asyncio.get_running_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(executor, self.score, text),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"兴趣度计算超时（{timeout}秒），消息: {text[:50]}")
            return 0.5  # 默认中立值

    def score_batch(self, texts: list[str]) -> list[float]:
        """批量计算兴趣度
        
        Args:
            texts: 消息文本列表
            
        Returns:
            兴趣分列表
        """
        if not self.is_loaded:
            raise ValueError("模型尚未加载")

        if not texts:
            return []

        start_time = time.time()

        try:
            # 优先使用 FastScorer
            if self._fast_scorer is not None:
                interests = self._fast_scorer.score_batch(texts)
                
                # 统计
                self.total_scores += len(texts)
                self.total_time += time.time() - start_time
                return interests
            else:
                # 回退到原始 sklearn 路径
                # 批量向量化
                X = self.vectorizer.transform(texts)

                # 批量预测
                proba = self.model.predict_proba(X)

                # 计算兴趣分
                interests = []
                for p_neg, p_neu, p_pos in proba:
                    interest = float(p_pos + 0.5 * p_neu)
                    interest = max(0.0, min(1.0, interest))
                    interests.append(interest)

                # 统计
                self.total_scores += len(texts)
                self.total_time += time.time() - start_time

                return interests

        except Exception as e:
            logger.error(f"批量兴趣度计算失败: {e}")
            return [0.5] * len(texts)

    async def score_batch_async(self, texts: list[str], timeout: float | None = None) -> list[float]:
        """异步批量计算兴趣度
        
        Args:
            texts: 消息文本列表
            timeout: 超时时间（秒），None 则使用单条超时*文本数
            
        Returns:
            兴趣分列表
        """
        if not texts:
            return []
        
        # 计算动态超时
        if timeout is None:
            timeout = DEFAULT_SCORE_TIMEOUT * len(texts)
        
        # 使用全局线程池
        executor = _get_global_executor()
        loop = asyncio.get_running_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(executor, self.score_batch, texts),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"批量兴趣度计算超时（{timeout}秒），批次大小: {len(texts)}")
            return [0.5] * len(texts)
    
    def _warmup(self, sample_texts: list[str] | None = None):
        """预热模型（执行几次推理以优化性能）
        
        Args:
            sample_texts: 预热用的样本文本，None 则使用默认样本
        """
        if not self.is_loaded:
            return
        
        if sample_texts is None:
            sample_texts = [
                "你好",
                "今天天气怎么样？",
                "我对这个话题很感兴趣"
            ]
        
        logger.debug(f"开始预热模型，样本数: {len(sample_texts)}")
        start_time = time.time()
        
        for text in sample_texts:
            try:
                self.score(text)
            except Exception:
                pass  # 忽略预热错误
        
        warmup_time = time.time() - start_time
        logger.debug(f"模型预热完成，耗时: {warmup_time:.3f}秒")
    
    async def _warmup_async(self, sample_texts: list[str] | None = None):
        """异步预热模型"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._warmup, sample_texts)

    def get_detailed_score(self, text: str) -> dict[str, Any]:
        """获取详细的兴趣度评分信息
        
        Args:
            text: 消息文本
            
        Returns:
            包含概率分布和最终分数的详细信息
        """
        if not self.is_loaded:
            raise ValueError("模型尚未加载")

        X = self.vectorizer.transform([text])
        proba = self.model.predict_proba(X)[0]
        pred_label = self.model.predict(X)[0]

        p_neg, p_neu, p_pos = proba
        interest = float(p_pos + 0.5 * p_neu)

        return {
            "interest_score": max(0.0, min(1.0, interest)),
            "proba_distribution": {
                "dislike": float(p_neg),
                "neutral": float(p_neu),
                "like": float(p_pos),
            },
            "predicted_label": int(pred_label),
            "text_preview": text[:100],
        }

    def get_statistics(self) -> dict[str, Any]:
        """获取评分器统计信息
        
        Returns:
            统计信息字典
        """
        avg_time = self.total_time / self.total_scores if self.total_scores > 0 else 0

        stats = {
            "is_loaded": self.is_loaded,
            "model_path": str(self.model_path),
            "total_scores": self.total_scores,
            "total_time": self.total_time,
            "avg_score_time": avg_time,
            "avg_score_time_ms": avg_time * 1000,  # 毫秒单位更直观
            "vocabulary_size": (
                self.vectorizer.get_vocabulary_size()
                if self.vectorizer and self.is_loaded
                else 0
            ),
            "use_fast_scorer": self._use_fast_scorer,
            "fast_scorer_enabled": self._fast_scorer is not None,
            "meta": self.meta,
        }
        
        # 如果启用了 FastScorer，添加其统计
        if self._fast_scorer is not None:
            stats["fast_scorer_stats"] = self._fast_scorer.get_statistics()
        
        return stats

    def __repr__(self) -> str:
        mode = "fast" if self._fast_scorer else "sklearn"
        return (
            f"SemanticInterestScorer("
            f"loaded={self.is_loaded}, "
            f"mode={mode}, "
            f"model={self.model_path.name})"
        )


class ModelManager:
    """模型管理器
    
    支持模型热更新、版本管理和人设感知的模型切换
    """

    def __init__(self, model_dir: Path):
        """初始化管理器
        
        Args:
            model_dir: 模型目录
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.current_scorer: SemanticInterestScorer | None = None
        self.current_version: str | None = None
        self.current_persona_info: dict[str, Any] | None = None
        self._lock = asyncio.Lock()
        
        # 自动训练器集成
        self._auto_trainer = None
        self._auto_training_started = False  # 防止重复启动自动训练

    async def load_model(self, version: str = "latest", persona_info: dict[str, Any] | None = None, use_async: bool = True) -> SemanticInterestScorer:
        """加载指定版本的模型，支持人设感知（使用单例）
        
        Args:
            version: 模型版本号或 "latest" 或 "auto"
            persona_info: 人设信息，用于自动选择匹配的模型
            use_async: 是否使用异步加载（推荐）
            
        Returns:
            评分器实例（单例）
        """
        async with self._lock:
            # 如果指定了人设信息，尝试使用自动训练器
            if persona_info is not None and version == "auto":
                model_path = await self._get_persona_model(persona_info)
            elif version == "latest":
                model_path = self._get_latest_model()
            else:
                model_path = self.model_dir / f"semantic_interest_{version}.pkl"

            if not model_path or not model_path.exists():
                raise FileNotFoundError(f"模型文件不存在: {model_path}")

            # 使用单例获取评分器
            scorer = await get_semantic_scorer(model_path, force_reload=False, use_async=use_async)
            
            self.current_scorer = scorer
            self.current_version = version
            self.current_persona_info = persona_info

            logger.info(f"模型管理器已加载版本: {version}, 文件: {model_path.name}")
            return scorer

    async def reload_current_model(self):
        """重新加载当前模型"""
        if not self.current_scorer:
            raise ValueError("尚未加载任何模型")

        async with self._lock:
            await self.current_scorer.reload_async()
            logger.info("模型已重新加载")

    def _get_latest_model(self) -> Path:
        """获取最新的模型文件
        
        Returns:
            最新模型文件路径
        """
        model_files = list(self.model_dir.glob("semantic_interest_*.pkl"))

        if not model_files:
            raise FileNotFoundError(f"在 {self.model_dir} 中未找到模型文件")

        # 按修改时间排序
        latest = max(model_files, key=lambda p: p.stat().st_mtime)
        return latest

    def get_scorer(self) -> SemanticInterestScorer:
        """获取当前评分器
        
        Returns:
            当前评分器实例
        """
        if not self.current_scorer:
            raise ValueError("尚未加载任何模型")

        return self.current_scorer

    async def _get_persona_model(self, persona_info: dict[str, Any]) -> Path | None:
        """根据人设信息获取或训练模型
        
        Args:
            persona_info: 人设信息
            
        Returns:
            模型文件路径
        """
        try:
            # 延迟导入避免循环依赖
            from src.chat.semantic_interest.auto_trainer import get_auto_trainer
            
            if self._auto_trainer is None:
                self._auto_trainer = get_auto_trainer()
            
            # 检查是否需要训练
            trained, model_path = await self._auto_trainer.auto_train_if_needed(
                persona_info=persona_info,
                days=7,
                max_samples=1000,  # 初始训练使用1000条消息
            )
            
            if trained and model_path:
                logger.info(f"[模型管理器] 使用新训练的模型: {model_path.name}")
                return model_path
            
            # 获取现有的人设模型
            model_path = self._auto_trainer.get_model_for_persona(persona_info)
            if model_path:
                return model_path
            
            # 降级到 latest
            logger.warning("[模型管理器] 未找到人设模型，使用 latest")
            return self._get_latest_model()
            
        except Exception as e:
            logger.error(f"[模型管理器] 获取人设模型失败: {e}")
            return self._get_latest_model()

    async def check_and_reload_for_persona(self, persona_info: dict[str, Any]) -> bool:
        """检查人设变化并重新加载模型
        
        Args:
            persona_info: 当前人设信息
            
        Returns:
            True 如果重新加载了模型
        """
        # 检查人设是否变化
        if self.current_persona_info == persona_info:
            return False
        
        logger.info("[模型管理器] 检测到人设变化，重新加载模型...")
        
        try:
            await self.load_model(version="auto", persona_info=persona_info)
            return True
        except Exception as e:
            logger.error(f"[模型管理器] 重新加载模型失败: {e}")
            return False

    async def start_auto_training(self, persona_info: dict[str, Any], interval_hours: int = 24):
        """启动自动训练任务
        
        Args:
            persona_info: 人设信息
            interval_hours: 检查间隔（小时）
        """
        # 使用锁防止并发启动
        async with self._lock:
            # 检查是否已经启动
            if self._auto_training_started:
                logger.debug(f"[模型管理器] 自动训练任务已启动，跳过")
                return
            
            try:
                from src.chat.semantic_interest.auto_trainer import get_auto_trainer
                
                if self._auto_trainer is None:
                    self._auto_trainer = get_auto_trainer()
                
                logger.info(f"[模型管理器] 启动自动训练任务，间隔: {interval_hours}小时")
                
                # 标记为已启动
                self._auto_training_started = True
                
                # 在后台任务中运行
                asyncio.create_task(
                    self._auto_trainer.scheduled_train(persona_info, interval_hours)
                )
                
            except Exception as e:
                logger.error(f"[模型管理器] 启动自动训练失败: {e}")
                self._auto_training_started = False  # 失败时重置标志


# 单例获取函数
async def get_semantic_scorer(
    model_path: str | Path,
    force_reload: bool = False,
    use_async: bool = True
) -> SemanticInterestScorer:
    """获取语义兴趣度评分器实例（单例模式）
    
    同一个模型路径只会创建一个评分器实例，避免重复加载模型。
    
    Args:
        model_path: 模型文件路径
        force_reload: 是否强制重新加载模型
        use_async: 是否使用异步加载（推荐）
        
    Returns:
        评分器实例（单例）
        
    Example:
        >>> scorer = await get_semantic_scorer("data/semantic_interest/models/model.pkl")
        >>> score = await scorer.score_async("今天天气真好")
    """
    model_path = Path(model_path)
    path_key = str(model_path.resolve())  # 使用绝对路径作为键
    
    async with _instance_lock:
        # 检查是否已存在实例
        if not force_reload and path_key in _scorer_instances:
            scorer = _scorer_instances[path_key]
            if scorer.is_loaded:
                logger.debug(f"[单例] 复用已加载的评分器: {model_path.name}")
                return scorer
            else:
                logger.info(f"[单例] 评分器未加载，重新加载: {model_path.name}")
        
        # 创建或重新加载实例
        if path_key not in _scorer_instances:
            logger.info(f"[单例] 创建新的评分器实例: {model_path.name}")
            scorer = SemanticInterestScorer(model_path)
            _scorer_instances[path_key] = scorer
        else:
            scorer = _scorer_instances[path_key]
            logger.info(f"[单例] 强制重新加载评分器: {model_path.name}")
        
        # 加载模型
        if use_async:
            await scorer.load_async()
        else:
            scorer.load()
        
        return scorer


def get_semantic_scorer_sync(
    model_path: str | Path,
    force_reload: bool = False
) -> SemanticInterestScorer:
    """获取语义兴趣度评分器实例（同步版本，单例模式）
    
    注意：这是同步版本，推荐使用异步版本 get_semantic_scorer()
    
    Args:
        model_path: 模型文件路径
        force_reload: 是否强制重新加载模型
        
    Returns:
        评分器实例（单例）
    """
    model_path = Path(model_path)
    path_key = str(model_path.resolve())
    
    # 检查是否已存在实例
    if not force_reload and path_key in _scorer_instances:
        scorer = _scorer_instances[path_key]
        if scorer.is_loaded:
            logger.debug(f"[单例] 复用已加载的评分器: {model_path.name}")
            return scorer
    
    # 创建或重新加载实例
    if path_key not in _scorer_instances:
        logger.info(f"[单例] 创建新的评分器实例: {model_path.name}")
        scorer = SemanticInterestScorer(model_path)
        _scorer_instances[path_key] = scorer
    else:
        scorer = _scorer_instances[path_key]
        logger.info(f"[单例] 强制重新加载评分器: {model_path.name}")
    
    # 加载模型
    scorer.load()
    return scorer


def clear_scorer_instances():
    """清空所有评分器实例（释放内存）"""
    global _scorer_instances
    count = len(_scorer_instances)
    _scorer_instances.clear()
    logger.info(f"[单例] 已清空 {count} 个评分器实例")


def get_all_scorer_instances() -> dict[str, SemanticInterestScorer]:
    """获取所有已创建的评分器实例
    
    Returns:
        {模型路径: 评分器实例} 的字典
    """
    return _scorer_instances.copy()
