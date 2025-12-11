"""运行时语义兴趣度评分器

在线推理时使用，提供快速的兴趣度评分
"""

import asyncio
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.common.logger import get_logger
from src.chat.semantic_interest.features_tfidf import TfidfFeatureExtractor
from src.chat.semantic_interest.model_lr import SemanticInterestModel

logger = get_logger("semantic_interest.scorer")


class SemanticInterestScorer:
    """语义兴趣度评分器
    
    加载训练好的模型，在运行时快速计算消息的语义兴趣度
    """

    def __init__(self, model_path: str | Path):
        """初始化评分器
        
        Args:
            model_path: 模型文件路径 (.pkl)
        """
        self.model_path = Path(model_path)
        self.vectorizer: TfidfFeatureExtractor | None = None
        self.model: SemanticInterestModel | None = None
        self.meta: dict[str, Any] = {}
        self.is_loaded = False

        # 统计信息
        self.total_scores = 0
        self.total_time = 0.0

    def load(self):
        """加载模型"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        logger.info(f"开始加载模型: {self.model_path}")
        start_time = time.time()

        try:
            bundle = joblib.load(self.model_path)

            self.vectorizer = bundle["vectorizer"]
            self.model = bundle["model"]
            self.meta = bundle.get("meta", {})

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

    def reload(self):
        """重新加载模型（热更新）"""
        logger.info("重新加载模型...")
        self.is_loaded = False
        self.load()

    def score(self, text: str) -> float:
        """计算单条消息的语义兴趣度
        
        Args:
            text: 消息文本
            
        Returns:
            兴趣分 [0.0, 1.0]，越高表示越感兴趣
        """
        if not self.is_loaded:
            raise ValueError("模型尚未加载，请先调用 load() 方法")

        start_time = time.time()

        try:
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

    async def score_async(self, text: str) -> float:
        """异步计算兴趣度
        
        Args:
            text: 消息文本
            
        Returns:
            兴趣分 [0.0, 1.0]
        """
        # 在线程池中执行，避免阻塞事件循环
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.score, text)

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

        return {
            "is_loaded": self.is_loaded,
            "model_path": str(self.model_path),
            "total_scores": self.total_scores,
            "total_time": self.total_time,
            "avg_score_time": avg_time,
            "vocabulary_size": (
                self.vectorizer.get_vocabulary_size()
                if self.vectorizer and self.is_loaded
                else 0
            ),
            "meta": self.meta,
        }

    def __repr__(self) -> str:
        return (
            f"SemanticInterestScorer("
            f"loaded={self.is_loaded}, "
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

    async def load_model(self, version: str = "latest", persona_info: dict[str, Any] | None = None) -> SemanticInterestScorer:
        """加载指定版本的模型，支持人设感知
        
        Args:
            version: 模型版本号或 "latest" 或 "auto"
            persona_info: 人设信息，用于自动选择匹配的模型
            
        Returns:
            评分器实例
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

            scorer = SemanticInterestScorer(model_path)
            scorer.load()

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
            self.current_scorer.reload()
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
                max_samples=500,
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
        try:
            from src.chat.semantic_interest.auto_trainer import get_auto_trainer
            
            if self._auto_trainer is None:
                self._auto_trainer = get_auto_trainer()
            
            logger.info(f"[模型管理器] 启动自动训练任务，间隔: {interval_hours}小时")
            
            # 在后台任务中运行
            asyncio.create_task(
                self._auto_trainer.scheduled_train(persona_info, interval_hours)
            )
            
        except Exception as e:
            logger.error(f"[模型管理器] 启动自动训练失败: {e}")
