"""自动训练调度器

监控人设变化，自动触发模型训练和切换
"""

import asyncio
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from src.common.logger import get_logger
from src.config.config import global_config
from src.chat.semantic_interest.trainer import SemanticInterestTrainer

logger = get_logger("semantic_interest.auto_trainer")


class AutoTrainer:
    """自动训练调度器
    
    功能：
    1. 监控人设变化
    2. 自动构建训练数据集
    3. 定期重新训练模型
    4. 管理多个人设的模型
    """

    def __init__(
        self,
        data_dir: Path | None = None,
        model_dir: Path | None = None,
        min_train_interval_hours: int = 24,  # 最小训练间隔（小时）
        min_samples_for_training: int = 100,  # 最小训练样本数
    ):
        """初始化自动训练器
        
        Args:
            data_dir: 数据集目录
            model_dir: 模型目录
            min_train_interval_hours: 最小训练间隔（小时）
            min_samples_for_training: 触发训练的最小样本数
        """
        self.data_dir = Path(data_dir or "data/semantic_interest/datasets")
        self.model_dir = Path(model_dir or "data/semantic_interest/models")
        self.min_train_interval = timedelta(hours=min_train_interval_hours)
        self.min_samples = min_samples_for_training

        # 人设状态缓存
        self.persona_cache_file = self.data_dir / "persona_cache.json"
        self.last_persona_hash: str | None = None
        self.last_train_time: datetime | None = None

        # 训练器实例
        self.trainer = SemanticInterestTrainer(
            data_dir=self.data_dir,
            model_dir=self.model_dir,
        )

        # 确保目录存在
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # 加载缓存的人设状态
        self._load_persona_cache()
        
        # 定时任务标志（防止重复启动）
        self._scheduled_task_running = False
        self._scheduled_task = None

        logger.info("[自动训练器] 初始化完成")
        logger.info(f"  - 数据目录: {self.data_dir}")
        logger.info(f"  - 模型目录: {self.model_dir}")
        logger.info(f"  - 最小训练间隔: {min_train_interval_hours}小时")

    def _load_persona_cache(self):
        """加载缓存的人设状态"""
        if self.persona_cache_file.exists():
            try:
                with open(self.persona_cache_file, "r", encoding="utf-8") as f:
                    cache = json.load(f)
                    self.last_persona_hash = cache.get("persona_hash")
                    last_train_str = cache.get("last_train_time")
                    if last_train_str:
                        self.last_train_time = datetime.fromisoformat(last_train_str)
                logger.info(f"[自动训练器] 加载人设缓存: hash={self.last_persona_hash[:8] if self.last_persona_hash else 'None'}")
            except Exception as e:
                logger.warning(f"[自动训练器] 加载人设缓存失败: {e}")

    def _save_persona_cache(self, persona_hash: str):
        """保存人设状态到缓存"""
        cache = {
            "persona_hash": persona_hash,
            "last_train_time": datetime.now().isoformat(),
        }
        try:
            with open(self.persona_cache_file, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
            logger.debug(f"[自动训练器] 保存人设缓存: hash={persona_hash[:8]}")
        except Exception as e:
            logger.error(f"[自动训练器] 保存人设缓存失败: {e}")

    def _calculate_persona_hash(self, persona_info: dict[str, Any]) -> str:
        """计算人设信息的哈希值
        
        Args:
            persona_info: 人设信息字典
            
        Returns:
            SHA256 哈希值
        """
        # 只关注影响模型的关键字段
        key_fields = {
            "name": persona_info.get("name", ""),
            "interests": sorted(persona_info.get("interests", [])),
            "dislikes": sorted(persona_info.get("dislikes", [])),
            "personality": persona_info.get("personality", ""),
        }
        
        # 转为JSON并计算哈希
        json_str = json.dumps(key_fields, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def check_persona_changed(self, persona_info: dict[str, Any]) -> bool:
        """检查人设是否发生变化
        
        Args:
            persona_info: 当前人设信息
            
        Returns:
            True 如果人设发生变化
        """
        current_hash = self._calculate_persona_hash(persona_info)
        
        if self.last_persona_hash is None:
            logger.info("[自动训练器] 首次检测人设")
            return True
            
        if current_hash != self.last_persona_hash:
            logger.info(f"[自动训练器] 检测到人设变化")
            logger.info(f"  - 旧哈希: {self.last_persona_hash[:8]}")
            logger.info(f"  - 新哈希: {current_hash[:8]}")
            return True
            
        return False

    def should_train(self, persona_info: dict[str, Any], force: bool = False) -> tuple[bool, str]:
        """判断是否应该训练模型
        
        Args:
            persona_info: 人设信息
            force: 强制训练
            
        Returns:
            (是否应该训练, 原因说明)
        """
        # 强制训练
        if force:
            return True, "强制训练"

        # 检查人设是否变化
        persona_changed = self.check_persona_changed(persona_info)
        if persona_changed:
            return True, "人设发生变化"

        # 检查训练间隔
        if self.last_train_time is None:
            return True, "从未训练过"

        time_since_last_train = datetime.now() - self.last_train_time
        if time_since_last_train >= self.min_train_interval:
            return True, f"距上次训练已{time_since_last_train.total_seconds() / 3600:.1f}小时"

        return False, "无需训练"

    async def auto_train_if_needed(
        self,
        persona_info: dict[str, Any],
        days: int = 7,
        max_samples: int = 500,
        force: bool = False,
    ) -> tuple[bool, Path | None]:
        """自动训练（如果需要）
        
        Args:
            persona_info: 人设信息
            days: 采样天数
            max_samples: 最大采样数
            force: 强制训练
            
        Returns:
            (是否训练了, 模型路径)
        """
        # 检查是否需要训练
        should_train, reason = self.should_train(persona_info, force)
        
        if not should_train:
            logger.debug(f"[自动训练器] {reason}，跳过训练")
            return False, None

        logger.info(f"[自动训练器] 开始自动训练: {reason}")

        try:
            # 计算人设哈希作为版本标识
            persona_hash = self._calculate_persona_hash(persona_info)
            model_version = f"auto_{persona_hash[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # 执行训练
            dataset_path, model_path, metrics = await self.trainer.full_training_pipeline(
                persona_info=persona_info,
                days=days,
                max_samples=max_samples,
                model_version=model_version,
                tfidf_config={
                    "analyzer": "char",
                    "ngram_range": (2, 4),
                    "max_features": 10000,
                    "min_df": 3,
                },
                model_config={
                    "class_weight": "balanced",
                    "max_iter": 1000,
                },
            )

            # 更新缓存
            self.last_persona_hash = persona_hash
            self.last_train_time = datetime.now()
            self._save_persona_cache(persona_hash)

            # 创建"latest"符号链接
            self._create_latest_link(model_path)

            logger.info(f"[自动训练器] 训练完成!")
            logger.info(f"  - 模型: {model_path.name}")
            logger.info(f"  - 准确率: {metrics.get('test_accuracy', 0):.4f}")

            return True, model_path

        except Exception as e:
            logger.error(f"[自动训练器] 训练失败: {e}")
            import traceback
            traceback.print_exc()
            return False, None

    def _create_latest_link(self, model_path: Path):
        """创建指向最新模型的符号链接
        
        Args:
            model_path: 模型文件路径
        """
        latest_path = self.model_dir / "semantic_interest_latest.pkl"
        
        try:
            # 删除旧链接
            if latest_path.exists() or latest_path.is_symlink():
                latest_path.unlink()
            
            # 创建新链接（Windows 需要管理员权限，使用复制代替）
            import shutil
            shutil.copy2(model_path, latest_path)
            
            logger.info(f"[自动训练器] 已更新 latest 模型")
            
        except Exception as e:
            logger.warning(f"[自动训练器] 创建 latest 链接失败: {e}")

    async def scheduled_train(
        self,
        persona_info: dict[str, Any],
        interval_hours: int = 24,
    ):
        """定时训练任务
        
        Args:
            persona_info: 人设信息
            interval_hours: 检查间隔（小时）
        """
        # 检查是否已经有任务在运行
        if self._scheduled_task_running:
            logger.debug(f"[自动训练器] 定时任务已在运行，跳过")
            return
        
        self._scheduled_task_running = True
        logger.info(f"[自动训练器] 启动定时训练任务，间隔: {interval_hours}小时")

        while True:
            try:
                # 检查并训练
                trained, model_path = await self.auto_train_if_needed(persona_info)
                
                if trained:
                    logger.info(f"[自动训练器] 定时训练完成: {model_path}")
                
                # 等待下次检查
                await asyncio.sleep(interval_hours * 3600)
                
            except Exception as e:
                logger.error(f"[自动训练器] 定时训练出错: {e}")
                # 出错后等待较短时间再试
                await asyncio.sleep(300)  # 5分钟

    def get_model_for_persona(self, persona_info: dict[str, Any]) -> Path | None:
        """获取当前人设对应的模型
        
        Args:
            persona_info: 人设信息
            
        Returns:
            模型文件路径，如果不存在则返回 None
        """
        persona_hash = self._calculate_persona_hash(persona_info)
        
        # 查找匹配的模型
        pattern = f"semantic_interest_auto_{persona_hash[:8]}_*.pkl"
        matching_models = list(self.model_dir.glob(pattern))
        
        if matching_models:
            # 返回最新的
            latest = max(matching_models, key=lambda p: p.stat().st_mtime)
            logger.debug(f"[自动训练器] 找到人设模型: {latest.name}")
            return latest
        
        # 没有找到，返回 latest
        latest_path = self.model_dir / "semantic_interest_latest.pkl"
        if latest_path.exists():
            logger.debug(f"[自动训练器] 使用 latest 模型")
            return latest_path
            
        logger.warning(f"[自动训练器] 未找到可用模型")
        return None

    def cleanup_old_models(self, keep_count: int = 5):
        """清理旧模型文件
        
        Args:
            keep_count: 保留最新的 N 个模型
        """
        try:
            # 获取所有自动训练的模型
            all_models = list(self.model_dir.glob("semantic_interest_auto_*.pkl"))
            
            if len(all_models) <= keep_count:
                return
            
            # 按修改时间排序
            all_models.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            
            # 删除旧模型
            for old_model in all_models[keep_count:]:
                old_model.unlink()
                logger.info(f"[自动训练器] 清理旧模型: {old_model.name}")
                
            logger.info(f"[自动训练器] 模型清理完成，保留 {keep_count} 个")
            
        except Exception as e:
            logger.error(f"[自动训练器] 清理模型失败: {e}")


# 全局单例
_auto_trainer: AutoTrainer | None = None


def get_auto_trainer() -> AutoTrainer:
    """获取自动训练器单例"""
    global _auto_trainer
    if _auto_trainer is None:
        _auto_trainer = AutoTrainer()
    return _auto_trainer
