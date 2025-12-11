"""语义兴趣度计算模块

基于 TF-IDF + Logistic Regression 的语义兴趣度计算系统
支持人设感知的自动训练和模型切换
"""

from .auto_trainer import AutoTrainer, get_auto_trainer
from .dataset import DatasetGenerator, generate_training_dataset
from .features_tfidf import TfidfFeatureExtractor
from .model_lr import SemanticInterestModel, train_semantic_model
from .runtime_scorer import ModelManager, SemanticInterestScorer
from .trainer import SemanticInterestTrainer

__all__ = [
    # 运行时评分
    "SemanticInterestScorer",
    "ModelManager",
    # 训练组件
    "TfidfFeatureExtractor",
    "SemanticInterestModel",
    "train_semantic_model",
    # 数据集生成
    "DatasetGenerator",
    "generate_training_dataset",
    # 训练器
    "SemanticInterestTrainer",
    # 自动训练
    "AutoTrainer",
    "get_auto_trainer",
]
