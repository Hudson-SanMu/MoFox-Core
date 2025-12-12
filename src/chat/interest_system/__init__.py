"""
兴趣度系统模块
目前仅保留兴趣计算器管理入口
"""

from src.common.data_models.bot_interest_data_model import InterestMatchResult

from .interest_manager import InterestManager, get_interest_manager

__all__ = [
    # 消息兴趣值计算管理
    "InterestManager",
    "InterestMatchResult",
    "get_interest_manager",
]
