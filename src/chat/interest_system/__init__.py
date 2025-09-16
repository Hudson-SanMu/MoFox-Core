"""
机器人兴趣标签系统
基于人设生成兴趣标签，使用embedding计算匹配度
"""

from .bot_interest_manager import BotInterestManager, bot_interest_manager
from src.common.data_models.bot_interest_data_model import (
    BotInterestTag, BotPersonalityInterests, InterestMatchResult
)

__all__ = [
    "BotInterestManager",
    "bot_interest_manager",
    "BotInterestTag",
    "BotPersonalityInterests",
    "InterestMatchResult"
]