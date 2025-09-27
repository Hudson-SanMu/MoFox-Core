"""
兴趣度系统模块
提供统一、稳定的消息兴趣度计算和管理功能
"""

from .interest_manager import (
    InterestManager,
    InterestSourceType,
    InterestFactor,
    InterestCalculator,
    MessageContentInterestCalculator,
    TopicInterestCalculator,
    UserInteractionInterestCalculator,
    interest_manager
)
from .bot_interest_manager import BotInterestManager, bot_interest_manager
from src.common.data_models.bot_interest_data_model import BotInterestTag, BotPersonalityInterests, InterestMatchResult

__all__ = [
    "InterestManager",
    "InterestSourceType",
    "InterestFactor",
    "InterestCalculator",
    "MessageContentInterestCalculator",
    "TopicInterestCalculator",
    "UserInteractionInterestCalculator",
    "interest_manager",
    "BotInterestManager",
    "bot_interest_manager",
    "BotInterestTag",
    "BotPersonalityInterests",
    "InterestMatchResult",
]
