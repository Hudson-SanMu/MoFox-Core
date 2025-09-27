"""
重构后的消息兴趣值计算系统
提供稳定、可靠的消息兴趣度计算和管理功能
"""

import time
from typing import Dict, List, Optional, Tuple, Any, Union, TypedDict
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

from src.common.logger import get_logger

logger = get_logger("interest_system")


class InterestSourceType(Enum):
    """兴趣度来源类型"""
    MESSAGE_CONTENT = "message_content"  # 消息内容
    USER_INTERACTION = "user_interaction"  # 用户交互
    TOPIC_RELEVANCE = "topic_relevance"  # 话题相关性
    RELATIONSHIP_SCORE = "relationship_score"  # 关系分数
    HISTORICAL_PATTERN = "historical_pattern"  # 历史模式


@dataclass
class InterestFactor:
    """兴趣度因子"""
    source_type: InterestSourceType
    value: float
    weight: float = 1.0
    decay_rate: float = 0.1  # 衰减率
    last_updated: float = field(default_factory=time.time)

    def get_current_value(self) -> float:
        """获取当前值（考虑时间衰减）"""
        age = time.time() - self.last_updated
        decay_factor = max(0.1, 1.0 - (age * self.decay_rate / (24 * 3600)))  # 按天衰减
        return self.value * decay_factor

    def update_value(self, new_value: float) -> None:
        """更新值"""
        self.value = max(0.0, min(1.0, new_value))
        self.last_updated = time.time()


class InterestCalculator(ABC):
    """兴趣度计算器抽象基类"""

    @abstractmethod
    def calculate(self, context: Dict[str, Any]) -> float:
        """计算兴趣度"""
        pass

    @abstractmethod
    def get_confidence(self) -> float:
        """获取计算置信度"""
        pass


class MessageData(TypedDict):
    """消息数据类型定义"""
    message_id: str
    processed_plain_text: str
    is_emoji: bool
    is_picid: bool
    is_mentioned: bool
    is_command: bool
    key_words: str
    user_id: str
    time: float


class InterestContext(TypedDict):
    """兴趣度计算上下文"""
    stream_id: str
    user_id: Optional[str]
    message: MessageData


class InterestResult(TypedDict):
    """兴趣度计算结果"""
    value: float
    confidence: float
    source_scores: Dict[InterestSourceType, float]
    cached: bool


class MessageContentInterestCalculator(InterestCalculator):
    """消息内容兴趣度计算器"""

    def calculate(self, context: Dict[str, Any]) -> float:
        """基于消息内容计算兴趣度"""
        message = context.get("message", {})
        if not message:
            return 0.3  # 默认值

        # 提取消息特征
        text_length = len(message.get("processed_plain_text", ""))
        has_emoji = message.get("is_emoji", False)
        has_image = message.get("is_picid", False)
        is_mentioned = message.get("is_mentioned", False)
        is_command = message.get("is_command", False)

        # 基础分数
        base_score = 0.3

        # 文本长度加权
        if text_length > 0:
            text_score = min(0.3, text_length / 200)  # 200字符为满分
            base_score += text_score * 0.3

        # 多媒体内容加权
        if has_emoji:
            base_score += 0.1
        if has_image:
            base_score += 0.2

        # 交互特征加权
        if is_mentioned:
            base_score += 0.2
        if is_command:
            base_score += 0.1

        return min(1.0, base_score)

    def get_confidence(self) -> float:
        return 0.8


class TopicInterestCalculator(InterestCalculator):
    """话题兴趣度计算器"""

    def __init__(self):
        self.topic_interests: Dict[str, float] = {}
        self.topic_decay_rate = 0.05  # 话题兴趣度衰减率

    def update_topic_interest(self, topic: str, interest_value: float):
        """更新话题兴趣度"""
        current_interest = self.topic_interests.get(topic, 0.3)
        # 平滑更新
        new_interest = current_interest * 0.7 + interest_value * 0.3
        self.topic_interests[topic] = max(0.0, min(1.0, new_interest))

        logger.debug(f"更新话题 '{topic}' 兴趣度: {current_interest:.3f} -> {new_interest:.3f}")

    def calculate(self, context: Dict[str, Any]) -> float:
        """基于话题相关性计算兴趣度"""
        message = context.get("message", {})
        keywords = message.get("key_words", "[]")

        try:
            import json
            keyword_list = json.loads(keywords) if keywords else []
        except (json.JSONDecodeError, TypeError):
            keyword_list = []

        if not keyword_list:
            return 0.4  # 无关键词时的默认值

        # 计算相关话题的平均兴趣度
        total_interest = 0.0
        relevant_topics = 0

        for keyword in keyword_list[:5]:  # 最多取前5个关键词
            # 查找相关话题
            for topic, interest in self.topic_interests.items():
                if keyword.lower() in topic.lower() or topic.lower() in keyword.lower():
                    total_interest += interest
                    relevant_topics += 1
                    break

        if relevant_topics > 0:
            return min(1.0, total_interest / relevant_topics)
        else:
            # 新话题，给予基础兴趣度
            for keyword in keyword_list[:3]:
                self.topic_interests[keyword] = 0.5
            return 0.5

    def get_confidence(self) -> float:
        return 0.7


class UserInteractionInterestCalculator(InterestCalculator):
    """用户交互兴趣度计算器"""

    def __init__(self):
        self.interaction_history: List[Dict] = []
        self.max_history_size = 100

    def add_interaction(self, user_id: str, interaction_type: str, value: float):
        """添加交互记录"""
        self.interaction_history.append({
            "user_id": user_id,
            "type": interaction_type,
            "value": value,
            "timestamp": time.time()
        })

        # 保持历史记录大小
        if len(self.interaction_history) > self.max_history_size:
            self.interaction_history = self.interaction_history[-self.max_history_size:]

    def calculate(self, context: Dict[str, Any]) -> float:
        """基于用户交互历史计算兴趣度"""
        user_id = context.get("user_id")
        if not user_id:
            return 0.3

        # 获取该用户的最近交互记录
        user_interactions = [
            interaction for interaction in self.interaction_history
            if interaction["user_id"] == user_id
        ]

        if not user_interactions:
            return 0.3

        # 计算加权平均（最近的交互权重更高）
        total_weight = 0.0
        weighted_sum = 0.0

        for interaction in user_interactions[-20:]:  # 最近20次交互
            age = time.time() - interaction["timestamp"]
            weight = max(0.1, 1.0 - age / (7 * 24 * 3600))  # 7天内衰减

            weighted_sum += interaction["value"] * weight
            total_weight += weight

        if total_weight > 0:
            return min(1.0, weighted_sum / total_weight)
        else:
            return 0.3

    def get_confidence(self) -> float:
        return 0.6


class InterestManager:
    """兴趣度管理器 - 统一管理所有兴趣度计算"""

    def __init__(self) -> None:
        self.calculators: Dict[InterestSourceType, InterestCalculator] = {
            InterestSourceType.MESSAGE_CONTENT: MessageContentInterestCalculator(),
            InterestSourceType.TOPIC_RELEVANCE: TopicInterestCalculator(),
            InterestSourceType.USER_INTERACTION: UserInteractionInterestCalculator(),
        }

        # 权重配置
        self.source_weights: Dict[InterestSourceType, float] = {
            InterestSourceType.MESSAGE_CONTENT: 0.4,
            InterestSourceType.TOPIC_RELEVANCE: 0.3,
            InterestSourceType.USER_INTERACTION: 0.3,
        }

        # 兴趣度缓存
        self.interest_cache: Dict[str, Tuple[float, float]] = {}  # message_id -> (value, timestamp)
        self.cache_ttl: int = 300  # 5分钟缓存

        # 统计信息
        self.stats: Dict[str, Union[int, float, List[str]]] = {
            "total_calculations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "average_calculation_time": 0.0,
            "calculator_usage": {calc_type.value: 0 for calc_type in InterestSourceType}
        }

        logger.info("兴趣度管理器初始化完成")

    def calculate_message_interest(self, message: Dict[str, Any], context: Dict[str, Any]) -> float:
        """计算消息兴趣度"""
        start_time = time.time()
        message_id = message.get("message_id", "")

        # 更新统计
        self.stats["total_calculations"] += 1

        # 检查缓存
        if message_id in self.interest_cache:
            cached_value, cached_time = self.interest_cache[message_id]
            if time.time() - cached_time < self.cache_ttl:
                self.stats["cache_hits"] += 1
                logger.debug(f"使用缓存兴趣度: {message_id} = {cached_value:.3f}")
                return cached_value
        else:
            self.stats["cache_misses"] += 1

        # 构建计算上下文
        calc_context: Dict[str, Any] = {
            "message": message,
            "user_id": message.get("user_id"),
            **context
        }

        # 计算各来源的兴趣度
        source_scores: Dict[InterestSourceType, float] = {}
        total_confidence = 0.0

        for source_type, calculator in self.calculators.items():
            try:
                score = calculator.calculate(calc_context)
                confidence = calculator.get_confidence()

                source_scores[source_type] = score
                total_confidence += confidence

                # 更新计算器使用统计
                self.stats["calculator_usage"][source_type.value] += 1

                logger.debug(f"{source_type.value} 兴趣度: {score:.3f} (置信度: {confidence:.3f})")

            except Exception as e:
                logger.warning(f"计算 {source_type.value} 兴趣度失败: {e}")
                source_scores[source_type] = 0.3

        # 加权计算最终兴趣度
        final_interest = 0.0
        total_weight = 0.0

        for source_type, score in source_scores.items():
            weight = self.source_weights.get(source_type, 0.0)
            final_interest += score * weight
            total_weight += weight

        if total_weight > 0:
            final_interest /= total_weight

        # 确保在合理范围内
        final_interest = max(0.0, min(1.0, final_interest))

        # 缓存结果
        self.interest_cache[message_id] = (final_interest, time.time())

        # 清理过期缓存
        self._cleanup_cache()

        # 更新平均计算时间
        calculation_time = time.time() - start_time
        total_calculations = self.stats["total_calculations"]
        self.stats["average_calculation_time"] = (
            (self.stats["average_calculation_time"] * (total_calculations - 1) + calculation_time)
            / total_calculations
        )

        logger.info(f"消息 {message_id} 最终兴趣度: {final_interest:.3f} (耗时: {calculation_time:.3f}s)")
        return final_interest

    def update_topic_interest(self, message: Dict[str, Any], interest_value: float) -> None:
        """更新话题兴趣度"""
        topic_calc = self.calculators.get(InterestSourceType.TOPIC_RELEVANCE)
        if isinstance(topic_calc, TopicInterestCalculator):
            # 提取关键词作为话题
            keywords = message.get("key_words", "[]")
            try:
                import json
                keyword_list: List[str] = json.loads(keywords) if keywords else []
                for keyword in keyword_list[:3]:  # 更新前3个关键词
                    topic_calc.update_topic_interest(keyword, interest_value)
            except (json.JSONDecodeError, TypeError):
                pass

    def add_user_interaction(self, user_id: str, interaction_type: str, value: float) -> None:
        """添加用户交互记录"""
        interaction_calc = self.calculators.get(InterestSourceType.USER_INTERACTION)
        if isinstance(interaction_calc, UserInteractionInterestCalculator):
            interaction_calc.add_interaction(user_id, interaction_type, value)

    def get_topic_interests(self) -> Dict[str, float]:
        """获取所有话题兴趣度"""
        topic_calc = self.calculators.get(InterestSourceType.TOPIC_RELEVANCE)
        if isinstance(topic_calc, TopicInterestCalculator):
            return topic_calc.topic_interests.copy()
        return {}

    def _cleanup_cache(self) -> None:
        """清理过期缓存"""
        current_time = time.time()
        expired_keys = [
            message_id for message_id, (_, timestamp) in self.interest_cache.items()
            if current_time - timestamp > self.cache_ttl
        ]

        for key in expired_keys:
            del self.interest_cache[key]

        if expired_keys:
            logger.debug(f"清理了 {len(expired_keys)} 个过期兴趣度缓存")

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "cache_size": len(self.interest_cache),
            "topic_count": len(self.get_topic_interests()),
            "calculators": list(self.calculators.keys()),
            "performance_stats": self.stats.copy(),
        }

    def add_calculator(self, source_type: InterestSourceType, calculator: InterestCalculator) -> None:
        """添加自定义计算器"""
        self.calculators[source_type] = calculator
        logger.info(f"添加计算器: {source_type.value}")

    def remove_calculator(self, source_type: InterestSourceType) -> None:
        """移除计算器"""
        if source_type in self.calculators:
            del self.calculators[source_type]
            logger.info(f"移除计算器: {source_type.value}")

    def set_source_weight(self, source_type: InterestSourceType, weight: float) -> None:
        """设置来源权重"""
        self.source_weights[source_type] = max(0.0, min(1.0, weight))
        logger.info(f"设置 {source_type.value} 权重: {weight}")

    def clear_cache(self) -> None:
        """清空缓存"""
        self.interest_cache.clear()
        logger.info("清空兴趣度缓存")

    def get_cache_hit_rate(self) -> float:
        """获取缓存命中率"""
        total_requests = self.stats.get("cache_hits", 0) + self.stats.get("cache_misses", 0)
        if total_requests == 0:
            return 0.0
        return self.stats["cache_hits"] / total_requests


# 全局兴趣度管理器实例
interest_manager = InterestManager()