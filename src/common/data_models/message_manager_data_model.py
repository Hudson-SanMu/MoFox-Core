"""
消息管理模块数据模型
定义消息管理器使用的数据结构
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, TYPE_CHECKING

from . import BaseDataModel
from src.plugin_system.base.component_types import ChatMode, ChatType
from src.common.logger import get_logger

if TYPE_CHECKING:
    from .database_data_model import DatabaseMessages

logger = get_logger("stream_context")


class MessageStatus(Enum):
    """消息状态枚举"""

    UNREAD = "unread"  # 未读消息
    READ = "read"  # 已读消息
    PROCESSING = "processing"  # 处理中


@dataclass
class StreamContext(BaseDataModel):
    """聊天流上下文信息"""

    stream_id: str
    chat_type: ChatType = ChatType.PRIVATE  # 聊天类型，默认为私聊
    chat_mode: ChatMode = ChatMode.NORMAL  # 聊天模式，默认为普通模式
    unread_messages: List["DatabaseMessages"] = field(default_factory=list)
    history_messages: List["DatabaseMessages"] = field(default_factory=list)
    last_check_time: float = field(default_factory=time.time)
    is_active: bool = True
    processing_task: Optional[asyncio.Task] = None
    interruption_count: int = 0  # 打断计数器
    last_interruption_time: float = 0.0  # 上次打断时间
    afc_threshold_adjustment: float = 0.0  # afc阈值调整量

    # 独立分发周期字段
    next_check_time: float = field(default_factory=time.time)  # 下次检查时间
    distribution_interval: float = 5.0  # 当前分发周期（秒）

    # 新增字段以替代ChatMessageContext功能
    current_message: Optional["DatabaseMessages"] = None
    priority_mode: Optional[str] = None
    priority_info: Optional[dict] = None

    def add_message(self, message: "DatabaseMessages"):
        """添加消息到上下文"""
        message.is_read = False
        self.unread_messages.append(message)

        # 自动检测和更新chat type
        self._detect_chat_type(message)

    def update_message_info(self, message_id: str, interest_degree: float = None, actions: list = None, should_reply: bool = None):
        """
        更新消息信息

        Args:
            message_id: 消息ID
            interest_degree: 兴趣度值
            actions: 执行的动作列表
            should_reply: 是否应该回复
        """
        # 在未读消息中查找并更新
        for message in self.unread_messages:
            if message.message_id == message_id:
                message.update_message_info(interest_degree, actions, should_reply)
                break

        # 在历史消息中查找并更新
        for message in self.history_messages:
            if message.message_id == message_id:
                message.update_message_info(interest_degree, actions, should_reply)
                break

    def add_action_to_message(self, message_id: str, action: str):
        """
        向指定消息添加执行的动作

        Args:
            message_id: 消息ID
            action: 要添加的动作名称
        """
        # 在未读消息中查找并更新
        for message in self.unread_messages:
            if message.message_id == message_id:
                message.add_action(action)
                break

        # 在历史消息中查找并更新
        for message in self.history_messages:
            if message.message_id == message_id:
                message.add_action(action)
                break

    def _detect_chat_type(self, message: "DatabaseMessages"):
        """根据消息内容自动检测聊天类型"""
        # 只有在第一次添加消息时才检测聊天类型，避免后续消息改变类型
        if len(self.unread_messages) == 1:  # 只有这条消息
            # 如果消息包含群组信息，则为群聊
            if hasattr(message, "chat_info_group_id") and message.chat_info_group_id:
                self.chat_type = ChatType.GROUP
            elif hasattr(message, "chat_info_group_name") and message.chat_info_group_name:
                self.chat_type = ChatType.GROUP
            else:
                self.chat_type = ChatType.PRIVATE

    def update_chat_type(self, chat_type: ChatType):
        """手动更新聊天类型"""
        self.chat_type = chat_type

    def set_chat_mode(self, chat_mode: ChatMode):
        """设置聊天模式"""
        self.chat_mode = chat_mode

    def is_group_chat(self) -> bool:
        """检查是否为群聊"""
        return self.chat_type == ChatType.GROUP

    def is_private_chat(self) -> bool:
        """检查是否为私聊"""
        return self.chat_type == ChatType.PRIVATE

    def get_chat_type_display(self) -> str:
        """获取聊天类型的显示名称"""
        if self.chat_type == ChatType.GROUP:
            return "群聊"
        elif self.chat_type == ChatType.PRIVATE:
            return "私聊"
        else:
            return "未知类型"

    def mark_message_as_read(self, message_id: str):
        """标记消息为已读"""
        for msg in self.unread_messages:
            if msg.message_id == message_id:
                msg.is_read = True
                self.history_messages.append(msg)
                self.unread_messages.remove(msg)
                break

    def get_unread_messages(self) -> List["DatabaseMessages"]:
        """获取未读消息"""
        return [msg for msg in self.unread_messages if not msg.is_read]

    def get_history_messages(self, limit: int = 20) -> List["DatabaseMessages"]:
        """获取历史消息"""
        # 优先返回最近的历史消息和所有未读消息
        recent_history = self.history_messages[-limit:] if len(self.history_messages) > limit else self.history_messages
        return recent_history

    def calculate_interruption_probability(self, max_limit: int, probability_factor: float) -> float:
        """计算打断概率"""
        if max_limit <= 0:
            return 0.0

        # 计算打断比例
        interruption_ratio = self.interruption_count / max_limit

        # 如果超过概率因子，概率下降
        if interruption_ratio > probability_factor:
            # 使用指数衰减，超过限制越多，概率越低
            excess_ratio = interruption_ratio - probability_factor
            probability = 1.0 * (0.5**excess_ratio)  # 基础概率0.5，指数衰减
        else:
            # 在限制内，保持较高概率
            probability = 0.8

        return max(0.0, min(1.0, probability))

    def increment_interruption_count(self):
        """增加打断计数"""
        self.interruption_count += 1
        self.last_interruption_time = time.time()

    def reset_interruption_count(self):
        """重置打断计数和afc阈值调整"""
        self.interruption_count = 0
        self.last_interruption_time = 0.0
        self.afc_threshold_adjustment = 0.0

    def apply_interruption_afc_reduction(self, reduction_value: float):
        """应用打断导致的afc阈值降低"""
        self.afc_threshold_adjustment += reduction_value
        logger.debug(f"应用afc阈值降低: {reduction_value}, 总调整量: {self.afc_threshold_adjustment}")

    def get_afc_threshold_adjustment(self) -> float:
        """获取当前的afc阈值调整量"""
        return self.afc_threshold_adjustment

    def set_current_message(self, message: "DatabaseMessages"):
        """设置当前消息"""
        self.current_message = message

    def get_template_name(self) -> Optional[str]:
        """获取模板名称"""
        if self.current_message and hasattr(self.current_message, 'additional_config') and self.current_message.additional_config:
            try:
                import json
                config = json.loads(self.current_message.additional_config)
                if config.get('template_info') and not config.get('template_default', True):
                    return config.get('template_name')
            except (json.JSONDecodeError, AttributeError):
                pass
        return None

    def get_last_message(self) -> Optional["DatabaseMessages"]:
        """获取最后一条消息"""
        if self.current_message:
            return self.current_message
        if self.unread_messages:
            return self.unread_messages[-1]
        if self.history_messages:
            return self.history_messages[-1]
        return None

    def check_types(self, types: list) -> bool:
        """检查消息类型"""
        if not self.current_message:
            return False

        # 检查消息是否支持指定的类型
        # 这里简化处理，实际应该根据消息的格式信息检查
        if hasattr(self.current_message, 'additional_config') and self.current_message.additional_config:
            try:
                import json
                config = json.loads(self.current_message.additional_config)
                if 'format_info' in config and 'accept_format' in config['format_info']:
                    accept_format = config['format_info']['accept_format']
                    for t in types:
                        if t not in accept_format:
                            return False
                    return True
            except (json.JSONDecodeError, AttributeError):
                pass
        return False

    def get_priority_mode(self) -> Optional[str]:
        """获取优先级模式"""
        return self.priority_mode

    def get_priority_info(self) -> Optional[dict]:
        """获取优先级信息"""
        return self.priority_info


@dataclass
class MessageManagerStats(BaseDataModel):
    """消息管理器统计信息"""

    total_streams: int = 0
    active_streams: int = 0
    total_unread_messages: int = 0
    total_processed_messages: int = 0
    start_time: float = field(default_factory=time.time)

    @property
    def uptime(self) -> float:
        """运行时间"""
        return time.time() - self.start_time


@dataclass
class StreamStats(BaseDataModel):
    """聊天流统计信息"""

    stream_id: str
    is_active: bool
    unread_count: int
    history_count: int
    last_check_time: float
    has_active_task: bool
