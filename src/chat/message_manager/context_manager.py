"""
重构后的聊天上下文管理器
提供统一、稳定的聊天上下文管理功能
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

from src.common.logger import get_logger
from src.config.config import global_config
from src.chat.interest_system import interest_manager
from src.chat.energy_system import energy_manager
from .distribution_manager import distribution_manager

logger = get_logger("context_manager")


class ContextEventType(Enum):
    """上下文事件类型"""
    MESSAGE_ADDED = "message_added"
    MESSAGE_UPDATED = "message_updated"
    ENERGY_CHANGED = "energy_changed"
    STREAM_ACTIVATED = "stream_activated"
    STREAM_DEACTIVATED = "stream_deactivated"
    CONTEXT_CLEARED = "context_cleared"
    VALIDATION_FAILED = "validation_failed"
    CLEANUP_COMPLETED = "cleanup_completed"
    INTEGRITY_CHECK = "integrity_check"

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"ContextEventType.{self.name}"


@dataclass
class ContextEvent:
    """上下文事件"""
    event_type: ContextEventType
    stream_id: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: f"event_{time.time()}_{id(object())}")
    priority: int = 0  # 事件优先级，数字越大优先级越高
    source: str = "system"  # 事件来源

    def __str__(self) -> str:
        return f"ContextEvent({self.event_type}, {self.stream_id}, ts={self.timestamp:.3f})"

    def __repr__(self) -> str:
        return f"ContextEvent(event_type={self.event_type}, stream_id={self.stream_id}, timestamp={self.timestamp}, event_id={self.event_id})"

    def get_age(self) -> float:
        """获取事件年龄（秒）"""
        return time.time() - self.timestamp

    def is_expired(self, max_age: float = 3600.0) -> bool:
        """检查事件是否已过期

        Args:
            max_age: 最大年龄（秒）

        Returns:
            bool: 是否已过期
        """
        return self.get_age() > max_age


class ContextValidator(ABC):
    """上下文验证器抽象基类"""

    @abstractmethod
    def validate_context(self, stream_id: str, context: Any) -> Tuple[bool, Optional[str]]:
        """验证上下文

        Args:
            stream_id: 流ID
            context: 上下文对象

        Returns:
            Tuple[bool, Optional[str]]: (是否有效, 错误信息)
        """
        pass


class DefaultContextValidator(ContextValidator):
    """默认上下文验证器"""

    def validate_context(self, stream_id: str, context: Any) -> Tuple[bool, Optional[str]]:
        """验证上下文基本完整性"""
        if not hasattr(context, 'stream_id'):
            return False, "缺少 stream_id 属性"
        if not hasattr(context, 'unread_messages'):
            return False, "缺少 unread_messages 属性"
        if not hasattr(context, 'history_messages'):
            return False, "缺少 history_messages 属性"
        return True, None


class StreamContextManager:
    """流上下文管理器 - 统一管理所有聊天流上下文"""

    def __init__(self, max_context_size: Optional[int] = None, context_ttl: Optional[int] = None):
        # 上下文存储
        self.stream_contexts: Dict[str, Any] = {}
        self.context_metadata: Dict[str, Dict[str, Any]] = {}

        # 事件监听器
        self.event_listeners: Dict[ContextEventType, List[Callable]] = {}
        self.event_history: List[ContextEvent] = []
        self.max_event_history = 1000

        # 验证器
        self.validators: List[ContextValidator] = [DefaultContextValidator()]

        # 统计信息
        self.stats: Dict[str, Union[int, float, str, Dict]] = {
            "total_messages": 0,
            "total_streams": 0,
            "active_streams": 0,
            "inactive_streams": 0,
            "last_activity": time.time(),
            "creation_time": time.time(),
            "validation_stats": {
                "total_validations": 0,
                "validation_failures": 0,
                "last_validation_time": 0.0,
            },
            "event_stats": {
                "total_events": 0,
                "events_by_type": {},
                "last_event_time": 0.0,
            },
        }

        # 配置参数
        self.max_context_size = max_context_size or getattr(global_config.chat, "max_context_size", 100)
        self.context_ttl = context_ttl or getattr(global_config.chat, "context_ttl", 24 * 3600)  # 24小时
        self.cleanup_interval = getattr(global_config.chat, "context_cleanup_interval", 3600)  # 1小时
        self.auto_cleanup = getattr(global_config.chat, "auto_cleanup_contexts", True)
        self.enable_validation = getattr(global_config.chat, "enable_context_validation", True)

        # 清理任务
        self.cleanup_task: Optional[Any] = None
        self.is_running = False

        logger.info(f"上下文管理器初始化完成 (最大上下文: {self.max_context_size}, TTL: {self.context_ttl}s)")

    def add_stream_context(self, stream_id: str, context: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """添加流上下文

        Args:
            stream_id: 流ID
            context: 上下文对象
            metadata: 上下文元数据

        Returns:
            bool: 是否成功添加
        """
        if stream_id in self.stream_contexts:
            logger.warning(f"流上下文已存在: {stream_id}")
            return False

        # 验证上下文
        if self.enable_validation:
            is_valid, error_msg = self._validate_context(stream_id, context)
            if not is_valid:
                logger.error(f"上下文验证失败: {stream_id} - {error_msg}")
                self._emit_event(ContextEventType.VALIDATION_FAILED, stream_id, {
                    "error": error_msg,
                    "context_type": type(context).__name__
                })
                return False

        # 添加上下文
        self.stream_contexts[stream_id] = context

        # 初始化元数据
        self.context_metadata[stream_id] = {
            "created_time": time.time(),
            "last_access_time": time.time(),
            "access_count": 0,
            "validation_errors": 0,
            "last_validation_time": 0.0,
            "custom_metadata": metadata or {},
        }

        # 更新统计
        self.stats["total_streams"] += 1
        self.stats["active_streams"] += 1
        self.stats["last_activity"] = time.time()

        # 触发事件
        self._emit_event(ContextEventType.STREAM_ACTIVATED, stream_id, {
            "context": context,
            "context_type": type(context).__name__,
            "metadata": metadata
        })

        logger.debug(f"添加流上下文: {stream_id} (类型: {type(context).__name__})")
        return True

    def remove_stream_context(self, stream_id: str) -> bool:
        """移除流上下文

        Args:
            stream_id: 流ID

        Returns:
            bool: 是否成功移除
        """
        if stream_id in self.stream_contexts:
            context = self.stream_contexts[stream_id]
            metadata = self.context_metadata.get(stream_id, {})

            del self.stream_contexts[stream_id]
            if stream_id in self.context_metadata:
                del self.context_metadata[stream_id]

            self.stats["active_streams"] = max(0, self.stats["active_streams"] - 1)
            self.stats["inactive_streams"] += 1
            self.stats["last_activity"] = time.time()

            # 触发事件
            self._emit_event(ContextEventType.STREAM_DEACTIVATED, stream_id, {
                "context": context,
                "context_type": type(context).__name__,
                "metadata": metadata,
                "uptime": time.time() - metadata.get("created_time", time.time())
            })

            logger.debug(f"移除流上下文: {stream_id} (类型: {type(context).__name__})")
            return True
        return False

    def get_stream_context(self, stream_id: str, update_access: bool = True) -> Optional[Any]:
        """获取流上下文

        Args:
            stream_id: 流ID
            update_access: 是否更新访问统计

        Returns:
            Optional[Any]: 上下文对象
        """
        context = self.stream_contexts.get(stream_id)
        if context and update_access:
            # 更新访问统计
            if stream_id in self.context_metadata:
                metadata = self.context_metadata[stream_id]
                metadata["last_access_time"] = time.time()
                metadata["access_count"] = metadata.get("access_count", 0) + 1
        return context

    def get_context_metadata(self, stream_id: str) -> Optional[Dict[str, Any]]:
        """获取上下文元数据

        Args:
            stream_id: 流ID

        Returns:
            Optional[Dict[str, Any]]: 元数据
        """
        return self.context_metadata.get(stream_id)

    def update_context_metadata(self, stream_id: str, updates: Dict[str, Any]) -> bool:
        """更新上下文元数据

        Args:
            stream_id: 流ID
            updates: 更新的元数据

        Returns:
            bool: 是否成功更新
        """
        if stream_id not in self.context_metadata:
            return False

        self.context_metadata[stream_id].update(updates)
        return True

    def add_message_to_context(self, stream_id: str, message: Any, skip_energy_update: bool = False) -> bool:
        """添加消息到上下文

        Args:
            stream_id: 流ID
            message: 消息对象
            skip_energy_update: 是否跳过能量更新

        Returns:
            bool: 是否成功添加
        """
        context = self.get_stream_context(stream_id)
        if not context:
            logger.warning(f"流上下文不存在: {stream_id}")
            return False

        try:
            # 添加消息到上下文
            if hasattr(context, 'add_message'):
                context.add_message(message)
            else:
                logger.error(f"上下文对象缺少 add_message 方法: {stream_id}")
                return False

            # 计算消息兴趣度
            interest_value = self._calculate_message_interest(message)
            if hasattr(message, 'interest_value'):
                message.interest_value = interest_value

            # 更新统计
            self.stats["total_messages"] += 1
            self.stats["last_activity"] = time.time()

            # 触发事件
            event_data = {
                "message": message,
                "interest_value": interest_value,
                "message_type": type(message).__name__,
                "message_id": getattr(message, "message_id", None),
            }
            self._emit_event(ContextEventType.MESSAGE_ADDED, stream_id, event_data)

            # 更新能量和分发
            if not skip_energy_update:
                self._update_stream_energy(stream_id)
                distribution_manager.add_stream_message(stream_id, 1)

            logger.debug(f"添加消息到上下文: {stream_id} (兴趣度: {interest_value:.3f})")
            return True

        except Exception as e:
            logger.error(f"添加消息到上下文失败 {stream_id}: {e}", exc_info=True)
            return False

    def update_message_in_context(self, stream_id: str, message_id: str, updates: Dict[str, Any]) -> bool:
        """更新上下文中的消息

        Args:
            stream_id: 流ID
            message_id: 消息ID
            updates: 更新的属性

        Returns:
            bool: 是否成功更新
        """
        context = self.get_stream_context(stream_id)
        if not context:
            logger.warning(f"流上下文不存在: {stream_id}")
            return False

        try:
            # 更新消息信息
            if hasattr(context, 'update_message_info'):
                context.update_message_info(message_id, **updates)
            else:
                logger.error(f"上下文对象缺少 update_message_info 方法: {stream_id}")
                return False

            # 触发事件
            self._emit_event(ContextEventType.MESSAGE_UPDATED, stream_id, {
                "message_id": message_id,
                "updates": updates,
                "update_time": time.time(),
            })

            # 如果更新了兴趣度，重新计算能量
            if "interest_value" in updates:
                self._update_stream_energy(stream_id)

            logger.debug(f"更新上下文消息: {stream_id}/{message_id}")
            return True

        except Exception as e:
            logger.error(f"更新上下文消息失败 {stream_id}/{message_id}: {e}", exc_info=True)
            return False

    def get_context_messages(self, stream_id: str, limit: Optional[int] = None, include_unread: bool = True) -> List[Any]:
        """获取上下文消息

        Args:
            stream_id: 流ID
            limit: 消息数量限制
            include_unread: 是否包含未读消息

        Returns:
            List[Any]: 消息列表
        """
        context = self.get_stream_context(stream_id)
        if not context:
            return []

        try:
            messages = []
            if include_unread and hasattr(context, 'get_unread_messages'):
                messages.extend(context.get_unread_messages())

            if hasattr(context, 'get_history_messages'):
                if limit:
                    messages.extend(context.get_history_messages(limit=limit))
                else:
                    messages.extend(context.get_history_messages())

            # 按时间排序
            messages.sort(key=lambda msg: getattr(msg, 'time', 0))

            # 应用限制
            if limit and len(messages) > limit:
                messages = messages[-limit:]

            return messages

        except Exception as e:
            logger.error(f"获取上下文消息失败 {stream_id}: {e}", exc_info=True)
            return []

    def get_unread_messages(self, stream_id: str) -> List[Any]:
        """获取未读消息

        Args:
            stream_id: 流ID

        Returns:
            List[Any]: 未读消息列表
        """
        context = self.get_stream_context(stream_id)
        if not context:
            return []

        try:
            if hasattr(context, 'get_unread_messages'):
                return context.get_unread_messages()
            else:
                logger.warning(f"上下文对象缺少 get_unread_messages 方法: {stream_id}")
                return []
        except Exception as e:
            logger.error(f"获取未读消息失败 {stream_id}: {e}", exc_info=True)
            return []

    def mark_messages_as_read(self, stream_id: str, message_ids: List[str]) -> bool:
        """标记消息为已读

        Args:
            stream_id: 流ID
            message_ids: 消息ID列表

        Returns:
            bool: 是否成功标记
        """
        context = self.get_stream_context(stream_id)
        if not context:
            logger.warning(f"流上下文不存在: {stream_id}")
            return False

        try:
            if not hasattr(context, 'mark_message_as_read'):
                logger.error(f"上下文对象缺少 mark_message_as_read 方法: {stream_id}")
                return False

            marked_count = 0
            for message_id in message_ids:
                try:
                    context.mark_message_as_read(message_id)
                    marked_count += 1
                except Exception as e:
                    logger.warning(f"标记消息已读失败 {message_id}: {e}")

            logger.debug(f"标记消息为已读: {stream_id} ({marked_count}/{len(message_ids)}条)")
            return marked_count > 0

        except Exception as e:
            logger.error(f"标记消息已读失败 {stream_id}: {e}", exc_info=True)
            return False

    def clear_context(self, stream_id: str) -> bool:
        """清空上下文

        Args:
            stream_id: 流ID

        Returns:
            bool: 是否成功清空
        """
        context = self.get_stream_context(stream_id)
        if not context:
            logger.warning(f"流上下文不存在: {stream_id}")
            return False

        try:
            # 清空消息
            if hasattr(context, 'unread_messages'):
                context.unread_messages.clear()
            if hasattr(context, 'history_messages'):
                context.history_messages.clear()

            # 重置状态
            reset_attrs = ['interruption_count', 'afc_threshold_adjustment', 'last_check_time']
            for attr in reset_attrs:
                if hasattr(context, attr):
                    if attr in ['interruption_count', 'afc_threshold_adjustment']:
                        setattr(context, attr, 0)
                    else:
                        setattr(context, attr, time.time())

            # 触发事件
            self._emit_event(ContextEventType.CONTEXT_CLEARED, stream_id, {
                "clear_time": time.time(),
                "reset_attributes": reset_attrs,
            })

            # 重新计算能量
            self._update_stream_energy(stream_id)

            logger.info(f"清空上下文: {stream_id}")
            return True

        except Exception as e:
            logger.error(f"清空上下文失败 {stream_id}: {e}", exc_info=True)
            return False

    def _calculate_message_interest(self, message: Any) -> float:
        """计算消息兴趣度"""
        try:
            # 将消息转换为字典格式
            message_dict = self._message_to_dict(message)

            # 使用兴趣度管理器计算
            context = {
                "stream_id": getattr(message, 'chat_info_stream_id', ''),
                "user_id": getattr(message, 'user_id', ''),
            }

            interest_value = interest_manager.calculate_message_interest(message_dict, context)

            # 更新话题兴趣度
            interest_manager.update_topic_interest(message_dict, interest_value)

            return interest_value

        except Exception as e:
            logger.error(f"计算消息兴趣度失败: {e}")
            return 0.5

    def _message_to_dict(self, message: Any) -> Dict[str, Any]:
        """将消息对象转换为字典"""
        try:
            # 获取user_id，优先从user_info.user_id获取，其次从user_id属性获取
            user_id = ""
            if hasattr(message, 'user_info') and hasattr(message.user_info, 'user_id'):
                user_id = getattr(message.user_info, 'user_id', "")
            else:
                user_id = getattr(message, 'user_id', "")

            return {
                "message_id": getattr(message, "message_id", ""),
                "processed_plain_text": getattr(message, "processed_plain_text", ""),
                "is_emoji": getattr(message, "is_emoji", False),
                "is_picid": getattr(message, "is_picid", False),
                "is_mentioned": getattr(message, "is_mentioned", False),
                "is_command": getattr(message, "is_command", False),
                "key_words": getattr(message, "key_words", "[]"),
                "user_id": user_id,
                "time": getattr(message, "time", time.time()),
            }
        except Exception as e:
            logger.error(f"转换消息为字典失败: {e}")
            return {}

    def _update_stream_energy(self, stream_id: str):
        """更新流能量"""
        try:
            # 获取所有消息
            all_messages = self.get_context_messages(stream_id, self.max_context_size)
            unread_messages = self.get_unread_messages(stream_id)
            combined_messages = all_messages + unread_messages

            # 获取用户ID
            user_id = None
            if combined_messages:
                last_message = combined_messages[-1]
                user_id = getattr(last_message, "user_id", None)

            # 计算能量
            energy = energy_manager.calculate_focus_energy(
                stream_id=stream_id,
                messages=combined_messages,
                user_id=user_id
            )

            # 更新分发管理器
            distribution_manager.update_stream_energy(stream_id, energy)

            # 触发事件
            self._emit_event(ContextEventType.ENERGY_CHANGED, stream_id, {
                "energy": energy,
                "message_count": len(combined_messages),
            })

        except Exception as e:
            logger.error(f"更新流能量失败 {stream_id}: {e}")

    def add_event_listener(self, event_type: ContextEventType, listener: Callable[[ContextEvent], None]) -> bool:
        """添加事件监听器

        Args:
            event_type: 事件类型
            listener: 监听器函数

        Returns:
            bool: 是否成功添加
        """
        if not callable(listener):
            logger.error(f"监听器必须是可调用对象: {type(listener)}")
            return False

        if event_type not in self.event_listeners:
            self.event_listeners[event_type] = []

        if listener not in self.event_listeners[event_type]:
            self.event_listeners[event_type].append(listener)
            logger.debug(f"添加事件监听器: {event_type} -> {getattr(listener, '__name__', 'anonymous')}")
            return True
        return False

    def remove_event_listener(self, event_type: ContextEventType, listener: Callable[[ContextEvent], None]) -> bool:
        """移除事件监听器

        Args:
            event_type: 事件类型
            listener: 监听器函数

        Returns:
            bool: 是否成功移除
        """
        if event_type in self.event_listeners:
            try:
                self.event_listeners[event_type].remove(listener)
                logger.debug(f"移除事件监听器: {event_type}")
                return True
            except ValueError:
                pass
        return False

    def _emit_event(self, event_type: ContextEventType, stream_id: str, data: Optional[Dict] = None, priority: int = 0) -> None:
        """触发事件

        Args:
            event_type: 事件类型
            stream_id: 流ID
            data: 事件数据
            priority: 事件优先级
        """
        if data is None:
            data = {}

        event = ContextEvent(event_type, stream_id, data, priority=priority)

        # 添加到事件历史
        self.event_history.append(event)
        if len(self.event_history) > self.max_event_history:
            self.event_history = self.event_history[-self.max_event_history:]

        # 更新事件统计
        event_stats = self.stats["event_stats"]
        event_stats["total_events"] += 1
        event_stats["last_event_time"] = time.time()
        event_type_str = str(event_type)
        event_stats["events_by_type"][event_type_str] = event_stats["events_by_type"].get(event_type_str, 0) + 1

        # 通知监听器
        if event_type in self.event_listeners:
            for listener in self.event_listeners[event_type]:
                try:
                    listener(event)
                except Exception as e:
                    logger.error(f"事件监听器执行失败: {e}", exc_info=True)

    def get_stream_statistics(self, stream_id: str) -> Optional[Dict[str, Any]]:
        """获取流统计信息

        Args:
            stream_id: 流ID

        Returns:
            Optional[Dict[str, Any]]: 统计信息
        """
        context = self.get_stream_context(stream_id, update_access=False)
        if not context:
            return None

        try:
            metadata = self.context_metadata.get(stream_id, {})
            current_time = time.time()
            created_time = metadata.get("created_time", current_time)
            last_access_time = metadata.get("last_access_time", current_time)
            access_count = metadata.get("access_count", 0)

            unread_messages = getattr(context, "unread_messages", [])
            history_messages = getattr(context, "history_messages", [])

            return {
                "stream_id": stream_id,
                "context_type": type(context).__name__,
                "total_messages": len(history_messages) + len(unread_messages),
                "unread_messages": len(unread_messages),
                "history_messages": len(history_messages),
                "is_active": getattr(context, "is_active", True),
                "last_check_time": getattr(context, "last_check_time", current_time),
                "interruption_count": getattr(context, "interruption_count", 0),
                "afc_threshold_adjustment": getattr(context, "afc_threshold_adjustment", 0.0),
                "created_time": created_time,
                "last_access_time": last_access_time,
                "access_count": access_count,
                "uptime_seconds": current_time - created_time,
                "idle_seconds": current_time - last_access_time,
                "validation_errors": metadata.get("validation_errors", 0),
            }
        except Exception as e:
            logger.error(f"获取流统计失败 {stream_id}: {e}", exc_info=True)
            return None

    def get_manager_statistics(self) -> Dict[str, Any]:
        """获取管理器统计信息

        Returns:
            Dict[str, Any]: 管理器统计信息
        """
        current_time = time.time()
        uptime = current_time - self.stats.get("creation_time", current_time)

        # 计算验证统计
        validation_stats = self.stats["validation_stats"]
        validation_success_rate = (
            (validation_stats.get("total_validations", 0) - validation_stats.get("validation_failures", 0)) /
            max(1, validation_stats.get("total_validations", 1))
        )

        # 计算事件统计
        event_stats = self.stats["event_stats"]
        events_by_type = event_stats.get("events_by_type", {})

        return {
            **self.stats,
            "uptime_hours": uptime / 3600,
            "stream_count": len(self.stream_contexts),
            "metadata_count": len(self.context_metadata),
            "event_history_size": len(self.event_history),
            "validators_count": len(self.validators),
            "event_listeners": {
                str(event_type): len(listeners)
                for event_type, listeners in self.event_listeners.items()
            },
            "validation_success_rate": validation_success_rate,
            "event_distribution": events_by_type,
            "max_event_history": self.max_event_history,
            "auto_cleanup_enabled": self.auto_cleanup,
            "cleanup_interval": self.cleanup_interval,
        }

    def cleanup_inactive_contexts(self, max_inactive_hours: int = 24) -> int:
        """清理不活跃的上下文

        Args:
            max_inactive_hours: 最大不活跃小时数

        Returns:
            int: 清理的上下文数量
        """
        current_time = time.time()
        max_inactive_seconds = max_inactive_hours * 3600

        inactive_streams = []
        for stream_id, context in self.stream_contexts.items():
            try:
                # 获取最后活动时间
                metadata = self.context_metadata.get(stream_id, {})
                last_activity = metadata.get("last_access_time", metadata.get("created_time", 0))
                context_last_activity = getattr(context, "last_check_time", 0)
                actual_last_activity = max(last_activity, context_last_activity)

                # 检查是否不活跃
                unread_count = len(getattr(context, "unread_messages", []))
                history_count = len(getattr(context, "history_messages", []))
                total_messages = unread_count + history_count

                if (current_time - actual_last_activity > max_inactive_seconds and
                    total_messages == 0):
                    inactive_streams.append(stream_id)
            except Exception as e:
                logger.warning(f"检查上下文活跃状态失败 {stream_id}: {e}")
                continue

        # 清理不活跃上下文
        cleaned_count = 0
        for stream_id in inactive_streams:
            if self.remove_stream_context(stream_id):
                cleaned_count += 1

        if cleaned_count > 0:
            logger.info(f"清理了 {cleaned_count} 个不活跃上下文")

        return cleaned_count

    def validate_context_integrity(self, stream_id: str) -> bool:
        """验证上下文完整性

        Args:
            stream_id: 流ID

        Returns:
            bool: 是否完整
        """
        context = self.get_stream_context(stream_id)
        if not context:
            return False

        try:
            # 检查基本属性
            required_attrs = ["stream_id", "unread_messages", "history_messages"]
            for attr in required_attrs:
                if not hasattr(context, attr):
                    logger.warning(f"上下文缺少必要属性: {attr}")
                    return False

            # 检查消息ID唯一性
            all_messages = getattr(context, "unread_messages", []) + getattr(context, "history_messages", [])
            message_ids = [msg.message_id for msg in all_messages if hasattr(msg, "message_id")]
            if len(message_ids) != len(set(message_ids)):
                logger.warning(f"上下文中存在重复消息ID: {stream_id}")
                return False

            return True

        except Exception as e:
            logger.error(f"验证上下文完整性失败 {stream_id}: {e}")
            return False

    def _validate_context(self, stream_id: str, context: Any) -> Tuple[bool, Optional[str]]:
        """验证上下文完整性

        Args:
            stream_id: 流ID
            context: 上下文对象

        Returns:
            Tuple[bool, Optional[str]]: (是否有效, 错误信息)
        """
        validation_stats = self.stats["validation_stats"]
        validation_stats["total_validations"] += 1
        validation_stats["last_validation_time"] = time.time()

        for validator in self.validators:
            try:
                is_valid, error_msg = validator.validate_context(stream_id, context)
                if not is_valid:
                    validation_stats["validation_failures"] += 1
                    return False, error_msg
            except Exception as e:
                validation_stats["validation_failures"] += 1
                return False, f"验证器执行失败: {e}"
        return True, None

    async def start(self) -> None:
        """启动上下文管理器"""
        if self.is_running:
            logger.warning("上下文管理器已经在运行")
            return

        await self.start_auto_cleanup()
        logger.info("上下文管理器已启动")

    async def stop(self) -> None:
        """停止上下文管理器"""
        if not self.is_running:
            return

        await self.stop_auto_cleanup()
        logger.info("上下文管理器已停止")

    async def start_auto_cleanup(self, interval: Optional[float] = None) -> None:
        """启动自动清理

        Args:
            interval: 清理间隔（秒）
        """
        if not self.auto_cleanup:
            logger.info("自动清理已禁用")
            return

        if self.is_running:
            logger.warning("自动清理已在运行")
            return

        self.is_running = True
        cleanup_interval = interval or self.cleanup_interval
        logger.info(f"启动自动清理（间隔: {cleanup_interval}s）")

        import asyncio
        self.cleanup_task = asyncio.create_task(self._cleanup_loop(cleanup_interval))

    async def stop_auto_cleanup(self) -> None:
        """停止自动清理"""
        self.is_running = False
        if self.cleanup_task and not self.cleanup_task.done():
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except Exception:
                pass
        logger.info("自动清理已停止")

    async def _cleanup_loop(self, interval: float) -> None:
        """清理循环

        Args:
            interval: 清理间隔
        """
        while self.is_running:
            try:
                await asyncio.sleep(interval)
                self.cleanup_inactive_contexts()
                self._cleanup_event_history()
                self._cleanup_expired_contexts()
                logger.debug("自动清理完成")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"清理循环出错: {e}", exc_info=True)
                await asyncio.sleep(interval)

    def _cleanup_event_history(self) -> None:
        """清理事件历史"""
        max_age = 24 * 3600  # 24小时

        # 清理过期事件
        self.event_history = [
            event for event in self.event_history
            if not event.is_expired(max_age)
        ]

        # 保持历史大小限制
        if len(self.event_history) > self.max_event_history:
            self.event_history = self.event_history[-self.max_event_history:]

    def _cleanup_expired_contexts(self) -> None:
        """清理过期上下文"""
        current_time = time.time()
        expired_contexts = []

        for stream_id, metadata in self.context_metadata.items():
            created_time = metadata.get("created_time", current_time)
            if current_time - created_time > self.context_ttl:
                expired_contexts.append(stream_id)

        for stream_id in expired_contexts:
            self.remove_stream_context(stream_id)

        if expired_contexts:
            logger.info(f"清理了 {len(expired_contexts)} 个过期上下文")

    def get_event_history(self, limit: int = 100, event_type: Optional[ContextEventType] = None) -> List[ContextEvent]:
        """获取事件历史

        Args:
            limit: 返回数量限制
            event_type: 过滤事件类型

        Returns:
            List[ContextEvent]: 事件列表
        """
        events = self.event_history
        if event_type:
            events = [event for event in events if event.event_type == event_type]
        return events[-limit:]

    def get_active_streams(self) -> List[str]:
        """获取活跃流列表

        Returns:
            List[str]: 活跃流ID列表
        """
        return list(self.stream_contexts.keys())

    def get_context_summary(self) -> Dict[str, Any]:
        """获取上下文摘要

        Returns:
            Dict[str, Any]: 上下文摘要信息
        """
        current_time = time.time()
        uptime = current_time - self.stats.get("creation_time", current_time)

        # 计算平均访问次数
        total_access = sum(meta.get("access_count", 0) for meta in self.context_metadata.values())
        avg_access = total_access / max(1, len(self.context_metadata))

        # 计算验证成功率
        validation_stats = self.stats["validation_stats"]
        total_validations = validation_stats.get("total_validations", 0)
        validation_success_rate = (
            (total_validations - validation_stats.get("validation_failures", 0)) /
            max(1, total_validations)
        ) if total_validations > 0 else 1.0

        return {
            "total_streams": len(self.stream_contexts),
            "active_streams": len(self.stream_contexts),
            "total_messages": self.stats.get("total_messages", 0),
            "uptime_hours": uptime / 3600,
            "average_access_count": avg_access,
            "validation_success_rate": validation_success_rate,
            "event_history_size": len(self.event_history),
            "validators_count": len(self.validators),
            "auto_cleanup_enabled": self.auto_cleanup,
            "cleanup_interval": self.cleanup_interval,
            "last_activity": self.stats.get("last_activity", 0),
        }

    def force_validation(self, stream_id: str) -> Tuple[bool, Optional[str]]:
        """强制验证上下文

        Args:
            stream_id: 流ID

        Returns:
            Tuple[bool, Optional[str]]: (是否有效, 错误信息)
        """
        context = self.get_stream_context(stream_id)
        if not context:
            return False, "上下文不存在"

        return self._validate_context(stream_id, context)

    def reset_statistics(self) -> None:
        """重置统计信息"""
        # 重置基本统计
        self.stats.update({
            "total_messages": 0,
            "total_streams": len(self.stream_contexts),
            "active_streams": len(self.stream_contexts),
            "inactive_streams": 0,
            "last_activity": time.time(),
            "creation_time": time.time(),
        })

        # 重置验证统计
        self.stats["validation_stats"].update({
            "total_validations": 0,
            "validation_failures": 0,
            "last_validation_time": 0.0,
        })

        # 重置事件统计
        self.stats["event_stats"].update({
            "total_events": 0,
            "events_by_type": {},
            "last_event_time": 0.0,
        })

        logger.info("上下文管理器统计信息已重置")

    def export_context_data(self, stream_id: str) -> Optional[Dict[str, Any]]:
        """导出上下文数据

        Args:
            stream_id: 流ID

        Returns:
            Optional[Dict[str, Any]]: 导出的数据
        """
        context = self.get_stream_context(stream_id, update_access=False)
        if not context:
            return None

        try:
            return {
                "stream_id": stream_id,
                "context_type": type(context).__name__,
                "metadata": self.context_metadata.get(stream_id, {}),
                "statistics": self.get_stream_statistics(stream_id),
                "export_time": time.time(),
                "unread_message_count": len(getattr(context, "unread_messages", [])),
                "history_message_count": len(getattr(context, "history_messages", [])),
            }
        except Exception as e:
            logger.error(f"导出上下文数据失败 {stream_id}: {e}")
            return None


# 全局上下文管理器实例
context_manager = StreamContextManager()