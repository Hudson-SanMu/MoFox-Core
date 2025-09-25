"""
消息管理模块
管理每个聊天流的上下文信息，包含历史记录和未读消息，定期检查并处理新消息
"""

import asyncio
import random
import time
import traceback
from typing import Dict, Optional, Any, TYPE_CHECKING, List

from src.common.logger import get_logger
from src.common.data_models.database_data_model import DatabaseMessages
from src.common.data_models.message_manager_data_model import StreamContext, MessageManagerStats, StreamStats
from src.chat.chatter_manager import ChatterManager
from src.chat.planner_actions.action_manager import ChatterActionManager
from .sleep_manager.sleep_manager import SleepManager
from .sleep_manager.wakeup_manager import WakeUpManager
from src.config.config import global_config

if TYPE_CHECKING:
    from src.common.data_models.message_manager_data_model import StreamContext

logger = get_logger("message_manager")


class MessageManager:
    """消息管理器"""

    def __init__(self, check_interval: float = 5.0):
        self.stream_contexts: Dict[str, StreamContext] = {}
        self.check_interval = check_interval
        self.is_running = False
        self.manager_task: Optional[asyncio.Task] = None

        # 并发控制信号量
        self.concurrent_semaphore: Optional[asyncio.Semaphore] = None

        # 统计信息
        self.stats = MessageManagerStats()

        # 初始化chatter manager
        self.action_manager = ChatterActionManager()
        self.chatter_manager = ChatterManager(self.action_manager)

        # 初始化睡眠和唤醒管理器
        self.sleep_manager = SleepManager()
        self.wakeup_manager = WakeUpManager(self.sleep_manager)

    async def start(self):
        """启动消息管理器"""
        if self.is_running:
            logger.warning("消息管理器已经在运行")
            return

        self.is_running = True
        self.manager_task = asyncio.create_task(self._manager_loop())
        if global_config.chat.concurrent_message_processing:
            limit = global_config.chat.concurrent_per_user_limit
            self.concurrent_semaphore = asyncio.Semaphore(limit)
            logger.info(f"并发处理已启用，全局并发限制: {limit}")
        await self.wakeup_manager.start()
        logger.info("消息管理器已启动")

    async def stop(self):
        """停止消息管理器"""
        if not self.is_running:
            return

        self.is_running = False

        # 停止所有流处理任务
        for context in self.stream_contexts.values():
            if hasattr(context, 'processing_task') and context.processing_task and not context.processing_task.done():
                context.processing_task.cancel()
            if hasattr(context, 'user_processing_tasks'):
                for task in context.user_processing_tasks.values():
                    if task and not task.done():
                        task.cancel()

        # 停止管理器任务
        if self.manager_task and not self.manager_task.done():
            self.manager_task.cancel()

        await self.wakeup_manager.stop()

        logger.info("消息管理器已停止")

    def add_message(self, stream_id: str, message: DatabaseMessages):
        """添加消息到指定聊天流"""
        # 获取或创建流上下文
        if stream_id not in self.stream_contexts:
            context = StreamContext(stream_id=stream_id)
            # 为并发处理添加队列和锁
            if global_config.chat.concurrent_message_processing:
                context.send_lock = asyncio.Lock()
                context.user_processing_tasks = {}
            self.stream_contexts[stream_id] = context
            self.stats.total_streams += 1
        
        context = self.stream_contexts[stream_id]
        context.set_chat_mode(ChatMode.FOCUS)
        context.add_message(message)

        logger.debug(f"添加消息到聊天流 {stream_id}: {message.message_id}")

    async def _manager_loop(self):
        """管理器主循环 - 独立聊天流分发周期版本"""
        while self.is_running:
            try:
                # 更新睡眠状态
                await self.sleep_manager.update_sleep_state(self.wakeup_manager)

                # 执行独立分发周期的检查
                await self._check_streams_with_individual_intervals()

                # 计算下次检查时间（使用最小间隔或固定间隔）
                next_check_delay = self.check_interval
                if global_config.chat.dynamic_distribution_enabled:
                    next_check_delay = self._calculate_next_manager_delay()

                await asyncio.sleep(next_check_delay)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"消息管理器循环出错: {e}")
                traceback.print_exc()

    async def _process_stream_messages(self, stream_id: str, unread_messages_override: List[DatabaseMessages]):
        """
        处理指定聊天流的消息 (非并发模式专用)
        """
        if stream_id not in self.stream_contexts:
            return

        context = self.stream_contexts[stream_id]
        context.processing_task = asyncio.current_task()
        user_id = unread_messages_override[0].user_info.user_id if unread_messages_override and hasattr(unread_messages_override[0], 'user_info') else None

        try:
            await self._check_and_handle_interruption(context, stream_id, unread_messages_override, user_id)

            if self.sleep_manager.is_sleeping():
                was_woken_up = False
                is_private = context.is_private_chat()
                for message in unread_messages_override:
                    is_mentioned = message.is_mentioned or False
                    if is_private or is_mentioned:
                        if self.wakeup_manager.add_wakeup_value(is_private, is_mentioned):
                            was_woken_up = True
                            break
                if not was_woken_up:
                    logger.debug(f"聊天流 {stream_id} 中没有唤醒触发器，保持消息未读状态。")
                    self._clear_specific_unread_messages(context, unread_messages_override)
                    return
                logger.info(f"Bot被聊天流 {stream_id} 中的消息吵醒，继续处理。")

            logger.debug(f"开始处理聊天流 {stream_id} 的 {len(unread_messages_override)} 条未读消息")
            
            results = await self.chatter_manager.process_stream_context(stream_id, context, unread_messages_override)
            if results.get("success", False):
                logger.debug(f"聊天流 {stream_id} 处理成功")
            else:
                logger.warning(f"聊天流 {stream_id} 处理失败: {results.get('error_message', '未知错误')}")
            
            self._clear_specific_unread_messages(context, unread_messages_override)

        except asyncio.CancelledError:
            logger.info(f"聊天流 {stream_id} 的处理任务被取消")
            self._clear_specific_unread_messages(context, unread_messages_override)
            raise
        except Exception as e:
            logger.error(f"处理聊天流 {stream_id} 时发生异常: {e}")
            traceback.print_exc()
            self._clear_specific_unread_messages(context, unread_messages_override)
        finally:
            context.processing_task = None
    
    def deactivate_stream(self, stream_id: str):
        """停用聊天流"""
        if stream_id in self.stream_contexts:
            context = self.stream_contexts[stream_id]
            context.is_active = False

            if hasattr(context, 'processing_task') and context.processing_task and not context.processing_task.done():
                context.processing_task.cancel()

            logger.info(f"停用聊天流: {stream_id}")

    def activate_stream(self, stream_id: str):
        """激活聊天流"""
        if stream_id in self.stream_contexts:
            self.stream_contexts[stream_id].is_active = True
            logger.info(f"激活聊天流: {stream_id}")

    def get_stream_stats(self, stream_id: str) -> Optional[StreamStats]:
        """获取聊天流统计"""
        if stream_id not in self.stream_contexts:
            return None

        context = self.stream_contexts[stream_id]
        return StreamStats(
            stream_id=stream_id,
            is_active=context.is_active,
            unread_count=len(context.get_unread_messages()),
            history_count=len(context.history_messages),
            last_check_time=context.last_check_time,
            has_active_task=bool(hasattr(context, 'processing_task') and context.processing_task and not context.processing_task.done()),
        )

    def get_manager_stats(self) -> Dict[str, Any]:
        """获取管理器统计"""
        return {
            "total_streams": self.stats.total_streams,
            "active_streams": self.stats.active_streams,
            "total_unread_messages": self.stats.total_unread_messages,
            "total_processed_messages": self.stats.total_processed_messages,
            "uptime": self.stats.uptime,
            "start_time": self.stats.start_time,
        }

    def cleanup_inactive_streams(self, max_inactive_hours: int = 24):
        """清理不活跃的聊天流"""
        current_time = time.time()
        max_inactive_seconds = max_inactive_hours * 3600

        inactive_streams = []
        for stream_id, context in self.stream_contexts.items():
            if current_time - context.last_check_time > max_inactive_seconds and not context.get_unread_messages():
                inactive_streams.append(stream_id)

        for stream_id in inactive_streams:
            self.deactivate_stream(stream_id)
            del self.stream_contexts[stream_id]
            logger.info(f"清理不活跃聊天流: {stream_id}")

    async def _check_and_handle_interruption(
        self, context: StreamContext, stream_id: str, unread_messages: List[DatabaseMessages], user_id: Optional[str] = None
    ):
        """检查并处理消息打断"""
        if not global_config.chat.interruption_enabled:
            return

        if context.interruption_count >= global_config.chat.interruption_max_limit:
            logger.debug(f"聊天流 {stream_id} 已达到最大打断次数 {context.interruption_count}/{global_config.chat.interruption_max_limit}，本次不进行打断")
            return

        task_to_check = None
        if global_config.chat.concurrent_message_processing and global_config.chat.process_by_user_id and user_id:
            task_to_check = context.user_processing_tasks.get(user_id)
        else:
            task_to_check = context.processing_task

        if task_to_check and not task_to_check.done():
            interruption_probability = context.calculate_interruption_probability(
                global_config.chat.interruption_max_limit, global_config.chat.interruption_probability_factor
            )

            if random.random() < interruption_probability:
                user_nickname = ""
                if user_id and unread_messages:
                    for msg in unread_messages:
                        if hasattr(msg, "user_info") and msg.user_info and msg.user_info.user_id == user_id:
                            user_nickname = msg.user_info.user_nickname
                            break
                
                if user_nickname:
                    log_target = f"用户'{user_nickname}({user_id})'在聊天流 {stream_id}"
                else:
                    log_target = f"用户 {user_id} 在聊天流 {stream_id}" if user_id else f"聊天流 {stream_id}"

                logger.info(f"{log_target} 触发消息打断，打断概率: {interruption_probability:.2f}")
                
                task_to_check.cancel()
                try:
                    await task_to_check
                except asyncio.CancelledError:
                    pass

                context.increment_interruption_count()
                context.apply_interruption_afc_reduction(global_config.chat.interruption_afc_reduction)
                logger.info(
                    f"聊天流 {stream_id} 已打断，当前打断次数: {context.interruption_count}/{global_config.chat.interruption_max_limit}, afc阈值调整: {context.get_afc_threshold_adjustment()}"
                )
            else:
                logger.debug(f"聊天流 {stream_id} 未触发打断，打断概率: {interruption_probability:.2f}")

    def _calculate_dynamic_distribution_interval(self, context: StreamContext) -> float:
        """计算单个聊天流的分发周期 - 基于阈值感知的focus_energy"""
        if not global_config.chat.dynamic_distribution_enabled:
            return self.check_interval

        focus_energy = 0.5
        avg_message_interest = 0.5

        if hasattr(context, 'chat_stream') and context.chat_stream:
            focus_energy = context.chat_stream.focus_energy
            if context.chat_stream.message_count > 0:
                avg_message_interest = context.chat_stream.message_interest_total / context.chat_stream.message_count

        reply_threshold = getattr(global_config.affinity_flow, 'reply_action_interest_threshold', 0.4)
        non_reply_threshold = getattr(global_config.affinity_flow, 'non_reply_action_interest_threshold', 0.2)
        high_match_threshold = getattr(global_config.affinity_flow, 'high_match_interest_threshold', 0.8)

        base_interval = global_config.chat.dynamic_distribution_base_interval
        min_interval = global_config.chat.dynamic_distribution_min_interval
        max_interval = global_config.chat.dynamic_distribution_max_interval
        jitter_factor = global_config.chat.dynamic_distribution_jitter_factor

        if avg_message_interest >= high_match_threshold:
            interval_multiplier = 0.3 + (focus_energy - 0.7) * 2.0
        elif avg_message_interest >= reply_threshold:
            gap_from_reply = (avg_message_interest - reply_threshold) / (high_match_threshold - reply_threshold)
            interval_multiplier = 0.6 + gap_from_reply * 0.4
        elif avg_message_interest >= non_reply_threshold:
            gap_from_non_reply = (avg_message_interest - non_reply_threshold) / (reply_threshold - non_reply_threshold)
            interval_multiplier = 1.2 + gap_from_non_reply * 1.8
        else:
            gap_ratio = max(0, avg_message_interest / non_reply_threshold)
            interval_multiplier = 3.0 + (1.0 - gap_ratio) * 3.0

        energy_adjustment = 1.0 + (focus_energy - 0.5) * 0.5
        interval = base_interval * interval_multiplier * energy_adjustment

        jitter = random.uniform(1.0 - jitter_factor, 1.0 + jitter_factor)
        final_interval = interval * jitter

        final_interval = max(min_interval, min(max_interval, final_interval))
        return final_interval

    def _calculate_next_manager_delay(self) -> float:
        """计算管理器下次检查的延迟时间"""
        current_time = time.time()
        min_delay = float('inf')

        for context in self.stream_contexts.values():
            if not context.is_active:
                continue

            time_until_check = context.next_check_time - current_time
            if time_until_check > 0:
                min_delay = min(min_delay, time_until_check)
            else:
                return 0.1

        if min_delay == float('inf'):
            return self.check_interval

        return max(0.1, min(min_delay, self.check_interval))

    async def _check_streams_with_individual_intervals(self):
        """检查所有达到检查时间的聊天流"""
        current_time = time.time()
        processed_streams = 0

        for stream_id, context in self.stream_contexts.items():
            if not context.is_active:
                continue

            if current_time >= context.next_check_time:
                context.last_check_time = current_time
                if global_config.chat.dynamic_distribution_enabled:
                    context.distribution_interval = self._calculate_stream_distribution_interval(context)
                else:
                    context.distribution_interval = self.check_interval
                context.next_check_time = current_time + context.distribution_interval

                unread_messages = context.get_unread_messages()
                if not unread_messages:
                    continue

                processed_streams += 1
                
                if global_config.chat.concurrent_message_processing:
                    if global_config.chat.process_by_user_id:
                        user_messages = {}
                        for msg in unread_messages:
                            user_id = msg.user_info.user_id if hasattr(msg, 'user_info') and msg.user_info else 'unknown_user'
                            if user_id not in user_messages:
                                user_messages[user_id] = []
                            user_messages[user_id].append(msg)
                        
                        for user_id, messages in user_messages.items():
                            await self._check_and_handle_interruption(context, stream_id, messages, user_id)
                            if not context.user_processing_tasks.get(user_id) or context.user_processing_tasks[user_id].done():
                                task = asyncio.create_task(self._process_and_send_reply(context, messages))
                                context.user_processing_tasks[user_id] = task
                else:
                    # Fix: Ensure unread_messages is available in this branch
                    all_unread_messages = context.get_unread_messages()
                    if all_unread_messages:
                        if not global_config.chat.concurrent_message_processing:
                             await self._check_and_handle_interruption(context, stream_id, all_unread_messages)
                             if not context.processing_task or context.processing_task.done():
                                 context.processing_task = asyncio.create_task(self._process_stream_messages(stream_id, all_unread_messages))
                        else:
                            await self._check_and_handle_interruption(context, stream_id, all_unread_messages)
                            if not context.processing_task or context.processing_task.done():
                                task = asyncio.create_task(self._process_and_send_reply(context, all_unread_messages))
                                context.processing_task = task
            # The original 'else' block for the 'if current_time >= context.next_check_time:' check
            # was problematic. It seems it tried to process messages even when it wasn't time.
            # Removing it should fix the UnboundLocalError and align with the logic of checking the time first.
    
    async def _process_and_send_reply(self, context: StreamContext, unread_messages: list):
        """在后台处理单批消息并加锁发送 (并发模式专用)"""
        if not self.concurrent_semaphore:
            logger.error("并发信号量未初始化")
            return

        user_id = unread_messages[0].user_info.user_id if global_config.chat.process_by_user_id and unread_messages and hasattr(unread_messages[0], 'user_info') else None

        async with self.concurrent_semaphore:
            try:
                # 思考和发送都在锁内，确保单次回复的原子性
                async with context.send_lock:
                    logger.debug(f"发送任务锁定聊天流 {context.stream_id}，准备处理和回复")
                    
                    results = await self.chatter_manager.process_stream_context(context.stream_id, context, unread_messages)
                    
                    if results.get("success", False):
                        self._clear_specific_unread_messages(context, unread_messages)
                        logger.debug(f"聊天流 {context.stream_id} 并发处理成功，清除了 {len(unread_messages)} 条未读消息")
                    else:
                        logger.warning(f"聊天流 {context.stream_id} 并发处理失败: {results.get('error_message', '未知错误')}")

                    reply_delay = random.uniform(1.5, 3.0)
                    await asyncio.sleep(reply_delay)
                    
                    logger.debug(f"发送任务解锁聊天流 {context.stream_id}")

            except asyncio.CancelledError:
                logger.info(f"用户 {user_id} 的任务被取消")
                self._clear_specific_unread_messages(context, unread_messages) # 取消时也清除消息
                raise
            except Exception as e:
                logger.error(f"后台回复处理任务出错: {e}")
                traceback.print_exc()
                self._clear_specific_unread_messages(context, unread_messages)
            finally:
                if user_id and user_id in context.user_processing_tasks:
                    if context.user_processing_tasks[user_id] is asyncio.current_task():
                        del context.user_processing_tasks[user_id]

    def _clear_specific_unread_messages(self, context: StreamContext, messages_to_clear: list):
        """清除指定上下文中的特定未读消息"""
        if not messages_to_clear:
            return

        message_ids_to_clear = {msg.message_id for msg in messages_to_clear}
        
        context.unread_messages = [msg for msg in context.unread_messages if msg.message_id not in message_ids_to_clear]
        
        for msg in messages_to_clear:
            context.history_messages.append(msg)
        
        if len(context.history_messages) > 100:
            context.history_messages = context.history_messages[-100:]

 
# 创建全局消息管理器实例
message_manager = MessageManager()
