"""
重构后的动态消息分发管理器
提供高效、智能的消息分发调度功能
"""

import asyncio
import time
from typing import Dict, List, Optional, Set, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from heapq import heappush, heappop
from abc import ABC, abstractmethod

from src.common.logger import get_logger
from src.config.config import global_config
from src.chat.energy_system import energy_manager

logger = get_logger("distribution_manager")


class DistributionPriority(Enum):
    """分发优先级"""
    CRITICAL = 0    # 关键（立即处理）
    HIGH = 1        # 高优先级
    NORMAL = 2      # 正常优先级
    LOW = 3         # 低优先级
    BACKGROUND = 4  # 后台优先级

    def __lt__(self, other: 'DistributionPriority') -> bool:
        """用于优先级比较"""
        return self.value < other.value


@dataclass
class DistributionTask:
    """分发任务"""
    stream_id: str
    priority: DistributionPriority
    energy: float
    message_count: int
    created_time: float = field(default_factory=time.time)
    retry_count: int = 0
    max_retries: int = 3
    task_id: str = field(default_factory=lambda: f"task_{time.time()}_{id(object())}")
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other: 'DistributionTask') -> bool:
        """用于优先队列排序"""
        # 首先按优先级排序
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value

        # 相同优先级按能量排序（能量高的优先）
        if abs(self.energy - other.energy) > 0.01:
            return self.energy > other.energy

        # 最后按创建时间排序（先创建的优先）
        return self.created_time < other.created_time

    def can_retry(self) -> bool:
        """检查是否可以重试"""
        return self.retry_count < self.max_retries

    def get_retry_delay(self, base_delay: float = 5.0) -> float:
        """获取重试延迟"""
        return base_delay * (2 ** min(self.retry_count, 3))


@dataclass
class StreamDistributionState:
    """流分发状态"""
    stream_id: str
    energy: float
    last_distribution_time: float
    next_distribution_time: float
    message_count: int
    consecutive_failures: int = 0
    is_active: bool = True
    total_distributions: int = 0
    total_failures: int = 0
    average_distribution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def should_distribute(self, current_time: float) -> bool:
        """检查是否应该分发"""
        return (self.is_active and
                current_time >= self.next_distribution_time and
                self.message_count > 0)

    def update_distribution_stats(self, distribution_time: float, success: bool) -> None:
        """更新分发统计"""
        if success:
            self.total_distributions += 1
            self.consecutive_failures = 0
        else:
            self.total_failures += 1
            self.consecutive_failures += 1

        # 更新平均分发时间
        total_attempts = self.total_distributions + self.total_failures
        if total_attempts > 0:
            self.average_distribution_time = (
                (self.average_distribution_time * (total_attempts - 1) + distribution_time)
                / total_attempts
            )


class DistributionExecutor(ABC):
    """分发执行器抽象基类"""

    @abstractmethod
    async def execute(self, stream_id: str, context: Dict[str, Any]) -> bool:
        """执行分发

        Args:
            stream_id: 流ID
            context: 分发上下文

        Returns:
            bool: 是否执行成功
        """
        pass

    @abstractmethod
    def get_priority(self, stream_id: str) -> DistributionPriority:
        """获取流优先级

        Args:
            stream_id: 流ID

        Returns:
            DistributionPriority: 优先级
        """
        pass


class DistributionManager:
    """分发管理器 - 统一管理消息分发调度"""

    def __init__(self, max_concurrent_tasks: Optional[int] = None, retry_delay: Optional[float] = None):
        # 流状态管理
        self.stream_states: Dict[str, StreamDistributionState] = {}

        # 任务队列
        self.task_queue: List[DistributionTask] = []
        self.processing_tasks: Set[str] = set()  # 正在处理的stream_id
        self.completed_tasks: List[DistributionTask] = []
        self.failed_tasks: List[DistributionTask] = []

        # 统计信息
        self.stats: Dict[str, Any] = {
            "total_distributed": 0,
            "total_failed": 0,
            "avg_distribution_time": 0.0,
            "current_queue_size": 0,
            "total_created_tasks": 0,
            "total_completed_tasks": 0,
            "total_failed_tasks": 0,
            "total_retry_attempts": 0,
            "peak_queue_size": 0,
            "start_time": time.time(),
            "last_activity_time": time.time(),
        }

        # 配置参数
        self.max_concurrent_tasks = (
            max_concurrent_tasks or
            getattr(global_config.chat, "max_concurrent_distributions", 3)
        )
        self.retry_delay = (
            retry_delay or
            getattr(global_config.chat, "distribution_retry_delay", 5.0)
        )
        self.max_queue_size = getattr(global_config.chat, "max_distribution_queue_size", 1000)
        self.max_history_size = getattr(global_config.chat, "max_task_history_size", 100)

        # 分发执行器
        self.executor: Optional[DistributionExecutor] = None
        self.executor_callbacks: Dict[str, Callable] = {}

        # 事件循环
        self.is_running = False
        self.distribution_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None

        # 性能监控
        self.performance_metrics: Dict[str, List[float]] = {
            "distribution_times": [],
            "queue_sizes": [],
            "processing_counts": [],
        }
        self.max_metrics_size = 1000

        logger.info(f"分发管理器初始化完成 (并发: {self.max_concurrent_tasks}, 重试延迟: {self.retry_delay}s)")

    async def start(self, cleanup_interval: float = 3600.0) -> None:
        """启动分发管理器

        Args:
            cleanup_interval: 清理间隔（秒）
        """
        if self.is_running:
            logger.warning("分发管理器已经在运行")
            return

        self.is_running = True
        self.distribution_task = asyncio.create_task(self._distribution_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop(cleanup_interval))

        logger.info("分发管理器已启动")

    async def stop(self) -> None:
        """停止分发管理器"""
        if not self.is_running:
            return

        self.is_running = False

        # 取消分发任务
        if self.distribution_task and not self.distribution_task.done():
            self.distribution_task.cancel()
            try:
                await self.distribution_task
            except asyncio.CancelledError:
                pass

        # 取消清理任务
        if self.cleanup_task and not self.cleanup_task.done():
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass

        # 取消所有处理中的任务
        for stream_id in list(self.processing_tasks):
            self._cancel_stream_processing(stream_id)

        logger.info("分发管理器已停止")

    def add_stream_message(self, stream_id: str, message_count: int = 1,
                           priority: Optional[DistributionPriority] = None) -> bool:
        """添加流消息

        Args:
            stream_id: 流ID
            message_count: 消息数量
            priority: 指定优先级（可选）

        Returns:
            bool: 是否成功添加
        """
        current_time = time.time()
        self.stats["last_activity_time"] = current_time

        # 检查队列大小限制
        if len(self.task_queue) >= self.max_queue_size:
            logger.warning(f"分发队列已满，拒绝添加: {stream_id}")
            return False

        # 获取或创建流状态
        if stream_id not in self.stream_states:
            self.stream_states[stream_id] = StreamDistributionState(
                stream_id=stream_id,
                energy=0.5,  # 默认能量
                last_distribution_time=current_time,
                next_distribution_time=current_time,
                message_count=0,
            )

        # 更新流状态
        state = self.stream_states[stream_id]
        state.message_count += message_count

        # 计算优先级
        if priority is None:
            priority = self._calculate_priority(state)

        # 创建分发任务
        task = DistributionTask(
            stream_id=stream_id,
            priority=priority,
            energy=state.energy,
            message_count=state.message_count,
        )

        # 添加到任务队列
        heappush(self.task_queue, task)
        self.stats["current_queue_size"] = len(self.task_queue)
        self.stats["peak_queue_size"] = max(self.stats["peak_queue_size"], len(self.task_queue))
        self.stats["total_created_tasks"] += 1

        # 记录性能指标
        self._record_performance_metric("queue_sizes", len(self.task_queue))

        logger.debug(f"添加分发任务: {stream_id} (优先级: {priority.name}, 消息数: {message_count})")
        return True

    def update_stream_energy(self, stream_id: str, energy: float) -> None:
        """更新流能量

        Args:
            stream_id: 流ID
            energy: 新的能量值
        """
        if stream_id in self.stream_states:
            self.stream_states[stream_id].energy = max(0.0, min(1.0, energy))

            # 失效能量管理器缓存
            energy_manager.invalidate_cache(stream_id)

            logger.debug(f"更新流能量: {stream_id} = {energy:.3f}")

    def _calculate_priority(self, state: StreamDistributionState) -> DistributionPriority:
        """计算分发优先级

        Args:
            state: 流状态

        Returns:
            DistributionPriority: 优先级
        """
        energy = state.energy
        message_count = state.message_count
        consecutive_failures = state.consecutive_failures
        total_distributions = state.total_distributions

        # 使用执行器获取优先级（如果设置）
        if self.executor:
            try:
                return self.executor.get_priority(state.stream_id)
            except Exception as e:
                logger.warning(f"获取执行器优先级失败: {e}")

        # 失败次数过多，降低优先级
        if consecutive_failures >= 3:
            return DistributionPriority.BACKGROUND

        # 高分发次数降低优先级
        if total_distributions > 50 and message_count < 2:
            return DistributionPriority.LOW

        # 基于能量和消息数计算优先级
        if energy >= 0.8 and message_count >= 3:
            return DistributionPriority.CRITICAL
        elif energy >= 0.6 or message_count >= 5:
            return DistributionPriority.HIGH
        elif energy >= 0.3 or message_count >= 2:
            return DistributionPriority.NORMAL
        else:
            return DistributionPriority.LOW

    async def _distribution_loop(self):
        """分发主循环"""
        while self.is_running:
            try:
                # 处理任务队列
                await self._process_task_queue()

                # 更新统计信息
                self._update_statistics()

                # 记录性能指标
                self._record_performance_metric("processing_counts", len(self.processing_tasks))

                # 动态调整循环间隔
                queue_size = len(self.task_queue)
                processing_count = len(self.processing_tasks)
                sleep_time = 0.05 if queue_size > 10 or processing_count > 0 else 0.2

                # 短暂休眠
                await asyncio.sleep(sleep_time)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"分发循环出错: {e}", exc_info=True)
                await asyncio.sleep(1.0)

    async def _process_task_queue(self):
        """处理任务队列"""
        current_time = time.time()

        # 检查是否有可用的处理槽位
        available_slots = self.max_concurrent_tasks - len(self.processing_tasks)
        if available_slots <= 0:
            return

        # 处理队列中的任务
        processed_count = 0
        while (self.task_queue and
               processed_count < available_slots and
               len(self.processing_tasks) < self.max_concurrent_tasks):

            task = heappop(self.task_queue)
            self.stats["current_queue_size"] = len(self.task_queue)

            # 检查任务是否仍然有效
            if not self._is_task_valid(task, current_time):
                self._handle_invalid_task(task)
                continue

            # 开始处理任务
            await self._start_task_processing(task)
            processed_count += 1

        # 记录处理统计
        if processed_count > 0:
            logger.debug(f"处理了 {processed_count} 个分发任务")

    def _is_task_valid(self, task: DistributionTask, current_time: float) -> bool:
        """检查任务是否有效

        Args:
            task: 分发任务
            current_time: 当前时间

        Returns:
            bool: 任务是否有效
        """
        state = self.stream_states.get(task.stream_id)
        if not state or not state.is_active:
            return False

        # 检查任务是否已过期
        if current_time - task.created_time > 3600:  # 1小时
            return False

        # 检查是否达到了分发时间
        return state.should_distribute(current_time)

    def _handle_invalid_task(self, task: DistributionTask) -> None:
        """处理无效任务

        Args:
            task: 无效的任务
        """
        logger.debug(f"任务无效，丢弃: {task.stream_id} (创建时间: {task.created_time})")
        # 可以添加到历史记录中用于分析
        if len(self.failed_tasks) < self.max_history_size:
            self.failed_tasks.append(task)

    async def _start_task_processing(self, task: DistributionTask) -> None:
        """开始处理任务

        Args:
            task: 分发任务
        """
        stream_id = task.stream_id
        state = self.stream_states[stream_id]
        current_time = time.time()

        # 标记为处理中
        self.processing_tasks.add(stream_id)
        state.last_distribution_time = current_time

        # 计算下次分发时间
        interval = energy_manager.get_distribution_interval(state.energy)
        state.next_distribution_time = current_time + interval

        # 记录开始处理
        logger.info(f"开始处理分发任务: {stream_id} "
                   f"(能量: {state.energy:.3f}, "
                   f"消息数: {state.message_count}, "
                   f"周期: {interval:.1f}s, "
                   f"重试次数: {task.retry_count})")

        # 创建处理任务
        asyncio.create_task(self._process_distribution_task(task))

    async def _process_distribution_task(self, task: DistributionTask) -> None:
        """处理分发任务

        Args:
            task: 分发任务
        """
        stream_id = task.stream_id
        start_time = time.time()

        try:
            # 调用外部处理函数
            success = await self._execute_distribution(stream_id)

            if success:
                # 处理成功
                self._handle_task_success(task, start_time)
            else:
                # 处理失败
                await self._handle_task_failure(task)

        except Exception as e:
            logger.error(f"处理分发任务失败 {stream_id}: {e}", exc_info=True)
            await self._handle_task_failure(task)

        finally:
            # 清理处理状态
            self.processing_tasks.discard(stream_id)
            self.stats["last_activity_time"] = time.time()

    async def _execute_distribution(self, stream_id: str) -> bool:
        """执行分发（需要外部实现）

        Args:
            stream_id: 流ID

        Returns:
            bool: 是否执行成功
        """
        # 使用执行器处理分发
        if self.executor:
            try:
                state = self.stream_states.get(stream_id)
                context = {
                    "stream_id": stream_id,
                    "energy": state.energy if state else 0.5,
                    "message_count": state.message_count if state else 0,
                    "task_metadata": {},
                }
                return await self.executor.execute(stream_id, context)
            except Exception as e:
                logger.error(f"执行器分发失败 {stream_id}: {e}")
                return False

        # 回退到回调函数
        callback = self.executor_callbacks.get(stream_id)
        if callback:
            try:
                result = callback(stream_id)
                if asyncio.iscoroutine(result):
                    return await result
                return bool(result)
            except Exception as e:
                logger.error(f"回调分发失败 {stream_id}: {e}")
                return False

        # 默认处理
        logger.debug(f"执行分发: {stream_id}")
        return True

    def _handle_task_success(self, task: DistributionTask, start_time: float) -> None:
        """处理任务成功

        Args:
            task: 成功的任务
            start_time: 开始时间
        """
        stream_id = task.stream_id
        state = self.stream_states.get(stream_id)
        distribution_time = time.time() - start_time

        if state:
            # 更新流状态
            state.update_distribution_stats(distribution_time, True)
            state.message_count = 0  # 清空消息计数

            # 更新全局统计
            self.stats["total_distributed"] += 1
            self.stats["total_completed_tasks"] += 1

            # 更新平均分发时间
            if self.stats["total_distributed"] > 0:
                self.stats["avg_distribution_time"] = (
                    (self.stats["avg_distribution_time"] * (self.stats["total_distributed"] - 1) + distribution_time)
                    / self.stats["total_distributed"]
                )

            # 记录性能指标
            self._record_performance_metric("distribution_times", distribution_time)

        # 添加到成功任务历史
        if len(self.completed_tasks) < self.max_history_size:
            self.completed_tasks.append(task)

        logger.info(f"分发任务成功: {stream_id} (耗时: {distribution_time:.2f}s, 重试: {task.retry_count})")

    async def _handle_task_failure(self, task: DistributionTask) -> None:
        """处理任务失败

        Args:
            task: 失败的任务
        """
        stream_id = task.stream_id
        state = self.stream_states.get(stream_id)
        distribution_time = time.time() - task.created_time

        if state:
            # 更新流状态
            state.update_distribution_stats(distribution_time, False)

            # 增加失败计数
            state.consecutive_failures += 1

            # 计算重试延迟
            retry_delay = task.get_retry_delay(self.retry_delay)
            task.retry_count += 1
            self.stats["total_retry_attempts"] += 1

            # 如果还有重试机会，重新添加到队列
            if task.can_retry():
                # 等待重试延迟
                await asyncio.sleep(retry_delay)

                # 重新计算优先级（失败后降低优先级）
                task.priority = DistributionPriority.LOW

                # 重新添加到队列
                heappush(self.task_queue, task)
                self.stats["current_queue_size"] = len(self.task_queue)

                logger.warning(f"分发任务失败，准备重试: {stream_id} "
                             f"(重试次数: {task.retry_count}/{task.max_retries}, "
                             f"延迟: {retry_delay:.1f}s)")
            else:
                # 超过重试次数，标记为不活跃
                state.is_active = False
                self.stats["total_failed"] += 1
                self.stats["total_failed_tasks"] += 1

                # 添加到失败任务历史
                if len(self.failed_tasks) < self.max_history_size:
                    self.failed_tasks.append(task)

                logger.error(f"分发任务最终失败: {stream_id} (重试次数: {task.retry_count})")

    def _cancel_stream_processing(self, stream_id: str) -> None:
        """取消流处理

        Args:
            stream_id: 流ID
        """
        # 从处理集合中移除
        self.processing_tasks.discard(stream_id)

        # 更新流状态
        if stream_id in self.stream_states:
            self.stream_states[stream_id].is_active = False

        logger.info(f"取消流处理: {stream_id}")

    def _update_statistics(self) -> None:
        """更新统计信息"""
        # 更新当前队列大小
        self.stats["current_queue_size"] = len(self.task_queue)

        # 更新运行时间
        if self.is_running:
            self.stats["uptime"] = time.time() - self.stats["start_time"]

        # 更新性能统计
        self.stats["avg_queue_size"] = (
            sum(self.performance_metrics["queue_sizes"]) /
            max(1, len(self.performance_metrics["queue_sizes"]))
        )

        self.stats["avg_processing_count"] = (
            sum(self.performance_metrics["processing_counts"]) /
            max(1, len(self.performance_metrics["processing_counts"]))
        )

    def _record_performance_metric(self, metric_name: str, value: float) -> None:
        """记录性能指标

        Args:
            metric_name: 指标名称
            value: 指标值
        """
        if metric_name in self.performance_metrics:
            metrics = self.performance_metrics[metric_name]
            metrics.append(value)
            # 保持大小限制
            if len(metrics) > self.max_metrics_size:
                metrics.pop(0)

    async def _cleanup_loop(self, interval: float) -> None:
        """清理循环

        Args:
            interval: 清理间隔
        """
        while self.is_running:
            try:
                await asyncio.sleep(interval)
                self._cleanup_expired_data()
                logger.debug(f"清理完成，保留 {len(self.completed_tasks)} 个成功任务，{len(self.failed_tasks)} 个失败任务")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"清理循环出错: {e}")

    def _cleanup_expired_data(self) -> None:
        """清理过期数据"""
        current_time = time.time()
        max_age = 24 * 3600  # 24小时

        # 清理过期的成功任务
        self.completed_tasks = [
            task for task in self.completed_tasks
            if current_time - task.created_time < max_age
        ]

        # 清理过期的失败任务
        self.failed_tasks = [
            task for task in self.failed_tasks
            if current_time - task.created_time < max_age
        ]

        # 清理性能指标
        for metric_name in self.performance_metrics:
            if len(self.performance_metrics[metric_name]) > self.max_metrics_size:
                self.performance_metrics[metric_name] = (
                    self.performance_metrics[metric_name][-self.max_metrics_size:]
                )

    def get_stream_status(self, stream_id: str) -> Optional[Dict[str, Any]]:
        """获取流状态

        Args:
            stream_id: 流ID

        Returns:
            Optional[Dict[str, Any]]: 流状态信息
        """
        if stream_id not in self.stream_states:
            return None

        state = self.stream_states[stream_id]
        current_time = time.time()
        time_until_next = max(0, state.next_distribution_time - current_time)

        return {
            "stream_id": state.stream_id,
            "energy": state.energy,
            "message_count": state.message_count,
            "last_distribution_time": state.last_distribution_time,
            "next_distribution_time": state.next_distribution_time,
            "time_until_next_distribution": time_until_next,
            "consecutive_failures": state.consecutive_failures,
            "total_distributions": state.total_distributions,
            "total_failures": state.total_failures,
            "average_distribution_time": state.average_distribution_time,
            "is_active": state.is_active,
            "is_processing": stream_id in self.processing_tasks,
            "uptime": current_time - state.last_distribution_time,
        }

    def get_queue_status(self) -> Dict[str, Any]:
        """获取队列状态

        Returns:
            Dict[str, Any]: 队列状态信息
        """
        current_time = time.time()
        uptime = current_time - self.stats["start_time"] if self.is_running else 0

        # 分析任务优先级分布
        priority_counts = {}
        for task in self.task_queue:
            priority_name = task.priority.name
            priority_counts[priority_name] = priority_counts.get(priority_name, 0) + 1

        return {
            "queue_size": len(self.task_queue),
            "processing_count": len(self.processing_tasks),
            "max_concurrent": self.max_concurrent_tasks,
            "max_queue_size": self.max_queue_size,
            "is_running": self.is_running,
            "uptime": uptime,
            "priority_distribution": priority_counts,
            "stats": self.stats.copy(),
            "performance_metrics": {
                name: {
                    "count": len(metrics),
                    "avg": sum(metrics) / max(1, len(metrics)),
                    "min": min(metrics) if metrics else 0,
                    "max": max(metrics) if metrics else 0,
                }
                for name, metrics in self.performance_metrics.items()
            },
        }

    def deactivate_stream(self, stream_id: str) -> bool:
        """停用流

        Args:
            stream_id: 流ID

        Returns:
            bool: 是否成功停用
        """
        if stream_id in self.stream_states:
            self.stream_states[stream_id].is_active = False
            # 取消正在处理的任务
            if stream_id in self.processing_tasks:
                self._cancel_stream_processing(stream_id)
            logger.info(f"停用流: {stream_id}")
            return True
        return False

    def activate_stream(self, stream_id: str) -> bool:
        """激活流

        Args:
            stream_id: 流ID

        Returns:
            bool: 是否成功激活
        """
        if stream_id in self.stream_states:
            self.stream_states[stream_id].is_active = True
            self.stream_states[stream_id].consecutive_failures = 0
            self.stream_states[stream_id].next_distribution_time = time.time()
            logger.info(f"激活流: {stream_id}")
            return True
        return False

    def cleanup_inactive_streams(self, max_inactive_hours: int = 24) -> int:
        """清理不活跃的流

        Args:
            max_inactive_hours: 最大不活跃小时数

        Returns:
            int: 清理的流数量
        """
        current_time = time.time()
        max_inactive_seconds = max_inactive_hours * 3600

        inactive_streams = []
        for stream_id, state in self.stream_states.items():
            if (not state.is_active and
                current_time - state.last_distribution_time > max_inactive_seconds and
                state.message_count == 0):
                inactive_streams.append(stream_id)

        for stream_id in inactive_streams:
            del self.stream_states[stream_id]
            # 同时清理处理中的任务
            self.processing_tasks.discard(stream_id)
            logger.debug(f"清理不活跃流: {stream_id}")

        if inactive_streams:
            logger.info(f"清理了 {len(inactive_streams)} 个不活跃流")

        return len(inactive_streams)

    def set_executor(self, executor: DistributionExecutor) -> None:
        """设置分发执行器

        Args:
            executor: 分发执行器实例
        """
        self.executor = executor
        logger.info(f"设置分发执行器: {executor.__class__.__name__}")

    def register_callback(self, stream_id: str, callback: Callable) -> None:
        """注册分发回调

        Args:
            stream_id: 流ID
            callback: 回调函数
        """
        self.executor_callbacks[stream_id] = callback
        logger.debug(f"注册分发回调: {stream_id}")

    def unregister_callback(self, stream_id: str) -> bool:
        """注销分发回调

        Args:
            stream_id: 流ID

        Returns:
            bool: 是否成功注销
        """
        if stream_id in self.executor_callbacks:
            del self.executor_callbacks[stream_id]
            logger.debug(f"注销分发回调: {stream_id}")
            return True
        return False

    def get_task_history(self, limit: int = 50) -> Dict[str, List[Dict[str, Any]]]:
        """获取任务历史

        Args:
            limit: 返回数量限制

        Returns:
            Dict[str, List[Dict[str, Any]]]: 任务历史
        """
        def task_to_dict(task: DistributionTask) -> Dict[str, Any]:
            return {
                "task_id": task.task_id,
                "stream_id": task.stream_id,
                "priority": task.priority.name,
                "energy": task.energy,
                "message_count": task.message_count,
                "created_time": task.created_time,
                "retry_count": task.retry_count,
                "max_retries": task.max_retries,
                "metadata": task.metadata,
            }

        return {
            "completed_tasks": [task_to_dict(task) for task in self.completed_tasks[-limit:]],
            "failed_tasks": [task_to_dict(task) for task in self.failed_tasks[-limit:]],
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要

        Returns:
            Dict[str, Any]: 性能摘要
        """
        current_time = time.time()
        uptime = current_time - self.stats["start_time"]

        # 计算成功率
        total_attempts = self.stats["total_completed_tasks"] + self.stats["total_failed_tasks"]
        success_rate = (
            self.stats["total_completed_tasks"] / max(1, total_attempts)
        ) if total_attempts > 0 else 0.0

        # 计算吞吐量
        throughput = (
            self.stats["total_completed_tasks"] / max(1, uptime / 3600)
        )  # 每小时完成任务数

        return {
            "uptime_hours": uptime / 3600,
            "success_rate": success_rate,
            "throughput_per_hour": throughput,
            "avg_distribution_time": self.stats["avg_distribution_time"],
            "total_retry_attempts": self.stats["total_retry_attempts"],
            "peak_queue_size": self.stats["peak_queue_size"],
            "active_streams": len(self.stream_states),
            "processing_tasks": len(self.processing_tasks),
        }

    def reset_statistics(self) -> None:
        """重置统计信息"""
        self.stats.update({
            "total_distributed": 0,
            "total_failed": 0,
            "avg_distribution_time": 0.0,
            "current_queue_size": len(self.task_queue),
            "total_created_tasks": 0,
            "total_completed_tasks": 0,
            "total_failed_tasks": 0,
            "total_retry_attempts": 0,
            "peak_queue_size": 0,
            "start_time": time.time(),
            "last_activity_time": time.time(),
        })

        # 清空性能指标
        for metrics in self.performance_metrics.values():
            metrics.clear()

        logger.info("分发管理器统计信息已重置")

    def get_all_stream_states(self) -> Dict[str, Dict[str, Any]]:
        """获取所有流状态

        Returns:
            Dict[str, Dict[str, Any]]: 所有流状态
        """
        return {
            stream_id: self.get_stream_status(stream_id)
            for stream_id in self.stream_states.keys()
        }

    def force_process_stream(self, stream_id: str) -> bool:
        """强制处理指定流

        Args:
            stream_id: 流ID

        Returns:
            bool: 是否成功触发处理
        """
        if stream_id not in self.stream_states:
            return False

        state = self.stream_states[stream_id]
        if not state.is_active:
            return False

        # 创建高优先级任务
        task = DistributionTask(
            stream_id=stream_id,
            priority=DistributionPriority.CRITICAL,
            energy=state.energy,
            message_count=state.message_count,
        )

        # 添加到队列
        heappush(self.task_queue, task)
        self.stats["current_queue_size"] = len(self.task_queue)

        logger.info(f"强制处理流: {stream_id}")
        return True


# 全局分发管理器实例
distribution_manager = DistributionManager()