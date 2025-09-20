# -*- coding: utf-8 -*-
"""
异步记忆系统优化器
解决记忆系统阻塞主程序的问题，将同步操作改为异步非阻塞操作
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from src.common.logger import get_logger
from src.config.config import global_config
from src.chat.memory_system.async_instant_memory_wrapper import get_async_instant_memory

logger = get_logger("async_memory_optimizer")


@dataclass
class MemoryTask:
    """记忆任务数据结构"""

    task_id: str
    task_type: str  # "store", "retrieve", "build"
    chat_id: str
    content: str
    priority: int = 1  # 1=低优先级, 2=中优先级, 3=高优先级
    callback: Optional[Callable] = None
    created_at: float = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()


class AsyncMemoryQueue:
    """异步记忆任务队列管理器"""

    def __init__(self, max_workers: int = 3):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.task_queue = asyncio.Queue()
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.completed_tasks: Dict[str, Any] = {}
        self.failed_tasks: Dict[str, str] = {}
        self.is_running = False
        self.worker_tasks: List[asyncio.Task] = []

    async def start(self):
        """启动异步队列处理器"""
        if self.is_running:
            return

        self.is_running = True
        # 启动多个工作协程
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.worker_tasks.append(worker)

        logger.info(f"异步记忆队列已启动，工作线程数: {self.max_workers}")

    async def stop(self):
        """停止队列处理器"""
        self.is_running = False

        # 等待所有工作任务完成
        for task in self.worker_tasks:
            task.cancel()

        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        self.executor.shutdown(wait=True)
        logger.info("异步记忆队列已停止")

    async def _worker(self, worker_name: str):
        """工作协程，处理队列中的任务"""
        logger.info(f"记忆处理工作线程 {worker_name} 启动")

        while self.is_running:
            try:
                # 等待任务，超时1秒避免永久阻塞
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)

                # 执行任务
                await self._execute_task(task, worker_name)

            except asyncio.TimeoutError:
                # 超时正常，继续下一次循环
                continue
            except Exception as e:
                logger.error(f"工作线程 {worker_name} 处理任务时出错: {e}")

    async def _execute_task(self, task: MemoryTask, worker_name: str):
        """执行具体的记忆任务"""
        try:
            logger.debug(f"[{worker_name}] 开始处理任务: {task.task_type} - {task.task_id}")
            start_time = time.time()

            # 根据任务类型执行不同的处理逻辑
            result = None
            if task.task_type == "store":
                result = await self._handle_store_task(task)
            elif task.task_type == "retrieve":
                result = await self._handle_retrieve_task(task)
            elif task.task_type == "build":
                result = await self._handle_build_task(task)
            else:
                raise ValueError(f"未知的任务类型: {task.task_type}")

            # 记录完成的任务
            self.completed_tasks[task.task_id] = result
            execution_time = time.time() - start_time

            logger.debug(f"[{worker_name}] 任务完成: {task.task_id} (耗时: {execution_time:.2f}s)")

            # 执行回调函数
            if task.callback:
                try:
                    if asyncio.iscoroutinefunction(task.callback):
                        await task.callback(result)
                    else:
                        task.callback(result)
                except Exception as e:
                    logger.error(f"任务回调执行失败: {e}")

        except Exception as e:
            error_msg = f"任务执行失败: {e}"
            logger.error(f"[{worker_name}] {error_msg}")
            self.failed_tasks[task.task_id] = error_msg

            # 执行错误回调
            if task.callback:
                try:
                    if asyncio.iscoroutinefunction(task.callback):
                        await task.callback(None)
                    else:
                        task.callback(None)
                except Exception:
                    pass

    @staticmethod
    async def _handle_store_task(task: MemoryTask) -> Any:
        """处理记忆存储任务"""
        # 这里需要根据具体的记忆系统来实现
        # 为了避免循环导入，这里使用延迟导入
        try:
            # 获取包装器实例
            memory_wrapper = get_async_instant_memory(task.chat_id)

            # 使用包装器中的llm_memory实例
            if memory_wrapper and memory_wrapper.llm_memory:
                await memory_wrapper.llm_memory.create_and_store_memory(task.content)
                return True
            else:
                logger.warning(f"无法获取记忆系统实例，存储任务失败: chat_id={task.chat_id}")
                return False
        except Exception as e:
            logger.error(f"记忆存储失败: {e}")
            return False

    @staticmethod
    async def _handle_retrieve_task(task: MemoryTask) -> Any:
        """处理记忆检索任务"""
        try:
            # 获取包装器实例
            memory_wrapper = get_async_instant_memory(task.chat_id)

            # 使用包装器中的llm_memory实例
            if memory_wrapper and memory_wrapper.llm_memory:
                memories = await memory_wrapper.llm_memory.get_memory(task.content)
                return memories or []
            else:
                logger.warning(f"无法获取记忆系统实例，检索任务失败: chat_id={task.chat_id}")
                return []
        except Exception as e:
            logger.error(f"记忆检索失败: {e}")
            return []

    @staticmethod
    async def _handle_build_task(task: MemoryTask) -> Any:
        """处理记忆构建任务（海马体系统）"""
        try:
            # 延迟导入避免循环依赖
            if global_config.memory.enable_memory:
                from src.chat.memory_system.Hippocampus import hippocampus_manager

                if hippocampus_manager._initialized:
                    await hippocampus_manager.build_memory()
                    return True
            return False
        except Exception as e:
            logger.error(f"记忆构建失败: {e}")
            return False

    async def add_task(self, task: MemoryTask) -> str:
        """添加任务到队列"""
        await self.task_queue.put(task)
        self.running_tasks[task.task_id] = task
        logger.debug(f"任务已加入队列: {task.task_type} - {task.task_id}")
        return task.task_id

    def get_task_result(self, task_id: str) -> Optional[Any]:
        """获取任务结果（非阻塞）"""
        return self.completed_tasks.get(task_id)

    def is_task_completed(self, task_id: str) -> bool:
        """检查任务是否完成"""
        return task_id in self.completed_tasks or task_id in self.failed_tasks

    def get_queue_status(self) -> Dict[str, Any]:
        """获取队列状态"""
        return {
            "is_running": self.is_running,
            "queue_size": self.task_queue.qsize(),
            "running_tasks": len(self.running_tasks),
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "worker_count": len(self.worker_tasks),
        }


class NonBlockingMemoryManager:
    """非阻塞记忆管理器"""

    def __init__(self):
        self.queue = AsyncMemoryQueue(max_workers=3)
        self.cache: Dict[str, Any] = {}
        self.cache_ttl: Dict[str, float] = {}
        self.cache_timeout = 300  # 缓存5分钟

    async def initialize(self):
        """初始化管理器"""
        await self.queue.start()
        logger.info("非阻塞记忆管理器已初始化")

    async def shutdown(self):
        """关闭管理器"""
        await self.queue.stop()
        logger.info("非阻塞记忆管理器已关闭")

    async def store_memory_async(self, chat_id: str, content: str, callback: Optional[Callable] = None) -> str:
        """异步存储记忆（非阻塞）"""
        task = MemoryTask(
            task_id=f"store_{chat_id}_{int(time.time() * 1000)}",
            task_type="store",
            chat_id=chat_id,
            content=content,
            priority=1,  # 存储优先级较低
            callback=callback,
        )

        return await self.queue.add_task(task)

    async def retrieve_memory_async(self, chat_id: str, query: str, callback: Optional[Callable] = None) -> str:
        """异步检索记忆（非阻塞）"""
        # 先检查缓存
        cache_key = f"retrieve_{chat_id}_{hash(query)}"
        if self._is_cache_valid(cache_key):
            result = self.cache[cache_key]
            if callback:
                if asyncio.iscoroutinefunction(callback):
                    await callback(result)
                else:
                    callback(result)
            return "cache_hit"

        task = MemoryTask(
            task_id=f"retrieve_{chat_id}_{int(time.time() * 1000)}",
            task_type="retrieve",
            chat_id=chat_id,
            content=query,
            priority=2,  # 检索优先级中等
            callback=self._create_cache_callback(cache_key, callback),
        )

        return await self.queue.add_task(task)

    async def build_memory_async(self, callback: Optional[Callable] = None) -> str:
        """异步构建记忆（非阻塞）"""
        task = MemoryTask(
            task_id=f"build_memory_{int(time.time() * 1000)}",
            task_type="build",
            chat_id="system",
            content="",
            priority=1,  # 构建优先级较低，避免影响用户体验
            callback=callback,
        )

        return await self.queue.add_task(task)

    def _is_cache_valid(self, cache_key: str) -> bool:
        """检查缓存是否有效"""
        if cache_key not in self.cache:
            return False

        return time.time() - self.cache_ttl.get(cache_key, 0) < self.cache_timeout

    def _create_cache_callback(self, cache_key: str, original_callback: Optional[Callable]):
        """创建带缓存的回调函数"""

        async def cache_callback(result):
            # 存储到缓存
            if result is not None:
                self.cache[cache_key] = result
                self.cache_ttl[cache_key] = time.time()

            # 执行原始回调
            if original_callback:
                if asyncio.iscoroutinefunction(original_callback):
                    await original_callback(result)
                else:
                    original_callback(result)

        return cache_callback

    def get_cached_memory(self, chat_id: str, query: str) -> Optional[Any]:
        """获取缓存的记忆（同步，立即返回）"""
        cache_key = f"retrieve_{chat_id}_{hash(query)}"
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]
        return None

    def get_status(self) -> Dict[str, Any]:
        """获取管理器状态"""
        status = self.queue.get_queue_status()
        status.update({"cache_entries": len(self.cache), "cache_timeout": self.cache_timeout})
        return status


# 全局实例
async_memory_manager = NonBlockingMemoryManager()


# 便捷函数
async def store_memory_nonblocking(chat_id: str, content: str) -> str:
    """非阻塞存储记忆的便捷函数"""
    return await async_memory_manager.store_memory_async(chat_id, content)


async def retrieve_memory_nonblocking(chat_id: str, query: str) -> Optional[Any]:
    """非阻塞检索记忆的便捷函数，支持缓存"""
    # 先尝试从缓存获取
    cached_result = async_memory_manager.get_cached_memory(chat_id, query)
    if cached_result is not None:
        return cached_result

    # 缓存未命中，启动异步检索
    await async_memory_manager.retrieve_memory_async(chat_id, query)
    return None  # 返回None表示需要异步获取


async def build_memory_nonblocking() -> str:
    """非阻塞构建记忆的便捷函数"""
    return await async_memory_manager.build_memory_async()
