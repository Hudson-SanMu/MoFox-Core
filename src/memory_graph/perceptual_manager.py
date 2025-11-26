"""
感知记忆层管理器 (Perceptual Memory Manager)

负责管理全局记忆堆：
- 消息分块处理
- 向量生成
- TopK 召回
- 激活次数统计
- FIFO 淘汰
"""

import asyncio
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.common.logger import get_logger
from src.memory_graph.models import MemoryBlock, PerceptualMemory
from src.memory_graph.utils.embeddings import EmbeddingGenerator
from src.memory_graph.utils.similarity import cosine_similarity_async, batch_cosine_similarity_async

logger = get_logger(__name__)


class PerceptualMemoryManager:
    """
    感知记忆层管理器

    全局单例，管理所有聊天流的感知记忆块。
    """

    def __init__(
        self,
        data_dir: Path | None = None,
        max_blocks: int = 50,
        block_size: int = 5,
        activation_threshold: int = 3,
        recall_top_k: int = 5,
        recall_similarity_threshold: float = 0.55,
        pending_message_ttl: int = 600,
        max_pending_per_stream: int = 50,
        max_pending_messages: int = 2000,
    ):
        """
        初始化感知记忆层管理器

        Args:
            data_dir: 数据存储目录
            max_blocks: 记忆堆最大容量
            block_size: 每个块包含的消息数量
            activation_threshold: 激活阈值（召回次数）
            recall_top_k: 召回时返回的最大块数
            recall_similarity_threshold: 召回的相似度阈值
            pending_message_ttl: 待组块消息最大保留时间（秒）
            max_pending_per_stream: 单个流允许的待组块消息上限
            max_pending_messages: 全部流的待组块消息总上限
        """
        self.data_dir = data_dir or Path("data/memory_graph")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # 配置参数
        self.max_blocks = max_blocks
        self.block_size = block_size
        self.activation_threshold = activation_threshold
        self.recall_top_k = recall_top_k
        self.recall_similarity_threshold = recall_similarity_threshold
        self.pending_message_ttl = max(0, pending_message_ttl)
        self.max_pending_per_stream = max(0, max_pending_per_stream)
        self.max_pending_messages = max(0, max_pending_messages)

        # 核心数据
        self.perceptual_memory: PerceptualMemory | None = None
        self.embedding_generator: EmbeddingGenerator | None = None

        # 状态
        self._initialized = False
        self._save_lock = asyncio.Lock()

    @property
    def memory(self) -> PerceptualMemory:
        """获取感知记忆对象（保证非 None）"""
        if self.perceptual_memory is None:
            raise RuntimeError("感知记忆管理器未初始化")
        return self.perceptual_memory

    async def initialize(self) -> None:
        """初始化管理器"""
        if self._initialized:
            logger.warning("感知记忆管理器已经初始化")
            return

        try:
            logger.debug("开始初始化感知记忆管理器...")

            # 初始化嵌入生成器
            self.embedding_generator = EmbeddingGenerator()

            # 尝试加载现有数据
            await self._load_from_disk()

            # 如果没有加载到数据，创建新的
            if not self.perceptual_memory:
                logger.debug("未找到现有数据，创建新的感知记忆堆")
                self.perceptual_memory = PerceptualMemory(
                    max_blocks=self.max_blocks,
                    block_size=self.block_size,
                )
            else:
                self._cleanup_pending_messages()

            self._initialized = True
            logger.info(
                f"感知记忆管理器初始化完成 "
                f"(已加载 {len(self.perceptual_memory.blocks)} 个记忆块)"
            )

        except Exception as e:
            logger.error(f"感知记忆管理器初始化失败: {e}")
            raise

    async def add_message(self, message: dict[str, Any]) -> MemoryBlock | None:
        """
        添加消息到感知记忆层

        消息会按 stream_id 组织，同一聊天流的消息才能进入同一个记忆块。
        当单个 stream_id 的消息累积到 block_size 条时自动创建记忆块。

        Args:
            message: 消息字典，需包含以下字段：
                - content: str - 消息内容
                - sender_id: str - 发送者ID
                - sender_name: str - 发送者名称
                - timestamp: float - 时间戳
                - stream_id: str - 聊天流ID
                - 其他可选字段

        Returns:
            如果创建了新块，返回 MemoryBlock；否则返回 None
        """
        if not self._initialized:
            await self.initialize()

        try:
            if not hasattr(self.perceptual_memory, "pending_messages"):
                self.perceptual_memory.pending_messages = []

            self._cleanup_pending_messages()

            stream_id = message.get("stream_id", "unknown")
            self._normalize_message_timestamp(message)
            self.perceptual_memory.pending_messages.append(message)
            self._enforce_pending_limits(stream_id)

            logger.debug(
                f"消息已添加到待处理队列 (stream={stream_id[:8]}, "
                f"总数={len(self.perceptual_memory.pending_messages)})"
            )

            # 按 stream_id 检查是否达到创建块的条件
            stream_messages = [
                msg
                for msg in self.perceptual_memory.pending_messages
                if msg.get("stream_id") == stream_id
            ]

            if len(stream_messages) >= self.block_size:
                new_block = await self._create_memory_block(stream_id)
                return new_block

            return None

        except Exception as e:
            logger.error(f"添加消息失败: {e}")
            return None

    async def _create_memory_block(self, stream_id: str) -> MemoryBlock | None:
        """
        从指定 stream_id 的待处理消息创建记忆块

        Args:
            stream_id: 聊天流ID

        Returns:
            新创建的记忆块，失败返回 None
        """
        try:
            self._cleanup_pending_messages()
            # 只取出指定 stream_id 的 block_size 条消息
            stream_messages = [msg for msg in self.perceptual_memory.pending_messages if msg.get("stream_id") == stream_id]
            
            if len(stream_messages) < self.block_size:
                logger.warning(f"stream {stream_id} 的消息不足 {self.block_size} 条，无法创建块")
                return None
            
            # 取前 block_size 条消息
            messages = stream_messages[:self.block_size]
            
            # 从 pending_messages 中移除这些消息
            for msg in messages:
                self.perceptual_memory.pending_messages.remove(msg)

            # 合并消息文本
            combined_text = self._combine_messages(messages)

            # 生成向量
            embedding = await self._generate_embedding(combined_text)

            # 创建记忆块
            block = MemoryBlock(
                id=f"block_{uuid.uuid4().hex[:12]}",
                messages=messages,
                combined_text=combined_text,
                embedding=embedding,
                metadata={"stream_id": stream_id}  # 添加 stream_id 元数据
            )

            # 添加到记忆堆顶部
            self.perceptual_memory.blocks.insert(0, block)

            # 更新所有块的位置
            for i, b in enumerate(self.perceptual_memory.blocks):
                b.position_in_stack = i

            # FIFO 淘汰：如果超过最大容量，移除最旧的块
            if len(self.perceptual_memory.blocks) > self.max_blocks:
                removed_blocks = self.perceptual_memory.blocks[self.max_blocks :]
                self.perceptual_memory.blocks = self.perceptual_memory.blocks[: self.max_blocks]
                logger.debug(f"记忆堆已满，移除 {len(removed_blocks)} 个旧块")

            logger.debug(
                f"✅ 创建新记忆块: {block.id} (stream={stream_id[:8]}, "
                f"堆大小={len(self.perceptual_memory.blocks)}/{self.max_blocks})"
            )

            # 异步保存
            asyncio.create_task(self._save_to_disk())

            return block

        except Exception as e:
            logger.error(f"创建记忆块失败: {e}")
            return None

    def _normalize_message_timestamp(self, message: dict[str, Any]) -> float:
        """确保消息包含 timestamp 字段并返回其值。"""
        raw_ts = message.get("timestamp", message.get("time"))
        try:
            timestamp = float(raw_ts)
        except (TypeError, ValueError):
            timestamp = time.time()
        message["timestamp"] = timestamp
        return timestamp

    def _cleanup_pending_messages(self) -> None:
        """移除过期/超限的待组块消息，避免内存无限增长。"""
        if not self.perceptual_memory or not getattr(self.perceptual_memory, "pending_messages", None):
            return

        pending = self.perceptual_memory.pending_messages
        now = time.time()
        removed = 0

        if self.pending_message_ttl > 0:
            filtered: list[dict[str, Any]] = []
            ttl = float(self.pending_message_ttl)
            for msg in pending:
                ts = msg.get("timestamp") or msg.get("time")
                try:
                    ts_value = float(ts)
                except (TypeError, ValueError):
                    ts_value = time.time()
                msg["timestamp"] = ts_value
                if now - ts_value <= ttl:
                    filtered.append(msg)
                else:
                    removed += 1

            if removed:
                pending[:] = filtered

        # 全局上限，按 FIFO 丢弃最旧的消息
        if self.max_pending_messages > 0 and len(pending) > self.max_pending_messages:
            overflow = len(pending) - self.max_pending_messages
            del pending[:overflow]
            removed += overflow

        if removed:
            logger.debug(f"清理待组块消息 {removed} 条 (剩余 {len(pending)})")

    def _enforce_pending_limits(self, stream_id: str) -> None:
        """保证单个 stream 的待组块消息不超过限制。"""
        if (
            not self.perceptual_memory
            or not getattr(self.perceptual_memory, "pending_messages", None)
            or self.max_pending_per_stream <= 0
        ):
            return

        pending = self.perceptual_memory.pending_messages
        indexes = [
            idx
            for idx, msg in enumerate(pending)
            if msg.get("stream_id") == stream_id
        ]

        overflow = len(indexes) - self.max_pending_per_stream
        if overflow <= 0:
            return

        for idx in reversed(indexes[:overflow]):
            pending.pop(idx)

        logger.warning(
            "stream %s 待组块消息过多，丢弃 %d 条旧消息 (保留 %d 条)",
            stream_id,
            overflow,
            self.max_pending_per_stream,
        )

    def _combine_messages(self, messages: list[dict[str, Any]]) -> str:
        """
        合并多条消息为单一文本

        Args:
            messages: 消息列表

        Returns:
            合并后的文本
        """
        lines = []
        for msg in messages:
            # 兼容新旧字段名
            sender = msg.get("sender_name") or msg.get("sender") or msg.get("sender_id", "Unknown")
            content = msg.get("content", "")
            timestamp = msg.get("timestamp", datetime.now())

            # 格式化时间
            if isinstance(timestamp, (int, float)):
                # Unix 时间戳
                time_str = datetime.fromtimestamp(timestamp).strftime("%H:%M")
            elif isinstance(timestamp, datetime):
                time_str = timestamp.strftime("%H:%M")
            else:
                time_str = str(timestamp)

            lines.append(f"[{time_str}] {sender}: {content}")

        return "\n".join(lines)

    async def _generate_embedding(self, text: str) -> np.ndarray | None:
        """
        生成文本向量

        Args:
            text: 文本内容

        Returns:
            向量数组，失败返回 None
        """
        try:
            if not self.embedding_generator:
                logger.error("嵌入生成器未初始化")
                return None

            embedding = await self.embedding_generator.generate(text)
            return embedding

        except Exception as e:
            logger.error(f"生成向量失败: {e}")
            return None

    async def _generate_embeddings_batch(self, texts: list[str]) -> list[np.ndarray | None]:
        """
        批量生成文本向量

        Args:
            texts: 文本列表

        Returns:
            向量列表，与输入一一对应
        """
        try:
            if not self.embedding_generator:
                logger.error("嵌入生成器未初始化")
                return [None] * len(texts)

            embeddings = await self.embedding_generator.generate_batch(texts)
            return embeddings

        except Exception as e:
            logger.error(f"批量生成向量失败: {e}")
            return [None] * len(texts)

    async def recall_blocks(
        self,
        query_text: str,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
    ) -> list[MemoryBlock]:
        """
        根据查询召回相关记忆块

        Args:
            query_text: 查询文本
            top_k: 返回的最大块数（None 则使用默认值）
            similarity_threshold: 相似度阈值（None 则使用默认值）

        Returns:
            召回的记忆块列表（按相似度降序）
        """
        if not self._initialized:
            await self.initialize()

        top_k = top_k or self.recall_top_k
        similarity_threshold = similarity_threshold or self.recall_similarity_threshold

        try:
            # 生成查询向量
            query_embedding = await self._generate_embedding(query_text)
            if query_embedding is None:
                logger.warning("查询向量生成失败，返回空列表")
                return []

            # 批量计算所有块的相似度（使用异步版本）
            blocks_with_embeddings = [
                block for block in self.perceptual_memory.blocks
                if block.embedding is not None
            ]

            if not blocks_with_embeddings:
                return []

            # 批量计算相似度
            block_embeddings = [block.embedding for block in blocks_with_embeddings]
            similarities = await batch_cosine_similarity_async(query_embedding, block_embeddings)

            # 过滤和排序
            scored_blocks = []
            for block, similarity in zip(blocks_with_embeddings, similarities):
                # 过滤低于阈值的块
                if similarity >= similarity_threshold:
                    scored_blocks.append((block, similarity))

            # 按相似度降序排序
            scored_blocks.sort(key=lambda x: x[1], reverse=True)

            # 取 TopK
            top_blocks = scored_blocks[:top_k]

            # 更新召回计数和位置
            recalled_blocks = []
            for block, similarity in top_blocks:
                block.increment_recall()
                recalled_blocks.append(block)

                # 检查是否达到激活阈值
                if block.recall_count >= self.activation_threshold:
                    logger.debug(
                        f"记忆块 {block.id} 被激活 "
                        f"(召回次数={block.recall_count}, 阈值={self.activation_threshold})"
                    )

            # 将召回的块移到堆顶（保持顺序）
            if recalled_blocks:
                await self._promote_blocks(recalled_blocks)

            # 检查是否有块达到激活阈值（需要转移到短期记忆）
            activated_blocks = [
                block for block in recalled_blocks 
                if block.recall_count >= self.activation_threshold
            ]
            
            if activated_blocks:
                # 设置标记供 unified_manager 处理
                for block in activated_blocks:
                    block.metadata["needs_transfer"] = True

            logger.debug(
                f"召回 {len(recalled_blocks)} 个记忆块 "
                f"(top_k={top_k}, threshold={similarity_threshold:.2f})"
            )

            # 异步保存
            asyncio.create_task(self._save_to_disk())

            return recalled_blocks

        except Exception as e:
            logger.error(f"召回记忆块失败: {e}")
            return []

    async def _promote_blocks(self, blocks_to_promote: list[MemoryBlock]) -> None:
        """
        将召回的块提升到堆顶

        Args:
            blocks_to_promote: 需要提升的块列表
        """
        try:
            # 从原位置移除这些块
            for block in blocks_to_promote:
                if block in self.perceptual_memory.blocks:
                    self.perceptual_memory.blocks.remove(block)

            # 将它们插入到堆顶（保持原有的相对顺序）
            for block in reversed(blocks_to_promote):
                self.perceptual_memory.blocks.insert(0, block)

            # 更新所有块的位置
            for i, block in enumerate(self.perceptual_memory.blocks):
                block.position_in_stack = i

            logger.debug(f"提升 {len(blocks_to_promote)} 个块到堆顶")

        except Exception as e:
            logger.error(f"提升块失败: {e}")

    def get_activated_blocks(self) -> list[MemoryBlock]:
        """
        获取已激活的记忆块（召回次数 >= 激活阈值）

        Returns:
            激活的记忆块列表
        """
        if not self._initialized or not self.perceptual_memory:
            return []

        activated = [
            block
            for block in self.perceptual_memory.blocks
            if block.recall_count >= self.activation_threshold
        ]

        return activated

    async def remove_block(self, block_id: str) -> bool:
        """
        移除指定的记忆块（通常在转为短期记忆后调用）

        Args:
            block_id: 记忆块ID

        Returns:
            是否成功移除
        """
        if not self._initialized:
            await self.initialize()

        try:
            # 查找并移除块
            for i, block in enumerate(self.perceptual_memory.blocks):
                if block.id == block_id:
                    self.perceptual_memory.blocks.pop(i)

                    # 更新剩余块的位置
                    for j, b in enumerate(self.perceptual_memory.blocks):
                        b.position_in_stack = j

                    # 异步保存
                    asyncio.create_task(self._save_to_disk())

                    return True

            logger.warning(f"记忆块不存在: {block_id}")
            return False

        except Exception as e:
            logger.error(f"移除记忆块失败: {e}")
            return False

    def get_statistics(self) -> dict[str, Any]:
        """
        获取感知记忆层统计信息

        Returns:
            统计信息字典
        """
        if not self._initialized or not self.perceptual_memory:
            return {}

        total_messages = sum(len(block.messages) for block in self.perceptual_memory.blocks)
        total_recalls = sum(block.recall_count for block in self.perceptual_memory.blocks)
        activated_count = len(self.get_activated_blocks())

        return {
            "total_blocks": len(self.perceptual_memory.blocks),
            "max_blocks": self.max_blocks,
            "pending_messages": len(self.perceptual_memory.pending_messages),
            "total_messages": total_messages,
            "total_recalls": total_recalls,
            "activated_blocks": activated_count,
            "block_size": self.block_size,
            "activation_threshold": self.activation_threshold,
        }

    async def _save_to_disk(self) -> None:
        """保存感知记忆到磁盘"""
        async with self._save_lock:
            try:
                if not self.perceptual_memory:
                    return

                self._cleanup_pending_messages()

                # 保存到 JSON 文件
                import orjson

                save_path = self.data_dir / "perceptual_memory.json"
                data = self.perceptual_memory.to_dict()

                save_path.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))

                logger.debug(f"感知记忆已保存到 {save_path}")

            except Exception as e:
                logger.error(f"保存感知记忆失败: {e}")

    async def _load_from_disk(self) -> None:
        """从磁盘加载感知记忆"""
        try:
            import orjson

            load_path = self.data_dir / "perceptual_memory.json"

            if not load_path.exists():
                logger.debug("未找到感知记忆数据文件")
                return

            data = orjson.loads(load_path.read_bytes())
            self.perceptual_memory = PerceptualMemory.from_dict(data)

            # 重新加载向量数据
            await self._reload_embeddings()

        except Exception as e:
            logger.error(f"加载感知记忆失败: {e}")

    async def _reload_embeddings(self) -> None:
        """重新生成记忆块的向量"""
        if not self.perceptual_memory:
            return

        blocks_to_process = []
        texts_to_process = []

        for block in self.perceptual_memory.blocks:
            if block.embedding is None and block.combined_text and block.combined_text.strip():
                blocks_to_process.append(block)
                texts_to_process.append(block.combined_text)

        if not blocks_to_process:
            return

        logger.debug(f"开始批量生成 {len(blocks_to_process)} 个块的向量...")

        embeddings = await self._generate_embeddings_batch(texts_to_process)

        success_count = 0
        for block, embedding in zip(blocks_to_process, embeddings):
            if embedding is not None:
                block.embedding = embedding
                success_count += 1

        logger.debug(f"向量重新生成完成（成功: {success_count}/{len(blocks_to_process)}）")

    async def shutdown(self) -> None:
        """关闭管理器"""
        if not self._initialized:
            return

        try:
            logger.debug("正在关闭感知记忆管理器...")

            # 最后一次保存
            await self._save_to_disk()

            self._initialized = False

        except Exception as e:
            logger.error(f"关闭感知记忆管理器失败: {e}")


# 全局单例
_perceptual_manager_instance: PerceptualMemoryManager | None = None


def get_perceptual_manager() -> PerceptualMemoryManager:
    """获取感知记忆管理器单例"""
    global _perceptual_manager_instance
    if _perceptual_manager_instance is None:
        _perceptual_manager_instance = PerceptualMemoryManager()
    return _perceptual_manager_instance
