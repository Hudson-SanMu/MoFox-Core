import asyncio
import time
from typing import List, Dict, Any
from dataclasses import dataclass
import threading

from src.common.logger import get_logger
from src.chat.utils.utils import get_embedding
from src.common.vector_db import vector_db_service


logger = get_logger("vector_instant_memory_v2")


@dataclass
class ChatMessage:
    """聊天消息数据结构"""

    message_id: str
    chat_id: str
    content: str
    timestamp: float
    sender: str = "unknown"
    message_type: str = "text"


class VectorInstantMemoryV2:
    """重构的向量瞬时记忆系统 V2

    新设计理念：
    1. 全量存储 - 所有聊天记录都存储为向量
    2. 定时清理 - 定期清理过期记录
    3. 实时匹配 - 新消息与历史记录做向量相似度匹配
    """

    def __init__(self, chat_id: str, retention_hours: int = 24, cleanup_interval: int = 3600):
        """
        初始化向量瞬时记忆系统

        Args:
            chat_id: 聊天ID
            retention_hours: 记忆保留时长(小时)
            cleanup_interval: 清理间隔(秒)
        """
        self.chat_id = chat_id
        self.retention_hours = retention_hours
        self.cleanup_interval = cleanup_interval
        self.collection_name = "instant_memory"

        # 清理任务相关
        self.cleanup_task = None
        self.is_running = True

        # 初始化系统
        self._init_chroma()
        self._start_cleanup_task()

        logger.info(f"向量瞬时记忆系统V2初始化完成: {chat_id} (保留{retention_hours}小时)")

    def _init_chroma(self):
        """使用全局服务初始化向量数据库集合"""
        try:
            # 现在我们只获取集合，而不是创建新的客户端
            vector_db_service.get_or_create_collection(name=self.collection_name, metadata={"hnsw:space": "cosine"})
            logger.info(f"向量记忆集合 '{self.collection_name}' 已准备就绪")
        except Exception as e:
            logger.error(f"获取向量记忆集合失败: {e}")

    def _start_cleanup_task(self):
        """启动定时清理任务"""

        def cleanup_worker():
            while self.is_running:
                try:
                    self._cleanup_expired_messages()
                    time.sleep(self.cleanup_interval)
                except Exception as e:
                    logger.error(f"清理任务异常: {e}")
                    time.sleep(60)  # 异常时等待1分钟再继续

        self.cleanup_task = threading.Thread(target=cleanup_worker, daemon=True)
        self.cleanup_task.start()
        logger.info(f"定时清理任务已启动，间隔{self.cleanup_interval}秒")

    def _cleanup_expired_messages(self):
        """清理过期的聊天记录"""
        try:
            expire_time = time.time() - (self.retention_hours * 3600)

            # 采用 get -> filter -> delete 模式，避免复杂的 where 查询
            # 1. 获取当前 chat_id 的所有文档
            results = vector_db_service.get(
                collection_name=self.collection_name, where={"chat_id": self.chat_id}, include=["metadatas"]
            )

            if not results or not results.get("ids"):
                logger.info(f"chat_id '{self.chat_id}' 没有找到任何记录，无需清理")
                return

            # 2. 在内存中过滤出过期的文档
            expired_ids = []
            metadatas = results.get("metadatas", [])
            ids = results.get("ids", [])

            for i, metadata in enumerate(metadatas):
                if metadata and metadata.get("timestamp", float("inf")) < expire_time:
                    expired_ids.append(ids[i])

            # 3. 如果有过期文档，根据 ID 进行删除
            if expired_ids:
                vector_db_service.delete(collection_name=self.collection_name, ids=expired_ids)
                logger.info(f"为 chat_id '{self.chat_id}' 清理了 {len(expired_ids)} 条过期记录")
            else:
                logger.info(f"chat_id '{self.chat_id}' 没有需要清理的过期记录")

        except Exception as e:
            logger.error(f"清理过期记录失败: {e}")

    async def store_message(self, content: str, sender: str = "user") -> bool:
        """
        存储聊天消息到向量库

        Args:
            content: 消息内容
            sender: 发送者

        Returns:
            bool: 是否存储成功
        """
        if not content.strip():
            return False

        try:
            # 生成消息向量
            message_vector = await get_embedding(content)
            if not message_vector:
                logger.warning(f"消息向量生成失败: {content[:50]}...")
                return False

            message_id = f"{self.chat_id}_{int(time.time() * 1000)}_{hash(content) % 10000}"

            message = ChatMessage(
                message_id=message_id, chat_id=self.chat_id, content=content, timestamp=time.time(), sender=sender
            )

            # 使用新的服务存储
            vector_db_service.add(
                collection_name=self.collection_name,
                embeddings=[message_vector],
                documents=[content],
                metadatas=[
                    {
                        "message_id": message.message_id,
                        "chat_id": message.chat_id,
                        "timestamp": message.timestamp,
                        "sender": message.sender,
                        "message_type": message.message_type,
                    }
                ],
                ids=[message_id],
            )

            logger.debug(f"消息已存储: {content[:50]}...")
            return True

        except Exception as e:
            logger.error(f"存储消息失败: {e}")
            return False

    async def find_similar_messages(
        self, query: str, top_k: int = 5, similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        查找与查询相似的历史消息

        Args:
            query: 查询内容
            top_k: 返回的最相似消息数量
            similarity_threshold: 相似度阈值

        Returns:
            List[Dict]: 相似消息列表，包含content、similarity、timestamp等信息
        """
        if not query.strip():
            return []

        try:
            query_vector = await get_embedding(query)
            if not query_vector:
                return []

            # 使用新的服务进行查询
            results = vector_db_service.query(
                collection_name=self.collection_name,
                query_embeddings=[query_vector],
                n_results=top_k,
                where={"chat_id": self.chat_id},
            )

            if not results.get("documents") or not results["documents"][0]:
                return []

            # 处理搜索结果
            similar_messages = []
            documents = results["documents"][0]
            distances = results["distances"][0] if results["distances"] else []
            metadatas = results["metadatas"][0] if results["metadatas"] else []

            for i, doc in enumerate(documents):
                # 计算相似度（ChromaDB返回距离，需转换）
                distance = distances[i] if i < len(distances) else 1.0
                similarity = 1 - distance

                # 过滤低相似度结果
                if similarity < similarity_threshold:
                    continue

                # 获取元数据
                metadata = metadatas[i] if i < len(metadatas) else {}

                # 安全获取timestamp
                timestamp = metadata.get("timestamp", 0) if isinstance(metadata, dict) else 0
                timestamp = float(timestamp) if isinstance(timestamp, (int, float)) else 0.0

                similar_messages.append(
                    {
                        "content": doc,
                        "similarity": similarity,
                        "timestamp": timestamp,
                        "sender": metadata.get("sender", "unknown") if isinstance(metadata, dict) else "unknown",
                        "message_id": metadata.get("message_id", "") if isinstance(metadata, dict) else "",
                        "time_ago": self._format_time_ago(timestamp),
                    }
                )

            # 按相似度排序
            similar_messages.sort(key=lambda x: x["similarity"], reverse=True)

            logger.debug(f"找到 {len(similar_messages)} 条相似消息 (查询: {query[:30]}...)")
            return similar_messages

        except Exception as e:
            logger.error(f"查找相似消息失败: {e}")
            return []

    @staticmethod
    def _format_time_ago(timestamp: float) -> str:
        """格式化时间差显示"""
        if timestamp <= 0:
            return "未知时间"

        try:
            now = time.time()
            diff = now - timestamp

            if diff < 60:
                return f"{int(diff)}秒前"
            elif diff < 3600:
                return f"{int(diff / 60)}分钟前"
            elif diff < 86400:
                return f"{int(diff / 3600)}小时前"
            else:
                return f"{int(diff / 86400)}天前"
        except Exception:
            return "时间格式错误"

    async def get_memory_for_context(self, current_message: str, context_size: int = 3) -> str:
        """
        获取与当前消息相关的记忆上下文

        Args:
            current_message: 当前消息
            context_size: 上下文消息数量

        Returns:
            str: 格式化的记忆上下文
        """
        similar_messages = await self.find_similar_messages(
            current_message,
            top_k=context_size,
            similarity_threshold=0.6,  # 降低阈值以获得更多上下文
        )

        if not similar_messages:
            return ""

        # 格式化上下文
        context_lines = []
        for msg in similar_messages:
            context_lines.append(
                f"[{msg['time_ago']}] {msg['sender']}: {msg['content']} (相似度: {msg['similarity']:.2f})"
            )

        return "相关的历史记忆:\n" + "\n".join(context_lines)

    def get_stats(self) -> Dict[str, Any]:
        """获取记忆系统统计信息"""
        stats = {
            "chat_id": self.chat_id,
            "retention_hours": self.retention_hours,
            "cleanup_interval": self.cleanup_interval,
            "system_status": "running" if self.is_running else "stopped",
            "total_messages": 0,
            "db_status": "connected",
        }

        try:
            # 注意：count() 现在没有 chat_id 过滤，返回的是整个集合的数量
            # 若要精确计数，需要 get(where={"chat_id": ...}) 然后 len(results['ids'])
            # 这里为了简化，暂时显示集合总数
            result = vector_db_service.count(collection_name=self.collection_name)
            stats["total_messages"] = result
        except Exception:
            stats["total_messages"] = "查询失败"
            stats["db_status"] = "disconnected"

        return stats

    def stop(self):
        """停止记忆系统"""
        self.is_running = False
        if self.cleanup_task and self.cleanup_task.is_alive():
            logger.info("正在停止定时清理任务...")
        logger.info(f"向量瞬时记忆系统已停止: {self.chat_id}")


# 为了兼容现有代码，提供工厂函数
def create_vector_memory_v2(chat_id: str, retention_hours: int = 24) -> VectorInstantMemoryV2:
    """创建向量瞬时记忆系统V2实例"""
    return VectorInstantMemoryV2(chat_id, retention_hours)


# 使用示例
async def demo():
    """使用演示"""
    memory = VectorInstantMemoryV2("demo_chat")

    # 存储一些测试消息
    await memory.store_message("今天天气不错，出去散步了", "用户")
    await memory.store_message("刚才买了个冰淇淋，很好吃", "用户")
    await memory.store_message("明天要开会，有点紧张", "用户")

    # 查找相似消息
    similar = await memory.find_similar_messages("天气怎么样")
    print("相似消息:", similar)

    # 获取上下文
    context = await memory.get_memory_for_context("今天心情如何")
    print("记忆上下文:", context)

    # 查看统计信息
    stats = memory.get_stats()
    print("系统状态:", stats)

    memory.stop()


if __name__ == "__main__":
    asyncio.run(demo())
