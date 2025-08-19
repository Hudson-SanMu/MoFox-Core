import asyncio
import time
import json
import hashlib
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass

import numpy as np
import chromadb
from chromadb.config import Settings
from sqlalchemy import select

from src.common.logger import get_logger
from src.chat.utils.utils import get_embedding
from src.config.config import global_config
from src.common.database.sqlalchemy_models import Memory
from src.common.database.sqlalchemy_database_api import get_db_session


logger = get_logger("vector_instant_memory")


@dataclass
class MemoryImportancePattern:
    """记忆重要性模式"""
    description: str
    vector: List[float]
    threshold: float = 0.6


class VectorInstantMemory:
    """基于向量的瞬时记忆系统
    
    完全替换原有的LLM判断方式，使用向量相似度进行：
    1. 记忆重要性判断
    2. 记忆内容去重
    3. 记忆检索匹配
    """
    
    def __init__(self, chat_id: str):
        self.chat_id = chat_id
        self.client = None
        self.collection = None
        self.importance_patterns = []
        self._init_chroma()
        
    def _init_chroma(self):
        """初始化ChromaDB连接"""
        try:
            db_path = f"./data/memory_vectors/{self.chat_id}"
            self.client = chromadb.PersistentClient(
                path=db_path,
                settings=Settings(anonymized_telemetry=False)
            )
            self.collection = self.client.get_or_create_collection(
                name="instant_memories",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"向量记忆数据库初始化成功: {db_path}")
        except Exception as e:
            logger.error(f"ChromaDB初始化失败: {e}")
            self.client = None
            self.collection = None
    
    async def _load_importance_patterns(self):
        """加载重要性判断模式向量"""
        if self.importance_patterns:
            return
            
        patterns = [
            "用户分享了重要的个人信息和经历",
            "讨论了未来的计划、安排或目标", 
            "表达了强烈的情感、观点或态度",
            "询问了重要的问题需要回答",
            "发生了有趣的对话和深入交流",
            "出现了新的话题或重要信息",
            "用户表现出明显的情绪变化",
            "涉及重要的决定或选择"
        ]
        
        try:
            for i, pattern in enumerate(patterns):
                vector = await get_embedding(pattern)
                if vector:
                    self.importance_patterns.append(
                        MemoryImportancePattern(
                            description=pattern,
                            vector=vector,
                            threshold=0.55 + i * 0.01  # 动态阈值
                        )
                    )
                    
            logger.info(f"加载了 {len(self.importance_patterns)} 个重要性判断模式")
        except Exception as e:
            logger.error(f"加载重要性模式失败: {e}")
    
    async def should_create_memory(self, chat_history: str) -> Tuple[bool, float]:
        """向量化判断是否需要创建记忆
        
        Args:
            chat_history: 聊天历史
            
        Returns:
            (是否需要记忆, 重要性分数)
        """
        if not chat_history.strip():
            return False, 0.0
            
        await self._load_importance_patterns()
        
        try:
            # 获取聊天历史的向量表示
            history_vector = await get_embedding(chat_history[-500:])  # 只取最后500字符
            if not history_vector:
                return False, 0.0
            
            # 与重要性模式向量计算相似度
            max_score = 0.0
            best_pattern = None
            
            for pattern in self.importance_patterns:
                similarity = self._cosine_similarity(history_vector, pattern.vector)
                if similarity > max_score:
                    max_score = similarity
                    best_pattern = pattern
            
            should_remember = max_score > 0.6  # 基础阈值
            
            if should_remember and best_pattern:
                logger.debug(f"触发记忆模式: {best_pattern.description} (相似度: {max_score:.3f})")
                
            return should_remember, max_score
            
        except Exception as e:
            logger.error(f"向量化判断记忆重要性失败: {e}")
            return False, 0.0
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        try:
            v1 = np.array(vec1)
            v2 = np.array(vec2)
            
            dot_product = np.dot(v1, v2)
            norms = np.linalg.norm(v1) * np.linalg.norm(v2)
            
            if norms == 0:
                return 0.0
                
            return dot_product / norms
            
        except Exception as e:
            logger.error(f"计算余弦相似度失败: {e}")
            return 0.0
    
    def _extract_key_content(self, chat_history: str) -> str:
        """快速提取关键内容（避免LLM调用）"""
        lines = chat_history.strip().split('\n')
        
        # 简单规则：取最后几行非空对话
        key_lines = []
        for line in reversed(lines):
            if line.strip() and ':' in line:  # 包含发言者格式
                key_lines.insert(0, line.strip())
                if len(key_lines) >= 3:  # 最多3行
                    break
        
        return '\n'.join(key_lines) if key_lines else chat_history[-200:]
    
    async def _is_duplicate_memory(self, content: str) -> bool:
        """检查是否为重复记忆"""
        if not self.collection:
            return False
            
        try:
            content_vector = await get_embedding(content)
            if not content_vector:
                return False
            
            # 查询最相似的记忆
            results = self.collection.query(
                query_embeddings=[content_vector],
                n_results=1
            )
            
            if results['distances'] and results['distances'][0]:
                similarity = 1 - results['distances'][0][0]  # ChromaDB用距离，转换为相似度
                return similarity > 0.85  # 85%相似度认为重复
                
        except Exception as e:
            logger.error(f"检查重复记忆失败: {e}")
            
        return False
    
    async def create_and_store_memory(self, chat_history: str):
        """创建并存储向量化记忆"""
        try:
            # 1. 向量化判断重要性
            should_store, importance_score = await self.should_create_memory(chat_history)
            
            if not should_store:
                logger.debug("聊天内容不需要记忆")
                return
            
            # 2. 提取关键内容
            key_content = self._extract_key_content(chat_history)
            
            # 3. 检查重复
            if await self._is_duplicate_memory(key_content):
                logger.debug("发现重复记忆，跳过存储")
                return
            
            # 4. 向量化存储
            await self._store_vector_memory(key_content, importance_score)
            
            logger.info(f"成功存储向量记忆 (重要性: {importance_score:.3f}): {key_content[:50]}...")
            
        except Exception as e:
            logger.error(f"创建向量记忆失败: {e}")
    
    async def _store_vector_memory(self, content: str, importance: float):
        """存储向量化记忆"""
        if not self.collection:
            logger.warning("ChromaDB未初始化，无法存储向量记忆")
            return
            
        try:
            # 生成向量
            content_vector = await get_embedding(content)
            if not content_vector:
                logger.error("生成记忆向量失败")
                return
            
            # 生成唯一ID
            memory_id = f"{self.chat_id}_{int(time.time() * 1000)}"
            
            # 存储到ChromaDB
            self.collection.add(
                embeddings=[content_vector],
                documents=[content],
                metadatas=[{
                    "chat_id": self.chat_id,
                    "timestamp": time.time(),
                    "importance": importance,
                    "type": "instant_memory"
                }],
                ids=[memory_id]
            )
            
            # 同时存储到原数据库（保持兼容性）
            await self._store_to_db(content, importance)
            
        except Exception as e:
            logger.error(f"存储向量记忆到ChromaDB失败: {e}")
    
    async def _store_to_db(self, content: str, importance: float):
        """存储到原数据库表"""
        try:
            with get_db_session() as session:
                memory = Memory(
                    memory_id=f"{self.chat_id}_{int(time.time() * 1000)}",
                    chat_id=self.chat_id,
                    memory_text=content,
                    keywords=[],  # 向量版本不需要关键词
                    create_time=time.time(),
                    last_view_time=time.time()
                )
                session.add(memory)
                session.commit()
        except Exception as e:
            logger.error(f"存储记忆到数据库失败: {e}")
    
    async def get_memory(self, target: str) -> Optional[str]:
        """向量化检索相关记忆"""
        if not self.collection:
            return await self._fallback_get_memory(target)
            
        try:
            target_vector = await get_embedding(target)
            if not target_vector:
                return None
            
            # 向量相似度搜索
            results = self.collection.query(
                query_embeddings=[target_vector],
                n_results=3,  # 取前3个最相关的
                where={"chat_id": self.chat_id}
            )
            
            if not results['documents'] or not results['documents'][0]:
                return None
            
            # 返回最相关的记忆
            best_memory = results['documents'][0][0]
            distance = results['distances'][0][0] if results['distances'] else 1.0
            similarity = 1 - distance
            
            if similarity > 0.7:  # 70%相似度阈值
                logger.debug(f"找到相关记忆 (相似度: {similarity:.3f}): {best_memory[:50]}...")
                return best_memory
            
            return None
            
        except Exception as e:
            logger.error(f"向量检索记忆失败: {e}")
            return await self._fallback_get_memory(target)
    
    async def _fallback_get_memory(self, target: str) -> Optional[str]:
        """回退到数据库检索"""
        try:
            with get_db_session() as session:
                query = session.execute(select(Memory).where(
                    Memory.chat_id == self.chat_id
                ).order_by(Memory.create_time.desc()).limit(10)).scalars()
                
                memories = list(query)
                
                # 简单的关键词匹配
                for memory in memories:
                    if any(word in memory.memory_text for word in target.split() if len(word) > 1):
                        return memory.memory_text
                        
                return memories[0].memory_text if memories else None
            
        except Exception as e:
            logger.error(f"回退检索记忆失败: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """获取记忆统计信息"""
        stats = {
            "chat_id": self.chat_id,
            "vector_enabled": self.collection is not None,
            "total_memories": 0,
            "importance_patterns": len(self.importance_patterns)
        }
        
        if self.collection:
            try:
                result = self.collection.count()
                stats["total_memories"] = result
            except:
                pass
                
        return stats