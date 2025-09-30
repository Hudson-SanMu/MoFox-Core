# -*- coding: utf-8 -*-
"""
向量数据库存储接口
为记忆系统提供高效的向量存储和语义搜索能力
"""

import os
import time
import orjson
import asyncio
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass
from datetime import datetime
import threading

import numpy as np
import pandas as pd
from pathlib import Path

from src.common.logger import get_logger
from src.llm_models.utils_model import LLMRequest
from src.config.config import model_config, global_config
from src.chat.memory_system.memory_chunk import MemoryChunk

logger = get_logger(__name__)

# 尝试导入FAISS，如果不可用则使用简单替代
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available, using simple vector storage")


@dataclass
class VectorStorageConfig:
    """向量存储配置"""
    dimension: int = 768
    similarity_threshold: float = 0.8
    index_type: str = "flat"  # flat, ivf, hnsw
    max_index_size: int = 100000
    storage_path: str = "data/memory_vectors"
    auto_save_interval: int = 100  # 每N次操作自动保存
    enable_compression: bool = True


class VectorStorageManager:
    """向量存储管理器"""

    def __init__(self, config: Optional[VectorStorageConfig] = None):
        self.config = config or VectorStorageConfig()
        self.storage_path = Path(self.config.storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # 向量索引
        self.vector_index = None
        self.memory_id_to_index = {}  # memory_id -> vector index
        self.index_to_memory_id = {}  # vector index -> memory_id

        # 内存缓存
        self.memory_cache: Dict[str, MemoryChunk] = {}
        self.vector_cache: Dict[str, List[float]] = {}

        # 统计信息
        self.storage_stats = {
            "total_vectors": 0,
            "index_build_time": 0.0,
            "average_search_time": 0.0,
            "cache_hit_rate": 0.0,
            "total_searches": 0,
            "cache_hits": 0
        }

        # 线程锁
        self._lock = threading.RLock()
        self._operation_count = 0

        # 初始化索引
        self._initialize_index()

        # 嵌入模型
        self.embedding_model: LLMRequest = None

    def _initialize_index(self):
        """初始化向量索引"""
        try:
            if FAISS_AVAILABLE:
                if self.config.index_type == "flat":
                    self.vector_index = faiss.IndexFlatIP(self.config.dimension)
                elif self.config.index_type == "ivf":
                    quantizer = faiss.IndexFlatIP(self.config.dimension)
                    nlist = min(100, max(1, self.config.max_index_size // 1000))
                    self.vector_index = faiss.IndexIVFFlat(quantizer, self.config.dimension, nlist)
                elif self.config.index_type == "hnsw":
                    self.vector_index = faiss.IndexHNSWFlat(self.config.dimension, 32)
                    self.vector_index.hnsw.efConstruction = 40
                else:
                    self.vector_index = faiss.IndexFlatIP(self.config.dimension)
            else:
                # 简单的向量存储实现
                self.vector_index = SimpleVectorIndex(self.config.dimension)

            logger.info(f"✅ 向量索引初始化完成，类型: {self.config.index_type}")

        except Exception as e:
            logger.error(f"❌ 向量索引初始化失败: {e}")
            # 回退到简单实现
            self.vector_index = SimpleVectorIndex(self.config.dimension)

    async def initialize_embedding_model(self):
        """初始化嵌入模型"""
        if self.embedding_model is None:
            self.embedding_model = LLMRequest(
                model_set=model_config.model_task_config.embedding,
                request_type="memory.embedding"
            )
            logger.info("✅ 嵌入模型初始化完成")

    async def store_memories(self, memories: List[MemoryChunk]):
        """存储记忆向量"""
        if not memories:
            return

        start_time = time.time()

        try:
            # 确保嵌入模型已初始化
            await self.initialize_embedding_model()

            # 批量获取嵌入向量
            memory_texts = []

            for memory in memories:
                # 预先缓存记忆，确保后续流程可访问
                self.memory_cache[memory.memory_id] = memory
                if memory.embedding is None:
                    # 如果没有嵌入向量，需要生成
                    text = self._prepare_embedding_text(memory)
                    memory_texts.append((memory.memory_id, text))
                else:
                    # 已有嵌入向量，直接使用
                    await self._add_single_memory(memory, memory.embedding)

            # 批量生成缺失的嵌入向量
            if memory_texts:
                await self._batch_generate_and_store_embeddings(memory_texts)

            # 自动保存检查
            self._operation_count += len(memories)
            if self._operation_count >= self.config.auto_save_interval:
                await self.save_storage()
                self._operation_count = 0

            storage_time = time.time() - start_time
            logger.debug(f"向量存储完成，{len(memories)} 条记忆，耗时 {storage_time:.3f}秒")

        except Exception as e:
            logger.error(f"❌ 向量存储失败: {e}", exc_info=True)

    def _prepare_embedding_text(self, memory: MemoryChunk) -> str:
        """准备用于嵌入的文本"""
        # 构建包含丰富信息的文本
        text_parts = [
            memory.text_content,
            f"类型: {memory.memory_type.value}",
            f"关键词: {', '.join(memory.keywords)}",
            f"标签: {', '.join(memory.tags)}"
        ]

        if memory.metadata.emotional_context:
            text_parts.append(f"情感: {memory.metadata.emotional_context}")

        return " | ".join(text_parts)

    async def _batch_generate_and_store_embeddings(self, memory_texts: List[Tuple[str, str]]):
        """批量生成和存储嵌入向量"""
        if not memory_texts:
            return

        try:
            texts = [text for _, text in memory_texts]
            memory_ids = [memory_id for memory_id, _ in memory_texts]

            # 批量生成嵌入向量
            embeddings = await self._batch_generate_embeddings(memory_ids, texts)

            # 存储向量和记忆
            for memory_id, embedding in embeddings.items():
                if embedding and len(embedding) == self.config.dimension:
                    memory = self.memory_cache.get(memory_id)
                    if memory:
                        await self._add_single_memory(memory, embedding)

        except Exception as e:
            logger.error(f"❌ 批量生成嵌入向量失败: {e}")

    async def _batch_generate_embeddings(self, memory_ids: List[str], texts: List[str]) -> Dict[str, List[float]]:
        """批量生成嵌入向量"""
        if not texts:
            return {}

        results: Dict[str, List[float]] = {}

        try:
            semaphore = asyncio.Semaphore(min(4, max(1, len(texts))))

            async def generate_embedding(memory_id: str, text: str) -> None:
                async with semaphore:
                    try:
                        embedding, _ = await self.embedding_model.get_embedding(text)
                        if embedding and len(embedding) == self.config.dimension:
                            results[memory_id] = embedding
                        else:
                            logger.warning(
                                "嵌入向量维度不匹配: 期望 %d, 实际 %d (memory_id=%s)",
                                self.config.dimension,
                                len(embedding) if embedding else 0,
                                memory_id,
                            )
                            results[memory_id] = []
                    except Exception as exc:
                        logger.warning("生成记忆 %s 的嵌入向量失败: %s", memory_id, exc)
                        results[memory_id] = []

            tasks = [asyncio.create_task(generate_embedding(mid, text)) for mid, text in zip(memory_ids, texts)]
            await asyncio.gather(*tasks, return_exceptions=True)

        except Exception as e:
            logger.error(f"❌ 批量生成嵌入向量失败: {e}")
            for memory_id in memory_ids:
                results.setdefault(memory_id, [])

        return results

    async def _add_single_memory(self, memory: MemoryChunk, embedding: List[float]):
        """添加单个记忆到向量存储"""
        with self._lock:
            try:
                # 规范化向量
                if embedding:
                    embedding = self._normalize_vector(embedding)

                # 添加到缓存
                self.memory_cache[memory.memory_id] = memory
                self.vector_cache[memory.memory_id] = embedding

                # 更新记忆的嵌入向量
                memory.set_embedding(embedding)

                # 添加到向量索引
                if hasattr(self.vector_index, 'add'):
                    # FAISS索引
                    if isinstance(embedding, np.ndarray):
                        vector_array = embedding.reshape(1, -1).astype('float32')
                    else:
                        vector_array = np.array([embedding], dtype='float32')

                    # 特殊处理IVF索引
                    if self.config.index_type == "ivf" and self.vector_index.ntotal == 0:
                        # IVF索引需要先训练
                        logger.debug("训练IVF索引...")
                        self.vector_index.train(vector_array)

                    self.vector_index.add(vector_array)
                    index_id = self.vector_index.ntotal - 1

                else:
                    # 简单索引
                    index_id = self.vector_index.add_vector(embedding)

                # 更新映射关系
                self.memory_id_to_index[memory.memory_id] = index_id
                self.index_to_memory_id[index_id] = memory.memory_id

                # 更新统计
                self.storage_stats["total_vectors"] += 1

            except Exception as e:
                logger.error(f"❌ 添加记忆到向量存储失败: {e}")

    def _normalize_vector(self, vector: List[float]) -> List[float]:
        """L2归一化向量"""
        if not vector:
            return vector

        try:
            vector_array = np.array(vector, dtype=np.float32)
            norm = np.linalg.norm(vector_array)
            if norm == 0:
                return vector

            normalized = vector_array / norm
            return normalized.tolist()

        except Exception as e:
            logger.warning(f"向量归一化失败: {e}")
            return vector

    async def search_similar_memories(
        self,
        query_vector: List[float],
        limit: int = 10,
        user_id: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """搜索相似记忆"""
        start_time = time.time()

        try:
            # 规范化查询向量
            query_vector = self._normalize_vector(query_vector)

            # 执行向量搜索
            with self._lock:
                if hasattr(self.vector_index, 'search'):
                    # FAISS索引
                    if isinstance(query_vector, np.ndarray):
                        query_array = query_vector.reshape(1, -1).astype('float32')
                    else:
                        query_array = np.array([query_vector], dtype='float32')

                    if self.config.index_type == "ivf" and self.vector_index.ntotal > 0:
                        # 设置IVF搜索参数
                        nprobe = min(self.vector_index.nlist, 10)
                        self.vector_index.nprobe = nprobe

                    distances, indices = self.vector_index.search(query_array, min(limit, self.storage_stats["total_vectors"]))
                    distances = distances.flatten().tolist()
                    indices = indices.flatten().tolist()
                else:
                    # 简单索引
                    results = self.vector_index.search(query_vector, limit)
                    distances = [score for _, score in results]
                    indices = [idx for idx, _ in results]

            # 处理搜索结果
            results = []
            for distance, index in zip(distances, indices):
                if index == -1:  # FAISS的无效索引标记
                    continue

                memory_id = self.index_to_memory_id.get(index)
                if memory_id:
                    # 应用用户过滤
                    if user_id:
                        memory = self.memory_cache.get(memory_id)
                        if memory and memory.user_id != user_id:
                            continue

                    similarity = max(0.0, min(1.0, distance))  # 确保在0-1范围内
                    results.append((memory_id, similarity))

            # 更新统计
            search_time = time.time() - start_time
            self.storage_stats["total_searches"] += 1
            self.storage_stats["average_search_time"] = (
                (self.storage_stats["average_search_time"] * (self.storage_stats["total_searches"] - 1) + search_time) /
                self.storage_stats["total_searches"]
            )

            return results[:limit]

        except Exception as e:
            logger.error(f"❌ 向量搜索失败: {e}")
            return []

    async def get_memory_by_id(self, memory_id: str) -> Optional[MemoryChunk]:
        """根据ID获取记忆"""
        # 先检查缓存
        if memory_id in self.memory_cache:
            self.storage_stats["cache_hits"] += 1
            return self.memory_cache[memory_id]

        self.storage_stats["total_searches"] += 1
        return None

    async def update_memory_embedding(self, memory_id: str, new_embedding: List[float]):
        """更新记忆的嵌入向量"""
        with self._lock:
            try:
                if memory_id not in self.memory_id_to_index:
                    logger.warning(f"记忆 {memory_id} 不存在于向量索引中")
                    return

                # 获取旧索引
                old_index = self.memory_id_to_index[memory_id]

                # 删除旧向量（如果支持）
                if hasattr(self.vector_index, 'remove_ids'):
                    try:
                        self.vector_index.remove_ids(np.array([old_index]))
                    except:
                        logger.warning("无法删除旧向量，将直接添加新向量")

                # 规范化新向量
                new_embedding = self._normalize_vector(new_embedding)

                # 添加新向量
                if hasattr(self.vector_index, 'add'):
                    if isinstance(new_embedding, np.ndarray):
                        vector_array = new_embedding.reshape(1, -1).astype('float32')
                    else:
                        vector_array = np.array([new_embedding], dtype='float32')

                    self.vector_index.add(vector_array)
                    new_index = self.vector_index.ntotal - 1
                else:
                    new_index = self.vector_index.add_vector(new_embedding)

                # 更新映射关系
                self.memory_id_to_index[memory_id] = new_index
                self.index_to_memory_id[new_index] = memory_id

                # 更新缓存
                self.vector_cache[memory_id] = new_embedding

                # 更新记忆对象
                memory = self.memory_cache.get(memory_id)
                if memory:
                    memory.set_embedding(new_embedding)

                logger.debug(f"更新记忆 {memory_id} 的嵌入向量")

            except Exception as e:
                logger.error(f"❌ 更新记忆嵌入向量失败: {e}")

    async def delete_memory(self, memory_id: str):
        """删除记忆"""
        with self._lock:
            try:
                if memory_id not in self.memory_id_to_index:
                    return

                # 获取索引
                index = self.memory_id_to_index[memory_id]

                # 从向量索引中删除（如果支持）
                if hasattr(self.vector_index, 'remove_ids'):
                    try:
                        self.vector_index.remove_ids(np.array([index]))
                    except:
                        logger.warning("无法从向量索引中删除，仅从缓存中移除")

                # 删除映射关系
                del self.memory_id_to_index[memory_id]
                if index in self.index_to_memory_id:
                    del self.index_to_memory_id[index]

                # 从缓存中删除
                self.memory_cache.pop(memory_id, None)
                self.vector_cache.pop(memory_id, None)

                # 更新统计
                self.storage_stats["total_vectors"] = max(0, self.storage_stats["total_vectors"] - 1)

                logger.debug(f"删除记忆 {memory_id}")

            except Exception as e:
                logger.error(f"❌ 删除记忆失败: {e}")

    async def save_storage(self):
        """保存向量存储到文件"""
        try:
            logger.info("正在保存向量存储...")

            # 保存记忆缓存
            cache_data = {
                memory_id: memory.to_dict()
                for memory_id, memory in self.memory_cache.items()
            }

            cache_file = self.storage_path / "memory_cache.json"
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(orjson.dumps(cache_data, option=orjson.OPT_INDENT_2).decode('utf-8'))

            # 保存向量缓存
            vector_cache_file = self.storage_path / "vector_cache.json"
            with open(vector_cache_file, 'w', encoding='utf-8') as f:
                f.write(orjson.dumps(self.vector_cache, option=orjson.OPT_INDENT_2).decode('utf-8'))

            # 保存映射关系
            mapping_file = self.storage_path / "id_mapping.json"
            mapping_data = {
                "memory_id_to_index": self.memory_id_to_index,
                "index_to_memory_id": self.index_to_memory_id
            }
            with open(mapping_file, 'w', encoding='utf-8') as f:
                f.write(orjson.dumps(mapping_data, option=orjson.OPT_INDENT_2).decode('utf-8'))

            # 保存FAISS索引（如果可用）
            if FAISS_AVAILABLE and hasattr(self.vector_index, 'save'):
                index_file = self.storage_path / "vector_index.faiss"
                faiss.write_index(self.vector_index, str(index_file))

            # 保存统计信息
            stats_file = self.storage_path / "storage_stats.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                f.write(orjson.dumps(self.storage_stats, option=orjson.OPT_INDENT_2).decode('utf-8'))

            logger.info("✅ 向量存储保存完成")

        except Exception as e:
            logger.error(f"❌ 保存向量存储失败: {e}")

    async def load_storage(self):
        """从文件加载向量存储"""
        try:
            logger.info("正在加载向量存储...")

            # 加载记忆缓存
            cache_file = self.storage_path / "memory_cache.json"
            if cache_file.exists():
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = orjson.loads(f.read())

                self.memory_cache = {
                    memory_id: MemoryChunk.from_dict(memory_data)
                    for memory_id, memory_data in cache_data.items()
                }

            # 加载向量缓存
            vector_cache_file = self.storage_path / "vector_cache.json"
            if vector_cache_file.exists():
                with open(vector_cache_file, 'r', encoding='utf-8') as f:
                    self.vector_cache = orjson.loads(f.read())

            # 加载映射关系
            mapping_file = self.storage_path / "id_mapping.json"
            if mapping_file.exists():
                with open(mapping_file, 'r', encoding='utf-8') as f:
                    mapping_data = orjson.loads(f.read())
                self.memory_id_to_index = mapping_data.get("memory_id_to_index", {})
                self.index_to_memory_id = mapping_data.get("index_to_memory_id", {})

            # 加载FAISS索引（如果可用）
            if FAISS_AVAILABLE:
                index_file = self.storage_path / "vector_index.faiss"
                if index_file.exists() and hasattr(self.vector_index, 'load'):
                    try:
                        loaded_index = faiss.read_index(str(index_file))
                        # 如果索引类型匹配，则替换
                        if type(loaded_index) == type(self.vector_index):
                            self.vector_index = loaded_index
                            logger.info("✅ FAISS索引加载完成")
                        else:
                            logger.warning("索引类型不匹配，重新构建索引")
                            await self._rebuild_index()
                    except Exception as e:
                        logger.warning(f"加载FAISS索引失败: {e}，重新构建")
                        await self._rebuild_index()

            # 加载统计信息
            stats_file = self.storage_path / "storage_stats.json"
            if stats_file.exists():
                with open(stats_file, 'r', encoding='utf-8') as f:
                    self.storage_stats = orjson.loads(f.read())

            # 更新向量计数
            self.storage_stats["total_vectors"] = len(self.memory_id_to_index)

            logger.info(f"✅ 向量存储加载完成，{self.storage_stats['total_vectors']} 个向量")

        except Exception as e:
            logger.error(f"❌ 加载向量存储失败: {e}")

    async def _rebuild_index(self):
        """重建向量索引"""
        try:
            logger.info("正在重建向量索引...")

            # 重新初始化索引
            self._initialize_index()

            # 重新添加所有向量
            for memory_id, embedding in self.vector_cache.items():
                if embedding:
                    memory = self.memory_cache.get(memory_id)
                    if memory:
                        await self._add_single_memory(memory, embedding)

            logger.info("✅ 向量索引重建完成")

        except Exception as e:
            logger.error(f"❌ 重建向量索引失败: {e}")

    async def optimize_storage(self):
        """优化存储"""
        try:
            logger.info("开始向量存储优化...")

            # 清理无效引用
            self._cleanup_invalid_references()

            # 重新构建索引（如果碎片化严重）
            if self.storage_stats["total_vectors"] > 1000:
                await self._rebuild_index()

            # 更新缓存命中率
            if self.storage_stats["total_searches"] > 0:
                self.storage_stats["cache_hit_rate"] = (
                    self.storage_stats["cache_hits"] / self.storage_stats["total_searches"]
                )

            logger.info("✅ 向量存储优化完成")

        except Exception as e:
            logger.error(f"❌ 向量存储优化失败: {e}")

    def _cleanup_invalid_references(self):
        """清理无效引用"""
        with self._lock:
            # 清理无效的memory_id到index的映射
            valid_memory_ids = set(self.memory_cache.keys())
            invalid_memory_ids = set(self.memory_id_to_index.keys()) - valid_memory_ids

            for memory_id in invalid_memory_ids:
                index = self.memory_id_to_index[memory_id]
                del self.memory_id_to_index[memory_id]
                if index in self.index_to_memory_id:
                    del self.index_to_memory_id[index]

            if invalid_memory_ids:
                logger.info(f"清理了 {len(invalid_memory_ids)} 个无效引用")

    def get_storage_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        stats = self.storage_stats.copy()
        if stats["total_searches"] > 0:
            stats["cache_hit_rate"] = stats["cache_hits"] / stats["total_searches"]
        else:
            stats["cache_hit_rate"] = 0.0
        return stats


class SimpleVectorIndex:
    """简单的向量索引实现（当FAISS不可用时的替代方案）"""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.vectors: List[List[float]] = []
        self.vector_ids: List[int] = []
        self.next_id = 0

    def add_vector(self, vector: List[float]) -> int:
        """添加向量"""
        if len(vector) != self.dimension:
            raise ValueError(f"向量维度不匹配，期望 {self.dimension}，实际 {len(vector)}")

        vector_id = self.next_id
        self.vectors.append(vector.copy())
        self.vector_ids.append(vector_id)
        self.next_id += 1

        return vector_id

    def search(self, query_vector: List[float], limit: int) -> List[Tuple[int, float]]:
        """搜索相似向量"""
        if len(query_vector) != self.dimension:
            raise ValueError(f"查询向量维度不匹配，期望 {self.dimension}，实际 {len(query_vector)}")

        results = []

        for i, vector in enumerate(self.vectors):
            similarity = self._calculate_cosine_similarity(query_vector, vector)
            results.append((self.vector_ids[i], similarity))

        # 按相似度排序
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:limit]

    def _calculate_cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """计算余弦相似度"""
        try:
            dot_product = sum(x * y for x, y in zip(v1, v2))
            norm1 = sum(x * x for x in v1) ** 0.5
            norm2 = sum(x * x for x in v2) ** 0.5

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return dot_product / (norm1 * norm2)

        except Exception:
            return 0.0

    @property
    def ntotal(self) -> int:
        """向量总数"""
        return len(self.vectors)