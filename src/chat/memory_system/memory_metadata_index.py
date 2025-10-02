# -*- coding: utf-8 -*-
"""
记忆元数据索引管理器
使用JSON文件存储记忆元数据，支持快速模糊搜索和过滤
"""

import orjson
import threading
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, asdict
from datetime import datetime

from src.common.logger import get_logger
from src.chat.memory_system.memory_chunk import MemoryType, ImportanceLevel, ConfidenceLevel

logger = get_logger(__name__)


@dataclass
class MemoryMetadataIndexEntry:
    """元数据索引条目（轻量级，只用于快速过滤）"""
    memory_id: str
    user_id: str
    
    # 分类信息
    memory_type: str  # MemoryType.value
    subjects: List[str]  # 主语列表
    objects: List[str]  # 宾语列表
    keywords: List[str]  # 关键词列表
    tags: List[str]  # 标签列表
    
    # 数值字段（用于范围过滤）
    importance: int  # ImportanceLevel.value (1-4)
    confidence: int  # ConfidenceLevel.value (1-4)
    created_at: float  # 创建时间戳
    access_count: int  # 访问次数
    
    # 可选字段
    chat_id: Optional[str] = None
    content_preview: Optional[str] = None  # 内容预览（前100字符）


class MemoryMetadataIndex:
    """记忆元数据索引管理器"""
    
    def __init__(self, index_file: str = "data/memory_metadata_index.json"):
        self.index_file = Path(index_file)
        self.index: Dict[str, MemoryMetadataIndexEntry] = {}  # memory_id -> entry
        
        # 倒排索引（用于快速查找）
        self.type_index: Dict[str, Set[str]] = {}  # type -> {memory_ids}
        self.subject_index: Dict[str, Set[str]] = {}  # subject -> {memory_ids}
        self.keyword_index: Dict[str, Set[str]] = {}  # keyword -> {memory_ids}
        self.tag_index: Dict[str, Set[str]] = {}  # tag -> {memory_ids}
        
        self.lock = threading.RLock()
        
        # 加载已有索引
        self._load_index()
    
    def _load_index(self):
        """从文件加载索引"""
        if not self.index_file.exists():
            logger.info(f"元数据索引文件不存在，将创建新索引: {self.index_file}")
            return
        
        try:
            with open(self.index_file, 'rb') as f:
                data = orjson.loads(f.read())
            
            # 重建内存索引
            for entry_data in data.get('entries', []):
                entry = MemoryMetadataIndexEntry(**entry_data)
                self.index[entry.memory_id] = entry
                self._update_inverted_indices(entry)
            
            logger.info(f"✅ 加载元数据索引: {len(self.index)} 条记录")
            
        except Exception as e:
            logger.error(f"加载元数据索引失败: {e}", exc_info=True)
    
    def _save_index(self):
        """保存索引到文件"""
        try:
            # 确保目录存在
            self.index_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 序列化所有条目
            entries = [asdict(entry) for entry in self.index.values()]
            data = {
                'version': '1.0',
                'count': len(entries),
                'last_updated': datetime.now().isoformat(),
                'entries': entries
            }
            
            # 写入文件（使用临时文件 + 原子重命名）
            temp_file = self.index_file.with_suffix('.tmp')
            with open(temp_file, 'wb') as f:
                f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))
            
            temp_file.replace(self.index_file)
            logger.debug(f"元数据索引已保存: {len(entries)} 条记录")
            
        except Exception as e:
            logger.error(f"保存元数据索引失败: {e}", exc_info=True)
    
    def _update_inverted_indices(self, entry: MemoryMetadataIndexEntry):
        """更新倒排索引"""
        memory_id = entry.memory_id
        
        # 类型索引
        self.type_index.setdefault(entry.memory_type, set()).add(memory_id)
        
        # 主语索引
        for subject in entry.subjects:
            subject_norm = subject.strip().lower()
            if subject_norm:
                self.subject_index.setdefault(subject_norm, set()).add(memory_id)
        
        # 关键词索引
        for keyword in entry.keywords:
            keyword_norm = keyword.strip().lower()
            if keyword_norm:
                self.keyword_index.setdefault(keyword_norm, set()).add(memory_id)
        
        # 标签索引
        for tag in entry.tags:
            tag_norm = tag.strip().lower()
            if tag_norm:
                self.tag_index.setdefault(tag_norm, set()).add(memory_id)
    
    def add_or_update(self, entry: MemoryMetadataIndexEntry):
        """添加或更新索引条目"""
        with self.lock:
            # 如果已存在，先从倒排索引中移除旧记录
            if entry.memory_id in self.index:
                self._remove_from_inverted_indices(entry.memory_id)
            
            # 添加新记录
            self.index[entry.memory_id] = entry
            self._update_inverted_indices(entry)
    
    def _remove_from_inverted_indices(self, memory_id: str):
        """从倒排索引中移除记录"""
        if memory_id not in self.index:
            return
        
        entry = self.index[memory_id]
        
        # 从类型索引移除
        if entry.memory_type in self.type_index:
            self.type_index[entry.memory_type].discard(memory_id)
        
        # 从主语索引移除
        for subject in entry.subjects:
            subject_norm = subject.strip().lower()
            if subject_norm in self.subject_index:
                self.subject_index[subject_norm].discard(memory_id)
        
        # 从关键词索引移除
        for keyword in entry.keywords:
            keyword_norm = keyword.strip().lower()
            if keyword_norm in self.keyword_index:
                self.keyword_index[keyword_norm].discard(memory_id)
        
        # 从标签索引移除
        for tag in entry.tags:
            tag_norm = tag.strip().lower()
            if tag_norm in self.tag_index:
                self.tag_index[tag_norm].discard(memory_id)
    
    def remove(self, memory_id: str):
        """移除索引条目"""
        with self.lock:
            if memory_id in self.index:
                self._remove_from_inverted_indices(memory_id)
                del self.index[memory_id]
    
    def batch_add_or_update(self, entries: List[MemoryMetadataIndexEntry]):
        """批量添加或更新"""
        with self.lock:
            for entry in entries:
                self.add_or_update(entry)
    
    def save(self):
        """保存索引到磁盘"""
        with self.lock:
            self._save_index()
    
    def search(
        self,
        memory_types: Optional[List[str]] = None,
        subjects: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        importance_min: Optional[int] = None,
        importance_max: Optional[int] = None,
        created_after: Optional[float] = None,
        created_before: Optional[float] = None,
        user_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[str]:
        """
        搜索符合条件的记忆ID列表（支持模糊匹配）
        
        Returns:
            List[str]: 符合条件的 memory_id 列表
        """
        with self.lock:
            # 初始候选集（所有记忆）
            candidate_ids: Optional[Set[str]] = None
            
            # 用户过滤（必选）
            if user_id:
                candidate_ids = {
                    mid for mid, entry in self.index.items()
                    if entry.user_id == user_id
                }
            else:
                candidate_ids = set(self.index.keys())
            
            # 类型过滤（OR关系）
            if memory_types:
                type_ids = set()
                for mtype in memory_types:
                    type_ids.update(self.type_index.get(mtype, set()))
                candidate_ids &= type_ids
            
            # 主语过滤（OR关系，支持模糊匹配）
            if subjects:
                subject_ids = set()
                for subject in subjects:
                    subject_norm = subject.strip().lower()
                    # 精确匹配
                    if subject_norm in self.subject_index:
                        subject_ids.update(self.subject_index[subject_norm])
                    # 模糊匹配（包含）
                    for indexed_subject, ids in self.subject_index.items():
                        if subject_norm in indexed_subject or indexed_subject in subject_norm:
                            subject_ids.update(ids)
                candidate_ids &= subject_ids
            
            # 关键词过滤（OR关系，支持模糊匹配）
            if keywords:
                keyword_ids = set()
                for keyword in keywords:
                    keyword_norm = keyword.strip().lower()
                    # 精确匹配
                    if keyword_norm in self.keyword_index:
                        keyword_ids.update(self.keyword_index[keyword_norm])
                    # 模糊匹配（包含）
                    for indexed_keyword, ids in self.keyword_index.items():
                        if keyword_norm in indexed_keyword or indexed_keyword in keyword_norm:
                            keyword_ids.update(ids)
                candidate_ids &= keyword_ids
            
            # 标签过滤（OR关系）
            if tags:
                tag_ids = set()
                for tag in tags:
                    tag_norm = tag.strip().lower()
                    tag_ids.update(self.tag_index.get(tag_norm, set()))
                candidate_ids &= tag_ids
            
            # 重要性过滤
            if importance_min is not None or importance_max is not None:
                importance_ids = {
                    mid for mid in candidate_ids
                    if (importance_min is None or self.index[mid].importance >= importance_min)
                    and (importance_max is None or self.index[mid].importance <= importance_max)
                }
                candidate_ids &= importance_ids
            
            # 时间范围过滤
            if created_after is not None or created_before is not None:
                time_ids = {
                    mid for mid in candidate_ids
                    if (created_after is None or self.index[mid].created_at >= created_after)
                    and (created_before is None or self.index[mid].created_at <= created_before)
                }
                candidate_ids &= time_ids
            
            # 转换为列表并排序（按创建时间倒序）
            result_ids = sorted(
                candidate_ids,
                key=lambda mid: self.index[mid].created_at,
                reverse=True
            )
            
            # 限制数量
            if limit:
                result_ids = result_ids[:limit]
            
            logger.debug(
                f"元数据索引搜索: types={memory_types}, subjects={subjects}, "
                f"keywords={keywords}, 返回={len(result_ids)}条"
            )
            
            return result_ids
    
    def get_entry(self, memory_id: str) -> Optional[MemoryMetadataIndexEntry]:
        """获取单个索引条目"""
        return self.index.get(memory_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取索引统计信息"""
        with self.lock:
            return {
                'total_memories': len(self.index),
                'types': {mtype: len(ids) for mtype, ids in self.type_index.items()},
                'subjects_count': len(self.subject_index),
                'keywords_count': len(self.keyword_index),
                'tags_count': len(self.tag_index),
            }
