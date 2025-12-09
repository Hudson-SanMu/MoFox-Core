"""数据库优化层

职责：
- 批量调度
- 多级缓存（内存缓存 + Redis缓存）
- 数据预加载
"""

from .batch_scheduler import (
    AdaptiveBatchScheduler,
    BatchOperation,
    BatchStats,
    Priority,
    close_batch_scheduler,
    get_batch_scheduler,
)
from .cache_backend import CacheBackend
from .cache_backend import CacheStats as BaseCacheStats
from .cache_manager import (
    CacheEntry,
    CacheStats,
    LRUCache,
    MultiLevelCache,
    close_cache,
    get_cache,
    get_cache_backend_type,
)
from .preloader import (
    AccessPattern,
    CommonDataPreloader,
    DataPreloader,
    close_preloader,
    get_preloader,
    record_preload_access,
)
from .redis_cache import RedisCache, close_redis_cache, get_redis_cache

__all__ = [
    "AccessPattern",
    # Batch Scheduler
    "AdaptiveBatchScheduler",
    "BaseCacheStats",
    "BatchOperation",
    "BatchStats",
    # Cache Backend (Abstract)
    "CacheBackend",
    "CacheEntry",
    "CacheStats",
    "CommonDataPreloader",
    # Preloader
    "DataPreloader",
    "LRUCache",
    # Memory Cache
    "MultiLevelCache",
    "Priority",
    # Redis Cache
    "RedisCache",
    "close_batch_scheduler",
    "close_cache",
    "close_preloader",
    "close_redis_cache",
    "get_batch_scheduler",
    "get_cache",
    "get_cache_backend_type",
    "get_preloader",
    "record_preload_access",
    "get_redis_cache"
]
