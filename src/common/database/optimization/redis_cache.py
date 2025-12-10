"""Redis 缓存后端实现

基于 redis-py 的异步 Redis 缓存实现，支持：
- 异步连接池
- 自动序列化/反序列化
- TTL 过期管理
- 模式删除
- 批量操作
- 统计信息
"""

import asyncio
import json
import pickle
from typing import Any

from src.common.database.optimization.cache_backend import CacheBackend, CacheStats
from src.common.logger import get_logger

logger = get_logger("redis_cache")

import redis.asyncio as aioredis
from redis.asyncio.connection import Connection, SSLConnection


class RedisCache(CacheBackend):
    """Redis 缓存后端

    特性：
    - 分布式缓存：支持多实例共享
    - 自动序列化：支持 JSON 和 Pickle
    - TTL 管理：Redis 原生过期机制
    - 模式删除：支持通配符删除
    - 连接池：高效连接复用
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        password: str | None = None,
        db: int = 0,
        key_prefix: str = "mofox:",
        default_ttl: int = 600,
        pool_size: int = 10,
        socket_timeout: float = 5.0,
        ssl: bool = False,
        serializer: str = "json",  # "json" 或 "pickle"
    ):
        """初始化 Redis 缓存

        Args:
            host: Redis 服务器地址
            port: Redis 服务器端口
            password: Redis 密码（可选）
            db: Redis 数据库编号
            key_prefix: 缓存键前缀
            default_ttl: 默认过期时间（秒）
            pool_size: 连接池大小
            socket_timeout: socket 超时时间（秒）
            ssl: 是否启用 SSL
            serializer: 序列化方式（json 或 pickle）
        """

        self.host = host
        self.port = port
        self.password = password if password else None
        self.db = db
        self.key_prefix = key_prefix
        self.default_ttl = default_ttl
        self.pool_size = pool_size
        self.socket_timeout = socket_timeout
        self.ssl = ssl
        self.serializer = serializer

        # 连接池和客户端（延迟初始化）
        self._pool: Any = None
        self._client: Any = None
        self._lock = asyncio.Lock()
        self._is_closing = False

        # 统计信息
        self._stats = CacheStats()
        self._stats_lock = asyncio.Lock()

        logger.info(
            f"Redis 缓存初始化: {host}:{port}/{db}, "
            f"前缀={key_prefix}, TTL={default_ttl}s, "
            f"序列化={serializer}"
        )

    async def _ensure_connection(self) -> Any:
        """确保 Redis 连接已建立"""
        if self._client is not None:
            return self._client

        async with self._lock:
            if self._client is not None:
                return self._client

            try:
                # redis-py 7.x+ 使用 connection_class 来指定 SSL 连接
                # 不再支持直接传递 ssl=True/False 给 ConnectionPool
                connection_class = SSLConnection if self.ssl else Connection

                # 创建连接池
                self._pool = aioredis.ConnectionPool(
                    host=self.host,
                    port=self.port,
                    password=self.password,
                    db=self.db,
                    max_connections=self.pool_size,
                    socket_timeout=self.socket_timeout,
                    socket_connect_timeout=self.socket_timeout,
                    decode_responses=False,  # 我们自己处理序列化
                    connection_class=connection_class,
                )

                # 创建客户端
                self._client = aioredis.Redis(connection_pool=self._pool)

                # 测试连接
                await self._client.ping()
                logger.info(f"Redis 连接成功: {self.host}:{self.port}/{self.db}")

                return self._client

            except Exception as e:
                logger.error(f"Redis 连接失败: {e}")
                self._client = None
                self._pool = None
                raise

    def _make_key(self, key: str) -> str:
        """生成带前缀的完整键名"""
        return f"{self.key_prefix}{key}"

    def _serialize(self, value: Any) -> bytes:
        """序列化值"""
        if self.serializer == "json":
            try:
                return json.dumps(value, ensure_ascii=False, default=str).encode("utf-8")
            except (TypeError, ValueError):
                # JSON 序列化失败，回退到 pickle
                return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)

    def _deserialize(self, data: bytes) -> Any:
        """反序列化值"""
        if self.serializer == "json":
            try:
                return json.loads(data.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                # JSON 反序列化失败，尝试 pickle
                try:
                    return pickle.loads(data)
                except Exception:
                    return None
        else:
            try:
                return pickle.loads(data)
            except Exception:
                return None

    async def get(self, key: str) -> Any | None:
        """从缓存获取数据"""
        try:
            client = await self._ensure_connection()
            full_key = self._make_key(key)

            data = await client.get(full_key)

            async with self._stats_lock:
                if data is not None:
                    self._stats.hits += 1
                    return self._deserialize(data)
                else:
                    self._stats.misses += 1
                    return None

        except Exception as e:
            logger.error(f"Redis GET 失败 [{key}]: {e}")
            async with self._stats_lock:
                self._stats.misses += 1
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: float | None = None,
    ) -> None:
        """设置缓存值"""
        try:
            client = await self._ensure_connection()
            full_key = self._make_key(key)
            data = self._serialize(value)

            # 使用 TTL
            expire_time = int(ttl) if ttl is not None else self.default_ttl

            await client.setex(full_key, expire_time, data)

            logger.debug(f"Redis SET: {key} (TTL={expire_time}s)")

        except Exception as e:
            logger.error(f"Redis SET 失败 [{key}]: {e}")

    async def delete(self, key: str) -> bool:
        """删除缓存条目"""
        try:
            client = await self._ensure_connection()
            full_key = self._make_key(key)

            result = await client.delete(full_key)

            if result > 0:
                async with self._stats_lock:
                    self._stats.evictions += 1
                logger.debug(f"Redis DEL: {key}")
                return True
            return False

        except Exception as e:
            logger.error(f"Redis DEL 失败 [{key}]: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """检查键是否存在"""
        try:
            client = await self._ensure_connection()
            full_key = self._make_key(key)
            return bool(await client.exists(full_key))
        except Exception as e:
            logger.error(f"Redis EXISTS 失败 [{key}]: {e}")
            return False

    async def clear(self) -> None:
        """清空所有带前缀的缓存"""
        try:
            client = await self._ensure_connection()
            pattern = self._make_key("*")

            # 使用 SCAN 避免阻塞
            cursor = 0
            deleted_count = 0

            while True:
                cursor, keys = await client.scan(cursor, match=pattern, count=100)
                if keys:
                    await client.delete(*keys)
                    deleted_count += len(keys)

                if cursor == 0:
                    break

            async with self._stats_lock:
                self._stats = CacheStats()

            logger.info(f"Redis 缓存已清空: 删除 {deleted_count} 个键")

        except Exception as e:
            logger.error(f"Redis CLEAR 失败: {e}")

    async def delete_pattern(self, pattern: str) -> int:
        """删除匹配模式的所有键

        Args:
            pattern: 键模式（支持 * 通配符）

        Returns:
            删除的键数量
        """
        try:
            client = await self._ensure_connection()
            full_pattern = self._make_key(pattern)

            # 使用 SCAN 避免阻塞
            cursor = 0
            deleted_count = 0

            while True:
                cursor, keys = await client.scan(cursor, match=full_pattern, count=100)
                if keys:
                    await client.delete(*keys)
                    deleted_count += len(keys)

                if cursor == 0:
                    break

            async with self._stats_lock:
                self._stats.evictions += deleted_count

            logger.debug(f"Redis 模式删除: {pattern} -> {deleted_count} 个键")
            return deleted_count

        except Exception as e:
            logger.error(f"Redis 模式删除失败 [{pattern}]: {e}")
            return 0

    async def mget(self, keys: list[str]) -> dict[str, Any]:
        """批量获取多个键的值"""
        if not keys:
            return {}

        try:
            client = await self._ensure_connection()
            full_keys = [self._make_key(k) for k in keys]

            values = await client.mget(full_keys)

            result = {}
            hits = 0
            misses = 0

            for key, value in zip(keys, values):
                if value is not None:
                    result[key] = self._deserialize(value)
                    hits += 1
                else:
                    misses += 1

            async with self._stats_lock:
                self._stats.hits += hits
                self._stats.misses += misses

            return result

        except Exception as e:
            logger.error(f"Redis MGET 失败: {e}")
            return {}

    async def mset(
        self,
        mapping: dict[str, Any],
        ttl: float | None = None,
    ) -> None:
        """批量设置多个键值对"""
        if not mapping:
            return

        try:
            client = await self._ensure_connection()
            expire_time = int(ttl) if ttl is not None else self.default_ttl

            # 使用 pipeline 提高效率
            async with client.pipeline(transaction=False) as pipe:
                for key, value in mapping.items():
                    full_key = self._make_key(key)
                    data = self._serialize(value)
                    pipe.setex(full_key, expire_time, data)

                await pipe.execute()

            logger.debug(f"Redis MSET: {len(mapping)} 个键")

        except Exception as e:
            logger.error(f"Redis MSET 失败: {e}")

    async def get_stats(self) -> dict[str, Any]:
        """获取缓存统计信息"""
        try:
            client = await self._ensure_connection()

            # 获取 Redis 服务器信息
            info = await client.info("memory")
            # keyspace_info 可用于扩展统计, 暂时不获取避免开销
            # keyspace_info = await client.info("keyspace")

            # 统计带前缀的键数量
            pattern = self._make_key("*")
            key_count = 0
            cursor = 0

            while True:
                cursor, keys = await client.scan(cursor, match=pattern, count=1000)
                key_count += len(keys)
                if cursor == 0:
                    break

            async with self._stats_lock:
                return {
                    "backend": "redis",
                    "hits": self._stats.hits,
                    "misses": self._stats.misses,
                    "hit_rate": self._stats.hit_rate,
                    "evictions": self._stats.evictions,
                    "key_count": key_count,
                    "redis_memory_used_mb": info.get("used_memory", 0) / (1024 * 1024),
                    "redis_memory_peak_mb": info.get("used_memory_peak", 0) / (1024 * 1024),
                    "redis_connected_clients": info.get("connected_clients", 0),
                    "key_prefix": self.key_prefix,
                    "default_ttl": self.default_ttl,
                }

        except Exception as e:
            logger.error(f"获取 Redis 统计信息失败: {e}")
            async with self._stats_lock:
                return {
                    "backend": "redis",
                    "hits": self._stats.hits,
                    "misses": self._stats.misses,
                    "hit_rate": self._stats.hit_rate,
                    "evictions": self._stats.evictions,
                    "error": str(e),
                }

    async def close(self) -> None:
        """关闭 Redis 连接"""
        self._is_closing = True

        if self._client is not None:
            try:
                await self._client.aclose()
                logger.info("Redis 连接已关闭")
            except Exception as e:
                logger.error(f"关闭 Redis 连接失败: {e}")
            finally:
                self._client = None
                self._pool = None

    @property
    def backend_type(self) -> str:
        """返回缓存后端类型标识"""
        return "redis"

    @property
    def is_distributed(self) -> bool:
        """Redis 是分布式缓存"""
        return True

    async def health_check(self) -> bool:
        """健康检查"""
        try:
            client = await self._ensure_connection()
            await client.ping()
            return True
        except Exception:
            return False

    async def ttl(self, key: str) -> int:
        """获取键的剩余 TTL

        Args:
            key: 缓存键

        Returns:
            剩余秒数，-1 表示无过期时间，-2 表示键不存在
        """
        try:
            client = await self._ensure_connection()
            full_key = self._make_key(key)
            return await client.ttl(full_key)
        except Exception as e:
            logger.error(f"Redis TTL 失败 [{key}]: {e}")
            return -2

    async def expire(self, key: str, ttl: int) -> bool:
        """更新键的 TTL

        Args:
            key: 缓存键
            ttl: 新的过期时间（秒）

        Returns:
            是否成功
        """
        try:
            client = await self._ensure_connection()
            full_key = self._make_key(key)
            return bool(await client.expire(full_key, ttl))
        except Exception as e:
            logger.error(f"Redis EXPIRE 失败 [{key}]: {e}")
            return False

    async def incr(self, key: str, amount: int = 1) -> int:
        """原子递增

        Args:
            key: 缓存键
            amount: 递增量

        Returns:
            递增后的值
        """
        try:
            client = await self._ensure_connection()
            full_key = self._make_key(key)
            return await client.incrby(full_key, amount)
        except Exception as e:
            logger.error(f"Redis INCR 失败 [{key}]: {e}")
            return 0

    async def decr(self, key: str, amount: int = 1) -> int:
        """原子递减

        Args:
            key: 缓存键
            amount: 递减量

        Returns:
            递减后的值
        """
        try:
            client = await self._ensure_connection()
            full_key = self._make_key(key)
            return await client.decrby(full_key, amount)
        except Exception as e:
            logger.error(f"Redis DECR 失败 [{key}]: {e}")
            return 0


# 全局 Redis 缓存实例
_global_redis_cache: RedisCache | None = None
_redis_cache_lock = asyncio.Lock()


async def get_redis_cache() -> RedisCache:
    """获取全局 Redis 缓存实例（单例）"""
    global _global_redis_cache

    if _global_redis_cache is None:
        async with _redis_cache_lock:
            if _global_redis_cache is None:
                # 从配置加载参数
                try:
                    from src.config.config import global_config

                    assert global_config is not None
                    db_config = global_config.database

                    _global_redis_cache = RedisCache(
                        host=db_config.redis_host,
                        port=db_config.redis_port,
                        password=db_config.redis_password or None,
                        db=db_config.redis_db,
                        key_prefix=db_config.redis_key_prefix,
                        default_ttl=db_config.redis_default_ttl,
                        pool_size=db_config.redis_connection_pool_size,
                        socket_timeout=db_config.redis_socket_timeout,
                        ssl=db_config.redis_ssl,
                    )
                except Exception as e:
                    logger.warning(f"无法从配置加载 Redis 参数，使用默认值: {e}")
                    _global_redis_cache = RedisCache()

    return _global_redis_cache


async def close_redis_cache() -> None:
    """关闭全局 Redis 缓存"""
    global _global_redis_cache

    if _global_redis_cache is not None:
        await _global_redis_cache.close()
        _global_redis_cache = None
        logger.info("全局 Redis 缓存已关闭")

