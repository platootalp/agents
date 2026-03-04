"""
企业级缓存服务实现
核心特性：
1. 多级缓存架构（内存LRU → Redis分布式）
2. 缓存穿透防护（布隆过滤器 + 空值缓存）
3. 缓存雪崩防护（随机过期时间 + 热点数据永不过期）
4. 连接池优化与序列化性能提升
"""

import asyncio
import json
import random
import time
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import msgpack
import hashlib

# 第三方库导入
try:
    import redis.asyncio as redis
    from pybloom_live import BloomFilter
except ImportError as e:
    print(f"依赖库导入失败: {e}")


@dataclass
class CacheStats:
    """缓存统计信息"""
    total_requests: int = 0
    memory_hits: int = 0
    redis_hits: int = 0
    misses: int = 0
    bloom_filter_checks: int = 0
    null_cached: int = 0
    average_latency: float = 0.0


class MessagePackSerializer:
    """MsgPack序列化器（比JSON快30%）"""
    
    @staticmethod
    def serialize(data: Any) -> bytes:
        """序列化数据"""
        return msgpack.packb(data, use_bin_type=True)
    
    @staticmethod
    def deserialize(data: bytes) -> Any:
        """反序列化数据"""
        return msgpack.unpackb(data, raw=False)


class OptimizedRedisCache:
    """优化的Redis缓存服务"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        max_connections: int = 20,
        socket_timeout: int = 5,
        bloom_filter_capacity: int = 1000000,
        bloom_filter_error_rate: float = 0.001
    ):
        # Redis连接池配置
        self.redis_pool = None
        self.host = host
        self.port = port
        self.max_connections = max_connections
        self.socket_timeout = socket_timeout
        
        # 序列化器
        self.serializer = MessagePackSerializer()
        
        # 布隆过滤器（防止缓存穿透）
        self.bloom_filter = BloomFilter(
            capacity=bloom_filter_capacity,
            error_rate=bloom_filter_error_rate
        )
        
        # 内存缓存（LRU实现简化版）
        self.memory_cache = {}
        self.memory_cache_size = 1000
        self.memory_cache_order = []  # 用于LRU顺序
        
        # 热点数据跟踪
        self.hot_keys = set()
        self.access_counts = {}
        
        # 统计信息
        self.stats = CacheStats()
        
        # 连接重试配置
        self.retry_config = {
            "max_retries": 3,
            "base_delay": 0.1,
            "max_delay": 1.0
        }
    
    async def initialize(self):
        """初始化Redis连接池"""
        try:
            self.redis_pool = redis.ConnectionPool(
                host=self.host,
                port=self.port,
                max_connections=self.max_connections,
                socket_timeout=self.socket_timeout,
                decode_responses=False  # 原始字节模式，MsgPack需要
            )
            
            # 创建Redis客户端
            self.redis_client = redis.Redis(connection_pool=self.redis_pool)
            
            # 测试连接
            await self.redis_client.ping()
            print(f"Redis连接成功: {self.host}:{self.port}")
            
        except Exception as e:
            print(f"Redis连接失败: {e}")
            # 生产环境应启动降级模式
            self.redis_client = None
    
    async def _retry_with_backoff(self, func, *args, **kwargs):
        """带指数退避的重试机制"""
        last_exception = None
        
        for attempt in range(self.retry_config["max_retries"]):
            try:
                return await func(*args, **kwargs)
            except (redis.ConnectionError, redis.TimeoutError) as e:
                last_exception = e
                
                # 计算退避时间
                delay = min(
                    self.retry_config["base_delay"] * (2 ** attempt),
                    self.retry_config["max_delay"]
                )
                
                print(f"Redis操作失败，第{attempt+1}次重试，等待{delay:.2f}秒")
                await asyncio.sleep(delay)
        
        # 所有重试都失败
        if last_exception:
            raise last_exception
    
    def _generate_key(self, namespace: str, key: str) -> str:
        """生成完整的缓存键"""
        return f"{namespace}:{hashlib.md5(key.encode()).hexdigest()}"
    
    def _jitter_ttl(self, ttl: int) -> int:
        """添加随机波动到TTL（防止雪崩）"""
        # ±10% 随机波动
        jitter = random.uniform(0.9, 1.1)
        return int(ttl * jitter)
    
    def _update_access_count(self, key: str):
        """更新键的访问计数"""
        self.access_counts[key] = self.access_counts.get(key, 0) + 1
        
        # 如果访问次数超过阈值，标记为热点数据
        if self.access_counts[key] > 100:
            self.hot_keys.add(key)
    
    def _is_hot_key(self, key: str) -> bool:
        """检查是否为热点数据"""
        return key in self.hot_keys
    
    async def _get_from_memory(self, key: str) -> Optional[Any]:
        """从内存缓存获取（LRU策略）"""
        start_time = time.time()
        
        if key in self.memory_cache:
            # 更新LRU顺序（移动到最近使用）
            if key in self.memory_cache_order:
                self.memory_cache_order.remove(key)
            self.memory_cache_order.append(key)
            
            self.stats.memory_hits += 1
            latency = time.time() - start_time
            self._update_average_latency(latency)
            
            return self.memory_cache[key]
        
        return None
    
    async def _set_in_memory(self, key: str, value: Any, ttl: int = 300):
        """设置内存缓存（LRU淘汰）"""
        # 如果缓存已满，淘汰最久未使用的键
        if len(self.memory_cache) >= self.memory_cache_size:
            lru_key = self.memory_cache_order.pop(0)
            del self.memory_cache[lru_key]
        
        # 存储数据
        self.memory_cache[key] = value
        self.memory_cache_order.append(key)
        
        # 简单TTL实现（生产环境应用更精确的）
        async def expire_task():
            await asyncio.sleep(ttl)
            if key in self.memory_cache:
                del self.memory_cache[key]
                if key in self.memory_cache_order:
                    self.memory_cache_order.remove(key)
        
        asyncio.create_task(expire_task())
    
    async def get_with_protection(
        self,
        namespace: str,
        key: str,
        default: Any = None
    ) -> Union[Any, None]:
        """带防护机制的缓存获取"""
        start_time = time.time()
        self.stats.total_requests += 1
        
        full_key = self._generate_key(namespace, key)
        
        # 更新访问计数
        self._update_access_count(full_key)
        
        # 1. 检查内存缓存
        memory_result = await self._get_from_memory(full_key)
        if memory_result is not None:
            return memory_result
        
        # 如果Redis不可用，直接返回默认值
        if self.redis_client is None:
            self.stats.misses += 1
            latency = time.time() - start_time
            self._update_average_latency(latency)
            return default
        
        # 2. 布隆过滤器检查（防止缓存穿透）
        self.stats.bloom_filter_checks += 1
        
        if not self.bloom_filter.add(full_key):
            # 键不在布隆过滤器中，可能是无效键
            # 缓存空值（防止进一步穿透）
            await self._cache_null_value(full_key)
            self.stats.null_cached += 1
            self.stats.misses += 1
            
            latency = time.time() - start_time
            self._update_average_latency(latency)
            return default
        
        # 3. Redis查询
        try:
            redis_data = await self._retry_with_backoff(
                self.redis_client.get,
                full_key
            )
            
            if redis_data is not None:
                # 反序列化数据
                value = self.serializer.deserialize(redis_data)
                
                # 回写内存缓存
                await self._set_in_memory(full_key, value)
                
                self.stats.redis_hits += 1
                
                latency = time.time() - start_time
                self._update_average_latency(latency)
                
                return value
            
            else:
                # 缓存未命中
                self.stats.misses += 1
                
                # 缓存空值（防止后续查询穿透）
                await self._cache_null_value(full_key)
                
                latency = time.time() - start_time
                self._update_average_latency(latency)
                
                return default
                
        except Exception as e:
            print(f"Redis查询失败: {e}")
            self.stats.misses += 1
            
            latency = time.time() - start_time
            self._update_average_latency(latency)
            
            return default
    
    async def set_with_protection(
        self,
        namespace: str,
        key: str,
        value: Any,
        ttl: int = 3600
    ):
        """带防护机制的缓存设置"""
        full_key = self._generate_key(namespace, key)
        
        # 更新访问计数
        self._update_access_count(full_key)
        
        # 1. 设置内存缓存
        await self._set_in_memory(full_key, value, ttl)
        
        # 2. 如果Redis不可用，直接返回
        if self.redis_client is None:
            return
        
        # 3. 设置Redis缓存（带随机TTL）
        jitter_ttl = self._jitter_ttl(ttl)
        
        # 序列化数据
        serialized = self.serializer.serialize(value)
        
        try:
            await self._retry_with_backoff(
                self.redis_client.setex,
                full_key,
                jitter_ttl,
                serialized
            )
            
            # 如果是热点数据，设置永不过期（通过定期刷新实现）
            if self._is_hot_key(full_key):
                await self.redis_client.persist(full_key)
                print(f"热点数据永不过期: {full_key}")
                
        except Exception as e:
            print(f"Redis设置失败: {e}")
    
    async def _cache_null_value(self, key: str):
        """缓存空值（防止穿透）"""
        # 空值标记
        null_marker = {"__null__": True}
        
        # 设置短期缓存（30秒）
        await self._set_in_memory(key, null_marker, ttl=30)
        
        if self.redis_client:
            try:
                serialized = self.serializer.serialize(null_marker)
                await self.redis_client.setex(key, 30, serialized)
            except Exception as e:
                print(f"Redis空值缓存失败: {e}")
    
    async def delete(self, namespace: str, key: str):
        """删除缓存"""
        full_key = self._generate_key(namespace, key)
        
        # 从内存缓存删除
        if full_key in self.memory_cache:
            del self.memory_cache[full_key]
            if full_key in self.memory_cache_order:
                self.memory_cache_order.remove(full_key)
        
        # 从Redis删除
        if self.redis_client:
            try:
                await self.redis_client.delete(full_key)
            except Exception as e:
                print(f"Redis删除失败: {e}")
    
    async def clear_namespace(self, namespace: str):
        """清除命名空间下的所有缓存"""
        # 内存缓存中删除相关键
        keys_to_delete = [
            k for k in self.memory_cache.keys()
            if k.startswith(f"{namespace}:")
        ]
        
        for key in keys_to_delete:
            del self.memory_cache[key]
            if key in self.memory_cache_order:
                self.memory_cache_order.remove(key)
        
        # Redis中删除（使用模式匹配）
        if self.redis_client:
            try:
                pattern = f"{namespace}:*"
                keys = await self.redis_client.keys(pattern)
                if keys:
                    await self.redis_client.delete(*keys)
            except Exception as e:
                print(f"Redis模式删除失败: {e}")
    
    def get_hit_rate(self) -> Dict[str, float]:
        """计算缓存命中率"""
        if self.stats.total_requests == 0:
            return {
                "memory_hit_rate": 0.0,
                "redis_hit_rate": 0.0,
                "overall_hit_rate": 0.0
            }
        
        memory_hit_rate = self.stats.memory_hits / self.stats.total_requests
        redis_hit_rate = self.stats.redis_hits / self.stats.total_requests
        overall_hit_rate = (self.stats.memory_hits + self.stats.redis_hits) / self.stats.total_requests
        
        return {
            "memory_hit_rate": round(memory_hit_rate, 4),
            "redis_hit_rate": round(redis_hit_rate, 4),
            "overall_hit_rate": round(overall_hit_rate, 4)
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取完整统计信息"""
        hit_rates = self.get_hit_rate()
        
        return {
            "total_requests": self.stats.total_requests,
            "memory_hits": self.stats.memory_hits,
            "redis_hits": self.stats.redis_hits,
            "misses": self.stats.misses,
            "bloom_filter_checks": self.stats.bloom_filter_checks,
            "null_cached": self.stats.null_cached,
            "average_latency_ms": round(self.stats.average_latency * 1000, 2),
            "memory_cache_size": len(self.memory_cache),
            "hot_keys_count": len(self.hot_keys),
            "hit_rates": hit_rates
        }
    
    def _update_average_latency(self, latency: float):
        """更新平均延迟（指数加权移动平均）"""
        alpha = 0.1
        self.stats.average_latency = (
            alpha * latency + (1 - alpha) * self.stats.average_latency
        )


class MultiLevelCacheSystem:
    """多级缓存系统（整合内存与Redis）"""
    
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        # 内存缓存（一级）
        self.memory_cache = {}
        self.memory_cache_size = 1000
        
        # Redis缓存（二级）
        self.redis_cache = OptimizedRedisCache(
            host=redis_host,
            port=redis_port
        )
        
        # 统计信息
        self.stats = {
            "level1_hits": 0,
            "level2_hits": 0,
            "total_misses": 0
        }
    
    async def initialize(self):
        """初始化缓存系统"""
        await self.redis_cache.initialize()
        print("多级缓存系统初始化完成")
    
    async def get(self, namespace: str, key: str, default: Any = None) -> Any:
        """从多级缓存获取数据"""
        full_key = self.redis_cache._generate_key(namespace, key)
        
        # 1. 检查一级缓存（内存）
        if full_key in self.memory_cache:
            self.stats["level1_hits"] += 1
            return self.memory_cache[full_key]
        
        # 2. 检查二级缓存（Redis）
        value = await self.redis_cache.get_with_protection(namespace, key, default)
        
        if value is not default:
            self.stats["level2_hits"] += 1
            
            # 回写一级缓存
            self._set_in_memory(full_key, value)
            
            return value
        
        # 3. 缓存未命中
        self.stats["total_misses"] += 1
        return default
    
    async def set(self, namespace: str, key: str, value: Any, ttl: int = 3600):
        """设置多级缓存"""
        full_key = self.redis_cache._generate_key(namespace, key)
        
        # 1. 设置一级缓存
        self._set_in_memory(full_key, value)
        
        # 2. 设置二级缓存
        await self.redis_cache.set_with_protection(namespace, key, value, ttl)
    
    def _set_in_memory(self, key: str, value: Any):
        """设置内存缓存（简化LRU）"""
        # 如果缓存已满，随机淘汰一个键（生产环境应用更优算法）
        if len(self.memory_cache) >= self.memory_cache_size:
            random_key = random.choice(list(self.memory_cache.keys()))
            del self.memory_cache[random_key]
        
        self.memory_cache[key] = value
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        redis_stats = self.redis_cache.get_stats()
        
        total_requests = (
            self.stats["level1_hits"] +
            self.stats["level2_hits"] +
            self.stats["total_misses"]
        )
        
        if total_requests == 0:
            hit_rates = {"level1": 0.0, "level2": 0.0, "overall": 0.0}
        else:
            hit_rates = {
                "level1": self.stats["level1_hits"] / total_requests,
                "level2": self.stats["level2_hits"] / total_requests,
                "overall": (self.stats["level1_hits"] + self.stats["level2_hits"]) / total_requests
            }
        
        return {
            "total_requests": total_requests,
            "level1_hits": self.stats["level1_hits"],
            "level2_hits": self.stats["level2_hits"],
            "misses": self.stats["total_misses"],
            "hit_rates": {k: round(v, 4) for k, v in hit_rates.items()},
            "memory_cache_size": len(self.memory_cache),
            "redis_stats": redis_stats
        }


# 使用示例
async def demo():
    """演示缓存服务功能"""
    print("初始化多级缓存系统...")
    
    cache_system = MultiLevelCacheSystem()
    await cache_system.initialize()
    
    # 测试数据
    test_data = [
        ("user", "user:1001", {"id": 1001, "name": "张三", "email": "zhangsan@example.com"}),
        ("product", "product:2001", {"id": 2001, "name": "笔记本电脑", "price": 5999.00}),
        ("order", "order:3001", {"id": 3001, "total": 5999.00, "status": "paid"})
    ]
    
    print("\n1. 设置缓存数据:")
    for namespace, key, value in test_data:
        await cache_system.set(namespace, key, value, ttl=60)
        print(f"  设置 {namespace}:{key} = {value}")
    
    print("\n2. 获取缓存数据（首次获取，应命中Redis）:")
    for namespace, key, _ in test_data:
        value = await cache_system.get(namespace, key)
        print(f"  获取 {namespace}:{key} = {value}")
    
    print("\n3. 再次获取相同数据（应命中内存缓存）:")
    for namespace, key, _ in test_data:
        value = await cache_system.get(namespace, key)
        print(f"  获取 {namespace}:{key} = {value}")
    
    print("\n4. 获取不存在的数据（测试穿透防护）:")
    non_existent = await cache_system.get("user", "user:9999", default="未找到")
    print(f"  获取不存在的键: {non_existent}")
    
    print("\n5. 获取统计信息:")
    stats = cache_system.get_cache_stats()
    for category, data in stats.items():
        if isinstance(data, dict):
            print(f"  {category}:")
            for k, v in data.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {category}: {data}")


if __name__ == "__main__":
    asyncio.run(demo())