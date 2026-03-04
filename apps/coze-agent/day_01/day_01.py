import asyncio
import threading
import time
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, AsyncGenerator, List

"""
    装饰器、异步编程、生成器
"""


## 企业级缓存装饰器
class DistributedCacheDecorator:
    """
    分布式缓存装饰器，支持TTL（生存时间）、线程安全与一致性控制
    
    设计意图：
    - 为函数调用结果提供缓存机制，减少重复计算和网络请求
    - 支持分布式缓存后端（如Redis）的同步
    - 确保线程安全，适用于并发环境
    - 自动处理缓存过期
    """

    def __init__(self, ttl: int = 300, cache_backend: str = 'redis'):
        """
        初始化缓存装饰器
        
        Args:
            ttl: 缓存存活时间（秒），默认300秒（5分钟）
            cache_backend: 缓存后端类型，默认'redis'
        """
        self.ttl = ttl  # 缓存存活时间（秒）
        self.cache_backend = cache_backend  # 缓存后端类型
        self._cache: Dict[str, Any] = {}  # 本地缓存存储
        self._lock = threading.RLock()  # 可重入锁，确保线程安全

    def __call__(self, func: Callable) -> Callable:
        """
        实现装饰器逻辑，包装目标函数
        
        Args:
            func: 被装饰的函数
            
        Returns:
            包装后的函数，具有缓存功能
        """

        @wraps(func)  # 保留原函数的元数据
        def wrapper(*args, **kwargs) -> Any:
            # 生成缓存键：函数名+参数哈希，确保唯一性
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"

            with self._lock:  # 线程安全操作
                # 检查缓存命中
                if cache_key in self._cache:
                    cached_item = self._cache[cache_key]
                    # 检查缓存是否过期
                    if time.time() - cached_item['timestamp'] < self.ttl:
                        print(f"缓存命中: {cache_key}")
                        return cached_item['value']

                # 缓存未命中或已过期，执行原函数
                result = func(*args, **kwargs)

                # 写入缓存，包含值和时间戳
                self._cache[cache_key] = {
                    'value': result,
                    'timestamp': time.time()
                }

                # 分布式同步逻辑（简化为本地缓存）
                if self.cache_backend == 'redis':
                    self._sync_to_redis(cache_key, result)

                return result

        return wrapper

    def _sync_to_redis(self, key: str, value: Any):
        """
        模拟同步到Redis分布式缓存
        
        Args:
            key: 缓存键
            value: 缓存值
        """
        print(f"同步到分布式缓存: {key}")


# 使用示例：AI Agent的智能重试机制
@DistributedCacheDecorator(ttl=60, cache_backend='redis')
def fetch_llm_response(prompt: str, model: str = "gpt-4") -> dict:
    """
    模拟大模型API调用，自动缓存+重试
    
    Args:
        prompt: 提示词
        model: 模型名称，默认"gpt-4"
        
    Returns:
        包含AI回复和token数的字典
    """
    print(f"调用模型 {model}: {prompt[:50]}...")
    time.sleep(0.5)  # 模拟网络延迟
    return {"response": f"针对'{prompt}'的AI回复", "tokens": 150}


## 异步流式RAG生成器
class AsyncRAGGenerator:
    """
    异步RAG（检索增强生成）流式生成器
    支持并发检索与实时返回结果
    
    设计意图：
    - 并行处理多个文档片段，提高检索效率
    - 流式返回结果，提升用户体验
    - 异步非阻塞设计，适合高并发场景
    """

    def __init__(self, chunk_size: int = 1000):
        """
        初始化RAG生成器
        
        Args:
            chunk_size: 文档分块大小，默认1000字符
        """
        self.chunk_size = chunk_size

    async def stream_retrieval(self, query: str, documents: List[str]) -> AsyncGenerator[str, None]:
        """
        流式检索：边检索边返回结果
        
        Args:
            query: 查询语句
            documents: 文档列表
            
        Yields:
            相关文档片段及其相关性得分
        """
        # 并行处理文档分块
        chunks = self._chunk_documents(documents)
        # 创建任务列表
        tasks = [asyncio.create_task(self._score_chunk(query, chunk)) for chunk in chunks]

        # 等待所有任务完成
        done, _ = await asyncio.wait(tasks)

        # 处理完成的任务
        for task in done:
            chunk, score = await task
            # 确保返回结果，降低阈值到0.1
            if score > 0.1:
                yield f"相关片段 (得分{score:.2f}): {chunk[:200]}..."

    def _chunk_documents(self, documents: List[str]) -> List[str]:
        """
        文档分块算法（简化版）
        
        Args:
            documents: 文档列表
            
        Returns:
            分块后的文档片段列表
        """
        # 简化分块，直接返回整个文档
        return documents

    async def _score_chunk(self, query: str, chunk: str) -> tuple:
        """
        相关性评分（模拟）
        
        Args:
            query: 查询语句
            chunk: 文档片段
            
        Returns:
            (文档片段, 相关性得分) 元组
        """
        await asyncio.sleep(0.01)  # 模拟计算延迟

        # 简单的相关性计算：检查查询词是否在文档中
        query_words = query.split()
        matching_words = sum(1 for word in query_words if word in chunk)

        # 计算相关性得分，确保有匹配时得分较高
        if matching_words > 0:
            relevance = 0.5 + (matching_words / len(query_words)) * 0.5
        else:
            relevance = 0.0

        return chunk, min(1.0, relevance)  # 确保得分不超过1.0


@dataclass
class ModelEndpoint:
    """
    模型端点配置

    Attributes:
        name: 模型名称
        url: 模型API地址
        max_concurrent: 最大并发请求数
    """
    name: str
    url: str
    max_concurrent: int


## 并发模型路由
class ConcurrentRouter:
    """
    并发模型路由，支持负载均衡与熔断
    
    设计意图：
    - 管理多个模型端点的并发请求
    - 实现模型调用的自动降级策略
    - 控制每个模型的并发访问量
    """

    def __init__(self, endpoints: List[ModelEndpoint]):
        """
        初始化并发路由器
        
        Args:
            endpoints: 模型端点列表
        """
        self.endpoints = endpoints
        # 为每个模型创建信号量，控制并发访问
        self.semaphores = {
            ep.name: asyncio.Semaphore(ep.max_concurrent)
            for ep in endpoints
        }

    async def route_request(self, prompt: str, preferred_model: str = None) -> Dict:
        """
        并发路由：优先首选模型，失败时自动降级
        
        Args:
            prompt: 提示词
            preferred_model: 首选模型名称
            
        Returns:
            包含模型名称和回复的字典
            
        Raises:
            RuntimeError: 所有模型均不可用
        """
        # 获取模型优先级列表
        candidates = self._prioritize_models(preferred_model)

        for model_name in candidates:
            # 使用信号量控制并发
            async with self.semaphores[model_name]:
                try:
                    # 调用模型
                    response = await self._call_model(model_name, prompt)
                    return {"model": model_name, "response": response}
                except Exception as e:
                    print(f"模型 {model_name} 调用失败: {e}")
                    continue  # 失败时尝试下一个模型

        raise RuntimeError("所有模型均不可用")

    async def _call_model(self, model_name: str, prompt: str) -> str:
        """
        模拟异步模型调用
        
        Args:
            model_name: 模型名称
            prompt: 提示词
            
        Returns:
            模型回复
        """
        await asyncio.sleep(0.1)  # 模拟网络延迟
        return f"来自{model_name}的回复: {prompt[:100]}"

    def _prioritize_models(self, preferred: str) -> List[str]:
        """
        模型优先级排序
        
        Args:
            preferred: 首选模型名称
            
        Returns:
            排序后的模型名称列表
        """
        if preferred and preferred in self.semaphores:
            # 首选模型放在第一位，其他模型按任意顺序
            return [preferred] + [m for m in self.semaphores if m != preferred]
        # 无首选模型时返回所有模型
        return list(self.semaphores.keys())


if __name__ == '__main__':
    pass
