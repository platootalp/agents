"""
企业级RAG服务实现
核心特性：
1. 多级缓存策略（内存LRU → Redis分布式 → 向量数据库）
2. 混合检索优化（向量相似度 + 关键词BM25）
3. 重排序模型（Cross-Encoder提升相关性）
4. 可追溯引用（返回文档片段、出处、置信度）
"""

import asyncio
import hashlib
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from functools import lru_cache
from dataclasses import dataclass
import numpy as np

# 第三方库导入（生产环境应捕获导入错误）
try:
    import redis.asyncio as redis
    from pymilvus import MilvusClient, DataType
    from sentence_transformers import CrossEncoder
    from rank_bm25 import BM25Okapi
    from openai import AsyncOpenAI
except ImportError as e:
    print(f"依赖库导入失败: {e}")
    print("请安装 requirements.txt 中的依赖")


@dataclass
class SearchResult:
    """检索结果数据结构"""
    content: str
    source_document: str
    page_number: int
    confidence: float
    embedding_similarity: float
    keyword_score: float
    combined_score: float


class MultiLevelCacheRAG:
    """多级缓存RAG服务"""
    
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        milvus_host: str = "localhost",
        milvus_port: int = 19530,
        embedding_model: str = "text-embedding-ada-002",
        rerank_model: str = "BAAI/bge-reranker-base"
    ):
        # 内存缓存（LRU，最大1000条）
        self.memory_cache = {}
        self.cache_size = 1000
        
        # Redis客户端
        self.redis_client = None
        self.redis_host = redis_host
        self.redis_port = redis_port
        
        # Milvus向量数据库
        self.milvus_client = None
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.collection_name = "knowledge_base"
        
        # 嵌入模型（实际生产应使用API或本地模型）
        self.embedding_model = embedding_model
        self.openai_client = AsyncOpenAI() if embedding_model.startswith("text-embedding") else None
        
        # 重排序模型
        self.rerank_model = CrossEncoder(rerank_model)
        
        # BM25关键词检索
        self.bm25_index = None
        self.documents = []
        
        # 统计指标
        self.metrics = {
            "memory_cache_hits": 0,
            "redis_cache_hits": 0,
            "vector_searches": 0,
            "average_latency": 0.0
        }
    
    async def initialize(self):
        """初始化所有组件"""
        # 初始化Redis
        self.redis_client = redis.Redis(
            host=self.redis_host,
            port=self.redis_port,
            decode_responses=True,
            max_connections=20
        )
        
        # 初始化Milvus
        self.milvus_client = MilvusClient(
            uri=f"http://{self.milvus_host}:{self.milvus_port}"
        )
        
        # 创建集合（如果不存在）
        if not self.milvus_client.has_collection(self.collection_name):
            schema = [
                {"name": "id", "type": DataType.INT64, "is_primary": True},
                {"name": "content", "type": DataType.VARCHAR, "max_length": 65535},
                {"name": "embedding", "type": DataType.FLOAT_VECTOR, "dim": 1536},
                {"name": "metadata", "type": DataType.JSON}
            ]
            self.milvus_client.create_collection(
                collection_name=self.collection_name,
                dimension=1536,
                schema=schema
            )
        
        # 加载测试文档（生产环境从数据库加载）
        self._load_sample_documents()
    
    def _load_sample_documents(self):
        """加载示例文档用于演示"""
        self.documents = [
            "AI Agent是能够感知环境、做出决策并执行动作的智能系统。",
            "RAG（检索增强生成）通过检索外部知识库来增强LLM生成质量。",
            "向量数据库专门用于存储和检索高维向量，支持相似度搜索。",
            "多级缓存策略包括内存缓存、分布式缓存和持久化存储。",
            "API网关负责路由转发、限流熔断、认证鉴权和日志记录。"
        ]
        
        # 构建BM25索引
        tokenized_docs = [doc.split() for doc in self.documents]
        self.bm25_index = BM25Okapi(tokenized_docs)
    
    def _generate_cache_key(self, question: str) -> str:
        """生成缓存键"""
        return f"rag:{hashlib.md5(question.encode()).hexdigest()}"
    
    async def _get_from_memory_cache(self, key: str) -> Optional[Dict]:
        """从内存缓存获取"""
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            # 检查过期时间（如果设置了）
            if "expiry" in entry and entry["expiry"] < time.time():
                del self.memory_cache[key]
                return None
            self.metrics["memory_cache_hits"] += 1
            return entry["data"]
        return None
    
    async def _get_from_redis_cache(self, key: str) -> Optional[Dict]:
        """从Redis缓存获取"""
        try:
            data = await self.redis_client.get(key)
            if data:
                self.metrics["redis_cache_hits"] += 1
                return json.loads(data)
        except Exception as e:
            print(f"Redis缓存读取失败: {e}")
        return None
    
    async def _cache_in_memory(self, key: str, data: Dict, ttl: int = 300):
        """缓存到内存"""
        # LRU淘汰策略
        if len(self.memory_cache) >= self.cache_size:
            # 简单淘汰：删除最早插入的键（生产环境应用更复杂策略）
            oldest_key = next(iter(self.memory_cache))
            del self.memory_cache[oldest_key]
        
        self.memory_cache[key] = {
            "data": data,
            "expiry": time.time() + ttl,
            "timestamp": time.time()
        }
    
    async def _cache_in_redis(self, key: str, data: Dict, ttl: int = 3600):
        """缓存到Redis"""
        try:
            await self.redis_client.setex(key, ttl, json.dumps(data))
        except Exception as e:
            print(f"Redis缓存写入失败: {e}")
    
    async def _embed_text(self, text: str) -> List[float]:
        """生成文本嵌入向量"""
        # 实际生产环境应使用OpenAI API或本地模型
        # 这里返回模拟向量用于演示
        np.random.seed(hash(text) % 2**32)
        return list(np.random.randn(1536))
    
    async def _hybrid_search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """混合检索：向量相似度 + 关键词BM25"""
        
        # 1. 向量相似度搜索
        query_embedding = await self._embed_text(query)
        vector_results = self.milvus_client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            limit=top_k,
            output_fields=["content", "metadata"]
        )
        
        # 2. 关键词BM25检索
        tokenized_query = query.split()
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        
        # 3. 结果融合（加权平均）
        combined_results = []
        
        for i, result in enumerate(vector_results[0]):
            # 获取向量相似度分数
            vector_score = result["score"]
            
            # 获取BM25分数（归一化）
            bm25_score = bm25_scores[i] if i < len(bm25_scores) else 0
            normalized_bm25 = bm25_score / (bm25_scores.max() if len(bm25_scores) > 0 else 1)
            
            # 加权融合（向量70%，关键词30%）
            combined_score = 0.7 * vector_score + 0.3 * normalized_bm25
            
            search_result = SearchResult(
                content=result["entity"]["content"],
                source_document=result["entity"]["metadata"].get("source", "unknown"),
                page_number=result["entity"]["metadata"].get("page", 0),
                confidence=combined_score,
                embedding_similarity=vector_score,
                keyword_score=normalized_bm25,
                combined_score=combined_score
            )
            combined_results.append(search_result)
        
        # 按综合分数排序
        combined_results.sort(key=lambda x: x.combined_score, reverse=True)
        return combined_results[:top_k]
    
    async def _rerank_results(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """使用重排序模型提升相关性"""
        if not results:
            return []
        
        # 准备重排序输入对
        pairs = [[query, r.content] for r in results]
        
        # 获取重排序分数
        rerank_scores = self.rerank_model.predict(pairs)
        
        # 更新结果分数
        for i, result in enumerate(results):
            result.confidence = float(rerank_scores[i])
            result.combined_score = result.confidence
        
        # 按重排序分数重新排序
        results.sort(key=lambda x: x.confidence, reverse=True)
        return results
    
    async def query(self, question: str, use_cache: bool = True) -> Dict[str, Any]:
        """执行RAG查询"""
        start_time = time.time()
        
        # 生成缓存键
        cache_key = self._generate_cache_key(question)
        
        # 1. 检查内存缓存
        if use_cache:
            memory_result = await self._get_from_memory_cache(cache_key)
            if memory_result:
                latency = time.time() - start_time
                self._update_metrics(latency, source="memory")
                return {
                    "answer": memory_result["answer"],
                    "sources": memory_result["sources"],
                    "cache_source": "memory",
                    "latency": latency,
                    "metrics": self.metrics.copy()
                }
        
        # 2. 检查Redis缓存
        if use_cache:
            redis_result = await self._get_from_redis_cache(cache_key)
            if redis_result:
                # 回写内存缓存
                await self._cache_in_memory(cache_key, redis_result)
                latency = time.time() - start_time
                self._update_metrics(latency, source="redis")
                return {
                    "answer": redis_result["answer"],
                    "sources": redis_result["sources"],
                    "cache_source": "redis",
                    "latency": latency,
                    "metrics": self.metrics.copy()
                }
        
        # 3. 执行向量检索（实际生产环境）
        self.metrics["vector_searches"] += 1
        
        # 混合检索
        search_results = await self._hybrid_search(question, top_k=10)
        
        # 重排序
        reranked_results = await self._rerank_results(question, search_results)
        
        # 生成答案（模拟）
        answer = f"基于检索到的信息，问题的答案是：{question} 的相关内容包含在以下文档中。"
        
        # 准备结果
        result_data = {
            "answer": answer,
            "sources": [
                {
                    "content": r.content,
                    "source": r.source_document,
                    "page": r.page_number,
                    "confidence": round(r.confidence, 4)
                }
                for r in reranked_results
            ]
        }
        
        # 4. 缓存结果
        if use_cache:
            await self._cache_in_redis(cache_key, result_data, ttl=3600)
            await self._cache_in_memory(cache_key, result_data, ttl=300)
        
        latency = time.time() - start_time
        self._update_metrics(latency, source="vector")
        
        return {
            "answer": answer,
            "sources": result_data["sources"],
            "cache_source": "vector_db",
            "latency": latency,
            "metrics": self.metrics.copy()
        }
    
    def _update_metrics(self, latency: float, source: str):
        """更新性能指标"""
        # 更新平均延迟（指数加权移动平均）
        alpha = 0.1
        self.metrics["average_latency"] = (
            alpha * latency + (1 - alpha) * self.metrics["average_latency"]
        )
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total_queries = sum([
            self.metrics["memory_cache_hits"],
            self.metrics["redis_cache_hits"],
            self.metrics["vector_searches"]
        ])
        
        if total_queries == 0:
            hit_rates = {"memory": 0.0, "redis": 0.0, "overall": 0.0}
        else:
            hit_rates = {
                "memory": self.metrics["memory_cache_hits"] / total_queries,
                "redis": self.metrics["redis_cache_hits"] / total_queries,
                "overall": (self.metrics["memory_cache_hits"] + self.metrics["redis_cache_hits"]) / total_queries
            }
        
        return {
            "memory_cache_size": len(self.memory_cache),
            "memory_cache_hits": self.metrics["memory_cache_hits"],
            "redis_cache_hits": self.metrics["redis_cache_hits"],
            "vector_searches": self.metrics["vector_searches"],
            "average_latency": round(self.metrics["average_latency"], 4),
            "hit_rates": {k: round(v, 4) for k, v in hit_rates.items()}
        }


# 使用示例
async def demo():
    """演示RAG服务功能"""
    print("初始化RAG服务...")
    rag = MultiLevelCacheRAG()
    await rag.initialize()
    
    # 执行查询
    questions = [
        "什么是AI Agent？",
        "RAG如何工作？",
        "向量数据库有什么用途？"
    ]
    
    for q in questions:
        print(f"\n查询: {q}")
        result = await rag.query(q)
        
        print(f"答案: {result['answer']}")
        print(f"缓存来源: {result['cache_source']}")
        print(f"延迟: {result['latency']:.3f}秒")
        
        if result["sources"]:
            print("来源:")
            for i, source in enumerate(result["sources"][:2], 1):
                print(f"  {i}. {source['content'][:80]}... (置信度: {source['confidence']})")
    
    # 获取统计信息
    stats = rag.get_cache_stats()
    print(f"\n缓存统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(demo())