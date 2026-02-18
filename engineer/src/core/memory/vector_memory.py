"""
向量存储记忆实现（长期记忆）
使用向量相似度检索相关历史信息
"""

# genAI_main_start
from typing import List, Dict, Any, Optional, Callable, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime
import json
import math
import copy

from .base_memory import (
    BaseMemory, BaseChatMessageHistory,
    MemoryVariables, MemoryType
)
from ..messages import BaseMessage, HumanMessage, AIMessage, Message
from .chat_history import InMemoryChatMessageHistory

if TYPE_CHECKING:
    pass  # 用于类型提示的导入


@dataclass
class MemoryDocument:
    """记忆文档
    
    存储在向量数据库中的文档单元
    
    Attributes:
        content: 文档内容
        embedding: 向量嵌入
        metadata: 元数据
        id: 文档ID
        score: 相似度分数（检索时填充）
    """
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = ""
    score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "content": self.content,
            "embedding": self.embedding,
            "metadata": self.metadata,
            "id": self.id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryDocument":
        """从字典创建"""
        return cls(
            content=data["content"],
            embedding=data.get("embedding"),
            metadata=data.get("metadata", {}),
            id=data.get("id", "")
        )


class SimpleVectorStore:
    """简单向量存储
    
    基于内存的向量存储，使用余弦相似度进行检索
    适用于中小规模数据，生产环境建议使用专业向量数据库
    
    Attributes:
        documents: 文档列表
        embedding_function: 嵌入函数
    """
    
    def __init__(
        self,
        embedding_function: Optional[Callable[[str], List[float]]] = None
    ):
        """初始化向量存储
        
        Args:
            embedding_function: 文本嵌入函数（输入文本，返回向量）
        """
        self.documents: List[MemoryDocument] = []
        self.embedding_function = embedding_function or self._default_embedding
        self._next_id = 0
    
    @staticmethod
    def _default_embedding(text: str) -> List[float]:
        """默认嵌入函数（简单的词袋模型）
        
        注意：这是一个非常简单的实现，仅用于演示
        生产环境请使用真实的嵌入模型
        """
        # 简单的字符级hash嵌入
        vector = [0.0] * 128
        for i, char in enumerate(text):
            idx = hash(char) % 128
            vector[idx] += 1.0 / (i + 1)
        
        # 归一化
        norm = math.sqrt(sum(x * x for x in vector))
        if norm > 0:
            vector = [x / norm for x in vector]
        
        return vector
    
    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def add_document(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """添加文档
        
        Args:
            content: 文档内容
            metadata: 元数据
        
        Returns:
            文档ID
        """
        doc_id = f"doc_{self._next_id}"
        self._next_id += 1
        
        embedding = self.embedding_function(content)
        doc = MemoryDocument(
            content=content,
            embedding=embedding,
            metadata=metadata or {},
            id=doc_id
        )
        self.documents.append(doc)
        
        return doc_id
    
    def add_documents(self, contents: List[str], metadatas: Optional[List[Dict]] = None) -> List[str]:
        """批量添加文档
        
        Args:
            contents: 文档内容列表
            metadatas: 元数据列表
        
        Returns:
            文档ID列表
        """
        metadatas = metadatas or [{}] * len(contents)
        return [self.add_document(c, m) for c, m in zip(contents, metadatas)]
    
    def search(self, query: str, k: int = 4) -> List[MemoryDocument]:
        """相似度搜索
        
        Args:
            query: 查询文本
            k: 返回结果数量
        
        Returns:
            相似文档列表（按相似度降序）
        """
        if not self.documents:
            return []
        
        query_embedding = self.embedding_function(query)
        
        # 计算所有文档的相似度
        scored_docs = []
        for doc in self.documents:
            if doc.embedding:
                score = self._cosine_similarity(query_embedding, doc.embedding)
                scored_doc = copy.deepcopy(doc)
                scored_doc.score = score
                scored_docs.append(scored_doc)
        
        # 按相似度排序
        scored_docs.sort(key=lambda x: x.score, reverse=True)
        
        return scored_docs[:k]
    
    def delete(self, doc_id: str) -> bool:
        """删除文档
        
        Args:
            doc_id: 文档ID
        
        Returns:
            是否删除成功
        """
        for i, doc in enumerate(self.documents):
            if doc.id == doc_id:
                self.documents.pop(i)
                return True
        return False
    
    def clear(self) -> None:
        """清空所有文档"""
        self.documents = []
        self._next_id = 0
    
    def __len__(self) -> int:
        """返回文档数量"""
        return len(self.documents)


class VectorStoreMemory(BaseMemory):
    """向量存储记忆
    
    基于向量相似度检索相关历史记忆
    适用于需要长期记忆和语义检索的场景
    
    Attributes:
        vector_store: 向量存储实例
        retrieval_k: 检索返回的文档数量
        score_threshold: 相似度阈值
    """
    
    def __init__(
        self,
        vector_store: Optional[SimpleVectorStore] = None,
        embedding_function: Optional[Callable[[str], List[float]]] = None,
        retrieval_k: int = 4,
        score_threshold: float = 0.0,
        memory_key: str = "context",
        input_key: str = "input",
        output_key: str = "output",
        combine_messages: bool = True,
        return_messages: bool = False,
        human_prefix: str = "Human",
        ai_prefix: str = "AI",
        verbose: bool = False
    ):
        """初始化向量存储记忆
        
        Args:
            vector_store: 向量存储实例（可选）
            embedding_function: 嵌入函数（可选）
            retrieval_k: 检索返回的文档数量
            score_threshold: 相似度阈值（低于此值的结果将被过滤）
            memory_key: 记忆键名
            input_key: 输入键名
            output_key: 输出键名
            combine_messages: 是否合并输入输出为一条记录
            return_messages: 是否返回消息列表
            human_prefix: 人类消息前缀
            ai_prefix: AI消息前缀
            verbose: 是否输出详细日志
        """
        super().__init__(
            memory_type=MemoryType.LONG_TERM,
            memory_key=memory_key,
            return_messages=return_messages,
            human_prefix=human_prefix,
            ai_prefix=ai_prefix,
            verbose=verbose
        )
        self.vector_store = vector_store or SimpleVectorStore(embedding_function)
        self.retrieval_k = retrieval_k
        self.score_threshold = score_threshold
        self.input_key = input_key
        self.output_key = output_key
        self.combine_messages = combine_messages
    
    @property
    def memory_variables(self) -> List[str]:
        """返回记忆变量名列表"""
        return [self.memory_key]
    
    def load_memory_variables(self, inputs: Optional[Dict[str, Any]] = None) -> MemoryVariables:
        """加载记忆变量
        
        根据输入检索相关记忆
        """
        if not inputs:
            return MemoryVariables()
        
        # 获取查询文本
        query = inputs.get(self.input_key, "")
        if not query:
            query = str(inputs)
        
        # 检索相关文档
        docs = self.vector_store.search(query, k=self.retrieval_k)
        
        # 过滤低分文档
        docs = [d for d in docs if d.score >= self.score_threshold]
        
        if self.verbose:
            print(f"[Memory] 检索到 {len(docs)} 条相关记忆")
            for doc in docs:
                print(f"  - 相似度 {doc.score:.3f}: {doc.content[:50]}...")
        
        # 格式化上下文
        context_parts = []
        for doc in docs:
            context_parts.append(doc.content)
        
        context = "\n---\n".join(context_parts)
        
        return MemoryVariables(
            context=context,
            extra={"retrieved_docs": [d.to_dict() for d in docs]}
        )
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """保存对话上下文到向量存储"""
        input_str = inputs.get(self.input_key, "")
        output_str = outputs.get(self.output_key, "")
        
        if self.combine_messages:
            # 合并为一条记录
            if input_str and output_str:
                content = f"{self.human_prefix}: {input_str}\n{self.ai_prefix}: {output_str}"
                metadata = {
                    "input": input_str,
                    "output": output_str,
                    "timestamp": datetime.now().isoformat()
                }
                self.vector_store.add_document(content, metadata)
                
                if self.verbose:
                    print(f"[Memory] 保存对话记录: {content[:50]}...")
        else:
            # 分别存储
            if input_str:
                self.vector_store.add_document(
                    f"{self.human_prefix}: {input_str}",
                    {"role": "user", "timestamp": datetime.now().isoformat()}
                )
            if output_str:
                self.vector_store.add_document(
                    f"{self.ai_prefix}: {output_str}",
                    {"role": "assistant", "timestamp": datetime.now().isoformat()}
                )
    
    def add_memory(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """直接添加记忆
        
        Args:
            content: 记忆内容
            metadata: 元数据
        
        Returns:
            文档ID
        """
        return self.vector_store.add_document(content, metadata)
    
    def search_memory(self, query: str, k: Optional[int] = None) -> List[MemoryDocument]:
        """搜索记忆
        
        Args:
            query: 查询文本
            k: 返回数量（可选，默认使用retrieval_k）
        
        Returns:
            相关记忆列表
        """
        k = k or self.retrieval_k
        return self.vector_store.search(query, k)
    
    def clear(self) -> None:
        """清空记忆"""
        self.vector_store.clear()
    
    def __repr__(self) -> str:
        return f"VectorStoreMemory(documents={len(self.vector_store)}, retrieval_k={self.retrieval_k})"
# genAI_main_end
