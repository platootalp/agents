"""
内存管理模块

提供 Agent 的内存系统，包括：
- 短期内存（BufferMemory）
- 长期内存（持久化存储）
- 向量内存（语义检索）
- 实体内存（关键信息提取）
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
import json
import hashlib


@dataclass
class MemoryEntry:
    """
    内存条目

    存储单条记忆的内容和元数据
    """

    content: str
    entry_type: str = "generic"  # generic, observation, thought, action, entity
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    entry_id: str = field(
        default_factory=lambda: hashlib.md5(str(datetime.now().timestamp()).encode()).hexdigest()[
            :12
        ]
    )
    importance: float = 1.0  # 重要性评分 (0.0 - 10.0)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "entry_id": self.entry_id,
            "content": self.content,
            "entry_type": self.entry_type,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "importance": self.importance,
        }


class BaseMemory(ABC):
    """
    内存基类

    所有内存实现的抽象基类
    """

    def __init__(self, name: str = "memory"):
        self.name = name

    @abstractmethod
    def add(self, content: str, **metadata) -> MemoryEntry:
        """添加记忆"""
        pass

    @abstractmethod
    def get(self, query: str, limit: int = 5) -> List[MemoryEntry]:
        """检索记忆"""
        pass

    @abstractmethod
    def clear(self) -> None:
        """清空记忆"""
        pass

    @abstractmethod
    def get_recent(self, limit: int = 10) -> List[MemoryEntry]:
        """获取最近的记忆"""
        pass


class BufferMemory(BaseMemory):
    """
    缓冲内存（短期记忆）

    简单的列表存储，适合短期对话历史
    """

    def __init__(self, name: str = "buffer_memory", max_size: int = 100):
        super().__init__(name)
        self.max_size = max_size
        self._entries: List[MemoryEntry] = []

    def add(self, content: str, **metadata) -> MemoryEntry:
        """添加记忆条目"""
        entry = MemoryEntry(
            content=content,
            entry_type=metadata.get("entry_type", "generic"),
            metadata=metadata,
            importance=metadata.get("importance", 1.0),
        )
        self._entries.append(entry)

        # 保持内存大小限制
        if len(self._entries) > self.max_size:
            self._entries = self._entries[-self.max_size:]

        return entry

    def get(self, query: str, limit: int = 5) -> List[MemoryEntry]:
        """
        简单关键词检索

        返回包含查询词的最新条目
        """
        query_lower = query.lower()
        matching = [entry for entry in self._entries if query_lower in entry.content.lower()]
        return matching[-limit:]

    def get_recent(self, limit: int = 10) -> List[MemoryEntry]:
        """获取最近的记忆"""
        return self._entries[-limit:]

    def clear(self) -> None:
        """清空记忆"""
        self._entries.clear()

    def get_all(self) -> List[MemoryEntry]:
        """获取所有记忆"""
        return self._entries.copy()

    def get_formatted_history(self, limit: Optional[int] = None) -> str:
        """获取格式化的历史记录"""
        entries = self._entries[-limit:] if limit else self._entries
        lines = []
        for entry in entries:
            prefix = f"[{entry.entry_type.upper()}]" if entry.entry_type != "generic" else ""
            lines.append(f"{prefix} {entry.content}")
        return "\n".join(lines)


class ConversationMemory(BufferMemory):
    """
    对话专用内存

    针对对话场景优化的内存实现
    """

    def __init__(self, max_turns: int = 20):
        super().__init__(name="conversation_memory", max_size=max_turns * 2)
        self.max_turns = max_turns

    def add_user_message(self, content: str) -> MemoryEntry:
        """添加用户消息"""
        return self.add(content, entry_type="user", role="user")

    def add_assistant_message(self, content: str) -> MemoryEntry:
        """添加助手消息"""
        return self.add(content, entry_type="assistant", role="assistant")

    def add_system_message(self, content: str) -> MemoryEntry:
        """添加系统消息"""
        return self.add(content, entry_type="system", role="system")

    def get_messages_for_model(self, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """
        获取适合模型输入的消息格式

        Returns:
            List[{"role": str, "content": str}]
        """
        entries = self._entries[-limit:] if limit else self._entries
        messages = []
        for entry in entries:
            role = entry.metadata.get("role", entry.entry_type)
            messages.append({"role": role, "content": entry.content})
        return messages

    def get_turns(self) -> int:
        """获取对话轮数"""
        return len([e for e in self._entries if e.entry_type in ("user", "assistant")]) // 2


class VectorMemory(BaseMemory):
    """
    向量内存（语义检索）

    基于向量相似度的记忆检索
    需要外部向量存储支持（如 Chroma, FAISS 等）
    """

    def __init__(
            self,
            name: str = "vector_memory",
            embedding_func: Optional[Callable[[str], List[float]]] = None,
    ):
        super().__init__(name)
        self.embedding_func = embedding_func
        self._entries: List[MemoryEntry] = []
        self._embeddings: List[List[float]] = []

    def add(self, content: str, **metadata) -> MemoryEntry:
        """添加记忆（带向量嵌入）"""
        entry = MemoryEntry(
            content=content,
            entry_type=metadata.get("entry_type", "generic"),
            metadata=metadata,
            importance=metadata.get("importance", 1.0),
        )
        self._entries.append(entry)

        # 计算嵌入
        if self.embedding_func:
            try:
                embedding = self.embedding_func(content)
                self._embeddings.append(embedding)
            except Exception:
                self._embeddings.append([])
        else:
            self._embeddings.append([])

        return entry

    def get(self, query: str, limit: int = 5) -> List[MemoryEntry]:
        """
        语义检索

        基于向量相似度返回最相关的记忆
        """
        if not self.embedding_func or not self._entries:
            # 降级为关键词检索
            query_lower = query.lower()
            matching = [entry for entry in self._entries if query_lower in entry.content.lower()]
            return matching[-limit:]

        try:
            query_embedding = self.embedding_func(query)
            # 计算相似度（余弦相似度）
            similarities = []
            for emb in self._embeddings:
                if emb:
                    sim = self._cosine_similarity(query_embedding, emb)
                    similarities.append(sim)
                else:
                    similarities.append(0.0)

            # 排序并返回 top-k
            indexed = list(enumerate(similarities))
            indexed.sort(key=lambda x: x[1], reverse=True)
            return [self._entries[i] for i, _ in indexed[:limit]]
        except Exception:
            # 降级处理
            query_lower = query.lower()
            matching = [entry for entry in self._entries if query_lower in entry.content.lower()]
            return matching[-limit:]

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """计算余弦相似度"""
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def get_recent(self, limit: int = 10) -> List[MemoryEntry]:
        """获取最近的记忆"""
        return self._entries[-limit:]

    def clear(self) -> None:
        """清空记忆"""
        self._entries.clear()
        self._embeddings.clear()


class EntityMemory(BaseMemory):
    """
    实体内存

    存储和检索关键实体信息（人名、地点、概念等）
    """

    def __init__(self, name: str = "entity_memory"):
        super().__init__(name)
        self._entities: Dict[str, MemoryEntry] = {}

    def add(self, content: str, entity_name: Optional[str] = None, **metadata) -> MemoryEntry:
        """
        添加实体

        Args:
            content: 实体描述
            entity_name: 实体名称（不指定则使用 content）
        """
        name = entity_name or content
        entry = MemoryEntry(
            content=content,
            entry_type="entity",
            metadata={"entity_name": name, **metadata},
            importance=metadata.get("importance", 5.0),  # 实体默认重要性较高
        )
        self._entities[name] = entry
        return entry

    def get(self, query: str, limit: int = 5) -> List[MemoryEntry]:
        """获取实体"""
        if query in self._entities:
            return [self._entities[query]]
        # 模糊匹配
        results = []
        for name, entry in self._entities.items():
            if query.lower() in name.lower():
                results.append(entry)
        return results[:limit]

    def get_entity(self, entity_name: str) -> Optional[MemoryEntry]:
        """精确获取单个实体"""
        return self._entities.get(entity_name)

    def update_entity(self, entity_name: str, content: str, **metadata) -> Optional[MemoryEntry]:
        """更新实体"""
        if entity_name in self._entities:
            entry = self._entities[entity_name]
            entry.content = content
            entry.metadata.update(metadata)
            entry.timestamp = datetime.now()
            return entry
        return None

    def get_recent(self, limit: int = 10) -> List[MemoryEntry]:
        """获取最近更新的实体"""
        sorted_entities = sorted(self._entities.values(), key=lambda e: e.timestamp, reverse=True)
        return sorted_entities[:limit]

    def clear(self) -> None:
        """清空所有实体"""
        self._entities.clear()

    def list_entities(self) -> List[str]:
        """列出所有实体名称"""
        return list(self._entities.keys())
