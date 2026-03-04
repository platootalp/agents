"""
记忆管理器
提供统一的记忆管理接口，支持多种记忆类型的组合使用
"""

# genAI_main_start
from typing import List, Dict, Any, Optional, Type, Union
from dataclasses import dataclass, field
from datetime import datetime

from .base_memory import BaseMemory, MemoryVariables, MemoryType
from .buffer_memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationTokenBufferMemory
)
from .summary_memory import (
    ConversationSummaryMemory,
    ConversationSummaryBufferMemory
)
from .vector_memory import VectorStoreMemory
from .entity_memory import ConversationEntityMemory


class CombinedMemory(BaseMemory):
    """组合记忆
    
    将多个记忆系统组合使用，支持短期和长期记忆的协同工作
    
    Attributes:
        memories: 记忆实例列表
    """
    
    def __init__(
        self,
        memories: List[BaseMemory],
        memory_key: str = "combined_memory",
        verbose: bool = False
    ):
        """初始化组合记忆
        
        Args:
            memories: 记忆实例列表
            memory_key: 记忆键名
            verbose: 是否输出详细日志
        """
        super().__init__(
            memory_type=MemoryType.WORKING,
            memory_key=memory_key,
            verbose=verbose
        )
        self.memories = memories
    
    @property
    def memory_variables(self) -> List[str]:
        """返回所有记忆的变量名列表"""
        variables = set()
        for memory in self.memories:
            variables.update(memory.memory_variables)
        return list(variables)
    
    def load_memory_variables(self, inputs: Optional[Dict[str, Any]] = None) -> MemoryVariables:
        """加载所有记忆的变量"""
        combined = MemoryVariables()
        
        for memory in self.memories:
            mem_vars = memory.load_memory_variables(inputs)
            
            # 合并历史
            if mem_vars.history:
                if combined.history:
                    combined.history += f"\n\n---\n\n{mem_vars.history}"
                else:
                    combined.history = mem_vars.history
            
            # 合并消息
            combined.messages.extend(mem_vars.messages)
            
            # 合并上下文
            if mem_vars.context:
                if combined.context:
                    combined.context += f"\n\n{mem_vars.context}"
                else:
                    combined.context = mem_vars.context
            
            # 合并摘要
            if mem_vars.summary:
                if combined.summary:
                    combined.summary += f"\n\n{mem_vars.summary}"
                else:
                    combined.summary = mem_vars.summary
            
            # 合并实体
            combined.entities.update(mem_vars.entities)
            
            # 合并额外变量
            combined.extra.update(mem_vars.extra)
        
        return combined
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """保存上下文到所有记忆"""
        for memory in self.memories:
            memory.save_context(inputs, outputs)
    
    def clear(self) -> None:
        """清空所有记忆"""
        for memory in self.memories:
            memory.clear()
    
    def add_memory(self, memory: BaseMemory) -> None:
        """添加记忆
        
        Args:
            memory: 记忆实例
        """
        self.memories.append(memory)
    
    def remove_memory(self, memory_type: Type[BaseMemory]) -> bool:
        """移除指定类型的记忆
        
        Args:
            memory_type: 记忆类型
        
        Returns:
            是否移除成功
        """
        for i, memory in enumerate(self.memories):
            if isinstance(memory, memory_type):
                self.memories.pop(i)
                return True
        return False
    
    def get_memory(self, memory_type: Type[BaseMemory]) -> Optional[BaseMemory]:
        """获取指定类型的记忆
        
        Args:
            memory_type: 记忆类型
        
        Returns:
            记忆实例或None
        """
        for memory in self.memories:
            if isinstance(memory, memory_type):
                return memory
        return None
    
    def __repr__(self) -> str:
        memory_types = [type(m).__name__ for m in self.memories]
        return f"CombinedMemory(memories={memory_types})"


class MemoryManager:
    """记忆管理器
    
    提供记忆的创建、管理和使用的统一接口
    """
    
    # 记忆类型映射
    MEMORY_TYPES: Dict[str, Type[BaseMemory]] = {
        "buffer": ConversationBufferMemory,
        "buffer_window": ConversationBufferWindowMemory,
        "token_buffer": ConversationTokenBufferMemory,
        "summary": ConversationSummaryMemory,
        "summary_buffer": ConversationSummaryBufferMemory,
        "vector": VectorStoreMemory,
        "entity": ConversationEntityMemory,
    }
    
    def __init__(self, verbose: bool = False):
        """初始化记忆管理器
        
        Args:
            verbose: 是否输出详细日志
        """
        self.verbose = verbose
        self._memories: Dict[str, BaseMemory] = {}
    
    def create_memory(
        self,
        memory_type: str,
        name: Optional[str] = None,
        **kwargs
    ) -> BaseMemory:
        """创建记忆实例
        
        Args:
            memory_type: 记忆类型
            name: 记忆名称（可选，用于后续引用）
            **kwargs: 传递给记忆类的参数
        
        Returns:
            记忆实例
        
        Raises:
            ValueError: 不支持的记忆类型
        
        Examples:
            >>> manager = MemoryManager()
            >>> memory = manager.create_memory("buffer", k=5)
            >>> memory = manager.create_memory("vector", retrieval_k=3)
        """
        memory_class = self.MEMORY_TYPES.get(memory_type.lower())
        if not memory_class:
            raise ValueError(
                f"不支持的记忆类型: {memory_type}. "
                f"支持的类型: {list(self.MEMORY_TYPES.keys())}"
            )
        
        # 添加verbose参数
        if "verbose" not in kwargs:
            kwargs["verbose"] = self.verbose
        
        memory = memory_class(**kwargs)
        
        # 如果指定了名称，保存引用
        if name:
            self._memories[name] = memory
            if self.verbose:
                print(f"[MemoryManager] 创建记忆 '{name}' ({memory_type})")
        
        return memory
    
    def get_memory(self, name: str) -> Optional[BaseMemory]:
        """获取记忆实例
        
        Args:
            name: 记忆名称
        
        Returns:
            记忆实例或None
        """
        return self._memories.get(name)
    
    def list_memories(self) -> Dict[str, str]:
        """列出所有记忆
        
        Returns:
            记忆名称到类型的映射
        """
        return {name: type(memory).__name__ for name, memory in self._memories.items()}
    
    def delete_memory(self, name: str) -> bool:
        """删除记忆
        
        Args:
            name: 记忆名称
        
        Returns:
            是否删除成功
        """
        if name in self._memories:
            del self._memories[name]
            return True
        return False
    
    def create_combined_memory(
        self,
        memory_types: List[str],
        name: Optional[str] = None,
        configs: Optional[Dict[str, Dict]] = None
    ) -> CombinedMemory:
        """创建组合记忆
        
        Args:
            memory_types: 记忆类型列表
            name: 组合记忆名称
            configs: 各记忆类型的配置
        
        Returns:
            组合记忆实例
        
        Examples:
            >>> manager = MemoryManager()
            >>> combined = manager.create_combined_memory(
            ...     ["buffer_window", "entity"],
            ...     configs={
            ...         "buffer_window": {"k": 5},
            ...         "entity": {}
            ...     }
            ... )
        """
        configs = configs or {}
        memories = []
        
        for mem_type in memory_types:
            config = configs.get(mem_type, {})
            memory = self.create_memory(mem_type, **config)
            memories.append(memory)
        
        combined = CombinedMemory(memories, verbose=self.verbose)
        
        if name:
            self._memories[name] = combined
        
        return combined
    
    def clear_all(self) -> None:
        """清空所有记忆"""
        for memory in self._memories.values():
            memory.clear()
    
    @classmethod
    def get_available_types(cls) -> List[str]:
        """获取所有可用的记忆类型
        
        Returns:
            记忆类型列表
        """
        return list(cls.MEMORY_TYPES.keys())
    
    @classmethod
    def get_memory_info(cls) -> Dict[str, str]:
        """获取记忆类型的说明信息
        
        Returns:
            记忆类型到说明的映射
        """
        return {
            "buffer": "完整对话缓冲区，存储所有历史消息",
            "buffer_window": "滑动窗口缓冲区，只保留最近k轮对话",
            "token_buffer": "Token限制缓冲区，按token数量限制历史",
            "summary": "摘要记忆，使用LLM生成对话摘要",
            "summary_buffer": "摘要+缓冲区混合，保留最近对话并维护历史摘要",
            "vector": "向量存储记忆，基于语义相似度检索",
            "entity": "实体记忆，提取和维护对话中的实体信息",
        }
    
    def __repr__(self) -> str:
        return f"MemoryManager(memories={list(self._memories.keys())})"


def create_memory(
    memory_type: str,
    **kwargs
) -> BaseMemory:
    """快速创建记忆实例
    
    Args:
        memory_type: 记忆类型
        **kwargs: 传递给记忆类的参数
    
    Returns:
        记忆实例
    
    Examples:
        >>> # 创建缓冲区记忆
        >>> memory = create_memory("buffer")
        
        >>> # 创建滑动窗口记忆
        >>> memory = create_memory("buffer_window", k=5)
        
        >>> # 创建向量记忆
        >>> memory = create_memory("vector", retrieval_k=3)
    """
    manager = MemoryManager()
    return manager.create_memory(memory_type, **kwargs)


def list_memory_types() -> str:
    """列出所有可用的记忆类型
    
    Returns:
        格式化的记忆类型列表
    """
    info = MemoryManager.get_memory_info()
    
    lines = ["\n=== 可用的记忆类型 ===\n"]
    
    # 短期记忆
    lines.append("📦 短期记忆 (Short-term Memory):")
    for mem_type in ["buffer", "buffer_window", "token_buffer", "summary", "summary_buffer"]:
        lines.append(f"  • {mem_type}: {info[mem_type]}")
    
    # 长期记忆
    lines.append("\n📦 长期记忆 (Long-term Memory):")
    for mem_type in ["vector", "entity"]:
        lines.append(f"  • {mem_type}: {info[mem_type]}")
    
    lines.append("")
    return "\n".join(lines)
# genAI_main_end
