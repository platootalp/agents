"""
记忆模块
提供短期记忆和长期记忆的完整实现，参考LangChain Memory设计
"""

# genAI_main_start
# 从统一的消息模块导入（向后兼容）
from ..messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    FunctionMessage,
    ToolMessage,
    Message,  # 别名
    messages_to_dict,
    messages_from_dict,
    get_buffer_string
)

from .base_memory import (
    BaseMemory,
    BaseChatMessageHistory,
    MemoryVariables,
    MemoryType
)

from .chat_history import (
    InMemoryChatMessageHistory,
    FileChatMessageHistory,
    SessionChatMessageHistory,
    get_chat_history
)

from .buffer_memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationTokenBufferMemory
)

from .summary_memory import (
    ConversationSummaryMemory,
    ConversationSummaryBufferMemory
)

from .vector_memory import (
    VectorStoreMemory,
    SimpleVectorStore,
    MemoryDocument
)

from .entity_memory import (
    ConversationEntityMemory,
    EntityStore,
    EntityInfo
)

from .memory_manager import (
    CombinedMemory,
    MemoryManager,
    create_memory,
    list_memory_types
)

__all__ = [
    # 消息类（从 messages 模块导入）
    "BaseMessage",
    "HumanMessage",
    "AIMessage",
    "SystemMessage",
    "FunctionMessage",
    "ToolMessage",
    "Message",  # 别名
    "messages_to_dict",
    "messages_from_dict",
    "get_buffer_string",
    
    # 基础类
    "BaseMemory",
    "BaseChatMessageHistory",
    "MemoryVariables",
    "MemoryType",
    
    # 聊天历史
    "InMemoryChatMessageHistory",
    "FileChatMessageHistory",
    "SessionChatMessageHistory",
    "get_chat_history",
    
    # 短期记忆 - 缓冲区
    "ConversationBufferMemory",
    "ConversationBufferWindowMemory",
    "ConversationTokenBufferMemory",
    
    # 短期记忆 - 摘要
    "ConversationSummaryMemory",
    "ConversationSummaryBufferMemory",
    
    # 长期记忆 - 向量存储
    "VectorStoreMemory",
    "SimpleVectorStore",
    "MemoryDocument",
    
    # 长期记忆 - 实体记忆
    "ConversationEntityMemory",
    "EntityStore",
    "EntityInfo",
    
    # 记忆管理
    "CombinedMemory",
    "MemoryManager",
    "create_memory",
    "list_memory_types",
]

__version__ = "1.0.0"
# genAI_main_end
