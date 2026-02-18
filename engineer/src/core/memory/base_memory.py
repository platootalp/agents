"""
记忆基础抽象类和接口定义
定义所有记忆类必须实现的标准接口，参考LangChain Memory设计
"""

# genAI_main_start
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import copy

# 从统一的消息模块导入
from ..messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    FunctionMessage,
    Message,  # 别名，向后兼容
    get_buffer_string
)


class MemoryType(Enum):
    """记忆类型枚举"""
    SHORT_TERM = "short_term"      # 短期记忆
    LONG_TERM = "long_term"        # 长期记忆
    WORKING = "working"            # 工作记忆
    EPISODIC = "episodic"          # 情节记忆
    SEMANTIC = "semantic"          # 语义记忆


@dataclass
class MemoryVariables:
    """记忆变量容器
    
    用于存储和传递记忆相关的变量
    
    Attributes:
        history: 对话历史字符串
        messages: 消息列表
        context: 上下文信息
        summary: 摘要信息
        entities: 实体信息
        extra: 额外变量
    """
    history: str = ""
    messages: List[Message] = field(default_factory=list)
    context: str = ""
    summary: str = ""
    entities: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "history": self.history,
            "messages": [m.to_dict() for m in self.messages],
            "context": self.context,
            "summary": self.summary,
            "entities": self.entities,
            "extra": self.extra
        }
    
    def get_prompt_variables(self) -> Dict[str, str]:
        """获取用于提示词的变量"""
        return {
            "history": self.history,
            "context": self.context,
            "summary": self.summary
        }


class BaseMemory(ABC):
    """记忆基础抽象类
    
    所有记忆类必须继承此类并实现必要的抽象方法。
    提供统一的记忆存取接口。
    
    Attributes:
        memory_type: 记忆类型
        memory_key: 记忆键名（用于提示词模板）
        return_messages: 是否返回消息列表而非字符串
        human_prefix: 人类消息前缀
        ai_prefix: AI消息前缀
        verbose: 是否输出详细日志
    """
    
    def __init__(
        self,
        memory_type: MemoryType = MemoryType.SHORT_TERM,
        memory_key: str = "history",
        return_messages: bool = False,
        human_prefix: str = "Human",
        ai_prefix: str = "AI",
        verbose: bool = False
    ):
        """初始化记忆实例
        
        Args:
            memory_type: 记忆类型
            memory_key: 记忆键名
            return_messages: 是否返回消息列表
            human_prefix: 人类消息前缀
            ai_prefix: AI消息前缀
            verbose: 是否输出详细日志
        """
        self.memory_type = memory_type
        self.memory_key = memory_key
        self.return_messages = return_messages
        self.human_prefix = human_prefix
        self.ai_prefix = ai_prefix
        self.verbose = verbose
    
    @property
    @abstractmethod
    def memory_variables(self) -> List[str]:
        """返回此记忆类提供的变量名列表
        
        Returns:
            变量名列表
        """
        pass
    
    @abstractmethod
    def load_memory_variables(self, inputs: Optional[Dict[str, Any]] = None) -> MemoryVariables:
        """加载记忆变量
        
        从记忆中加载相关变量，用于填充提示词模板
        
        Args:
            inputs: 输入变量（可选，用于过滤相关记忆）
        
        Returns:
            MemoryVariables对象，包含记忆变量
        """
        pass
    
    @abstractmethod
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """保存对话上下文
        
        将输入输出对保存到记忆中
        
        Args:
            inputs: 用户输入
            outputs: AI输出
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """清空记忆
        
        清除所有存储的记忆内容
        """
        pass
    
    def add_message(self, message: Message) -> None:
        """添加单条消息
        
        Args:
            message: 要添加的消息
        """
        # 默认实现：通过save_context添加
        if message.role == "user":
            self.save_context({"input": message.content}, {})
        elif message.role == "assistant":
            self.save_context({}, {"output": message.content})
    
    def add_user_message(self, content: str) -> None:
        """添加用户消息
        
        Args:
            content: 消息内容
        """
        self.add_message(HumanMessage(content=content))
    
    def add_ai_message(self, content: str) -> None:
        """添加AI消息
        
        Args:
            content: 消息内容
        """
        self.add_message(AIMessage(content=content))
    
    def format_messages(self, messages: List[BaseMessage]) -> str:
        """格式化消息列表为字符串
        
        Args:
            messages: 消息列表
        
        Returns:
            格式化后的字符串
        """
        return get_buffer_string(
            messages,
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """将记忆序列化为字典
        
        Returns:
            记忆的字典表示
        """
        return {
            "memory_type": self.memory_type.value,
            "memory_key": self.memory_key,
            "return_messages": self.return_messages,
            "human_prefix": self.human_prefix,
            "ai_prefix": self.ai_prefix
        }
    
    def __repr__(self) -> str:
        """记忆的字符串表示"""
        return f"{self.__class__.__name__}(memory_type={self.memory_type.value}, memory_key='{self.memory_key}')"


class BaseChatMessageHistory(ABC):
    """聊天消息历史基类
    
    定义消息历史存储的标准接口
    """
    
    @property
    @abstractmethod
    def messages(self) -> List[Message]:
        """获取所有消息
        
        Returns:
            消息列表
        """
        pass
    
    @abstractmethod
    def add_message(self, message: Message) -> None:
        """添加消息
        
        Args:
            message: 要添加的消息
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """清空消息历史"""
        pass
    
    def add_user_message(self, content: str) -> None:
        """添加用户消息"""
        self.add_message(Message(role="user", content=content))
    
    def add_ai_message(self, content: str) -> None:
        """添加AI消息"""
        self.add_message(Message(role="assistant", content=content))
    
    def add_system_message(self, content: str) -> None:
        """添加系统消息"""
        self.add_message(Message(role="system", content=content))
    
    def get_messages_as_dicts(self) -> List[Dict[str, Any]]:
        """获取消息的字典列表格式"""
        return [m.to_dict() for m in self.messages]
    
    def __len__(self) -> int:
        """返回消息数量"""
        return len(self.messages)
    
    def __iter__(self):
        """迭代消息"""
        return iter(self.messages)
# genAI_main_end
