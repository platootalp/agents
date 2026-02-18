"""
消息类模块
定义统一的消息类，供语言模型和记忆系统共同使用
对齐 LangChain 的消息设计
"""

# genAI_main_start
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC


class BaseMessage:
    """基础消息类 (与 LangChain 兼容)
    
    所有消息类型的基类，包含消息内容和角色信息
    
    Attributes:
        content: 消息内容
        role: 消息角色（system、user、assistant等）
        timestamp: 消息时间戳
        additional_kwargs: 额外的关键字参数
    """
    
    def __init__(
        self,
        content: str,
        role: str,
        timestamp: Optional[datetime] = None,
        **kwargs: Any
    ):
        """初始化消息
        
        Args:
            content: 消息内容
            role: 消息角色
            timestamp: 时间戳（可选，默认为当前时间）
            **kwargs: 额外的关键字参数
        """
        self.content = content
        self.role = role
        self.timestamp = timestamp or datetime.now()
        self.additional_kwargs = kwargs
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式（用于API调用）
        
        Returns:
            包含 role 和 content 的字典
        """
        result = {"role": self.role, "content": self.content}
        if self.additional_kwargs:
            result.update(self.additional_kwargs)
        return result
    
    def to_full_dict(self) -> Dict[str, Any]:
        """转换为完整字典格式（包含时间戳和元数据）
        
        Returns:
            包含所有信息的字典
        """
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metadata": self.additional_kwargs
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseMessage":
        """从字典创建消息
        
        Args:
            data: 消息字典
        
        Returns:
            BaseMessage实例
        """
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        # 根据role选择合适的子类
        role = data.get("role", "user")
        content = data.get("content", "")
        metadata = data.get("metadata", {})
        
        message_classes = {
            "user": HumanMessage,
            "assistant": AIMessage,
            "system": SystemMessage,
            "function": FunctionMessage,
        }
        
        message_class = message_classes.get(role, BaseMessage)
        
        if message_class == FunctionMessage:
            return message_class(
                content=content,
                name=metadata.get("name", ""),
                timestamp=timestamp,
                **metadata
            )
        elif message_class == BaseMessage:
            return message_class(
                content=content,
                role=role,
                timestamp=timestamp,
                **metadata
            )
        else:
            return message_class(content=content, timestamp=timestamp, **metadata)
    
    def copy(self) -> "BaseMessage":
        """创建消息副本
        
        Returns:
            消息副本
        """
        return self.__class__(
            content=self.content,
            role=self.role,
            timestamp=self.timestamp,
            **self.additional_kwargs
        )
    
    def __eq__(self, other: object) -> bool:
        """判断消息是否相等"""
        if not isinstance(other, BaseMessage):
            return False
        return self.role == other.role and self.content == other.content
    
    def __repr__(self) -> str:
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"{self.__class__.__name__}(role='{self.role}', content='{content_preview}')"


class HumanMessage(BaseMessage):
    """人类消息类 (用户输入)
    
    表示用户发送的消息，对应 role='user'
    
    Examples:
        >>> msg = HumanMessage(content="你好")
        >>> print(msg.role)  # 'user'
    """
    
    def __init__(self, content: str, timestamp: Optional[datetime] = None, **kwargs: Any):
        super().__init__(content=content, role="user", timestamp=timestamp, **kwargs)


class AIMessage(BaseMessage):
    """AI 消息类 (助手回复)
    
    表示 AI 助手的回复，对应 role='assistant'
    
    Examples:
        >>> msg = AIMessage(content="你好！有什么可以帮助你的？")
        >>> print(msg.role)  # 'assistant'
    """
    
    def __init__(self, content: str, timestamp: Optional[datetime] = None, **kwargs: Any):
        super().__init__(content=content, role="assistant", timestamp=timestamp, **kwargs)


class SystemMessage(BaseMessage):
    """系统消息类
    
    用于设置系统级别的指令或行为，对应 role='system'
    
    Examples:
        >>> msg = SystemMessage(content="你是一个友好的助手")
        >>> print(msg.role)  # 'system'
    """
    
    def __init__(self, content: str, timestamp: Optional[datetime] = None, **kwargs: Any):
        super().__init__(content=content, role="system", timestamp=timestamp, **kwargs)


class FunctionMessage(BaseMessage):
    """函数消息类
    
    用于函数调用的返回结果
    
    Attributes:
        name: 函数名称
    
    Examples:
        >>> msg = FunctionMessage(content='{"result": 42}', name="calculator")
        >>> print(msg.role)  # 'function'
        >>> print(msg.name)  # 'calculator'
    """
    
    def __init__(
        self,
        content: str,
        name: str,
        timestamp: Optional[datetime] = None,
        **kwargs: Any
    ):
        super().__init__(content=content, role="function", timestamp=timestamp, name=name, **kwargs)
        self.name = name
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = super().to_dict()
        result["name"] = self.name
        return result


class ToolMessage(BaseMessage):
    """工具消息类
    
    用于工具调用的返回结果（OpenAI风格）
    
    Attributes:
        tool_call_id: 工具调用ID
    
    Examples:
        >>> msg = ToolMessage(content='{"result": 42}', tool_call_id="call_123")
        >>> print(msg.role)  # 'tool'
    """
    
    def __init__(
        self,
        content: str,
        tool_call_id: str,
        timestamp: Optional[datetime] = None,
        **kwargs: Any
    ):
        super().__init__(content=content, role="tool", timestamp=timestamp, **kwargs)
        self.tool_call_id = tool_call_id
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = super().to_dict()
        result["tool_call_id"] = self.tool_call_id
        return result


# ==================== 消息工具函数 ====================

def messages_to_dict(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
    """将消息列表转换为字典列表
    
    Args:
        messages: 消息列表
    
    Returns:
        字典列表（适用于API调用）
    
    Examples:
        >>> msgs = [HumanMessage("你好"), AIMessage("你好！")]
        >>> dicts = messages_to_dict(msgs)
    """
    return [m.to_dict() for m in messages]


def messages_from_dict(data: List[Dict[str, Any]]) -> List[BaseMessage]:
    """从字典列表创建消息列表
    
    Args:
        data: 字典列表
    
    Returns:
        消息列表
    """
    return [BaseMessage.from_dict(d) for d in data]


def get_buffer_string(
    messages: List[BaseMessage],
    human_prefix: str = "Human",
    ai_prefix: str = "AI"
) -> str:
    """将消息列表格式化为字符串
    
    Args:
        messages: 消息列表
        human_prefix: 人类消息前缀
        ai_prefix: AI消息前缀
    
    Returns:
        格式化的字符串
    
    Examples:
        >>> msgs = [HumanMessage("你好"), AIMessage("你好！")]
        >>> print(get_buffer_string(msgs))
        # Human: 你好
        # AI: 你好！
    """
    formatted = []
    for msg in messages:
        if msg.role == "user":
            formatted.append(f"{human_prefix}: {msg.content}")
        elif msg.role == "assistant":
            formatted.append(f"{ai_prefix}: {msg.content}")
        elif msg.role == "system":
            formatted.append(f"System: {msg.content}")
        elif msg.role == "function":
            name = getattr(msg, 'name', 'function')
            formatted.append(f"Function({name}): {msg.content}")
        elif msg.role == "tool":
            tool_id = getattr(msg, 'tool_call_id', 'tool')
            formatted.append(f"Tool({tool_id}): {msg.content}")
        else:
            formatted.append(f"{msg.role}: {msg.content}")
    return "\n".join(formatted)


# ==================== 类型别名（向后兼容） ====================

# 为了向后兼容 memory 模块中使用的 Message 名称
Message = BaseMessage
# genAI_main_end
