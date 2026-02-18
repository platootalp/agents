"""
聊天消息历史实现
提供多种消息历史存储方式：内存、文件、数据库等
"""

# genAI_main_start
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import os
from pathlib import Path

from .base_memory import BaseChatMessageHistory
from ..messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, Message


class InMemoryChatMessageHistory(BaseChatMessageHistory):
    """内存聊天消息历史
    
    将消息存储在内存中，适用于单次会话
    
    Attributes:
        _messages: 消息列表
    """
    
    def __init__(self, messages: Optional[List[Message]] = None):
        """初始化内存消息历史
        
        Args:
            messages: 初始消息列表（可选）
        """
        self._messages: List[Message] = messages or []
    
    @property
    def messages(self) -> List[Message]:
        """获取所有消息"""
        return self._messages
    
    def add_message(self, message: Message) -> None:
        """添加消息"""
        self._messages.append(message)
    
    def clear(self) -> None:
        """清空消息历史"""
        self._messages = []
    
    def get_last_n_messages(self, n: int) -> List[Message]:
        """获取最后n条消息
        
        Args:
            n: 消息数量
        
        Returns:
            最后n条消息的列表
        """
        return self._messages[-n:] if n > 0 else []
    
    def get_messages_by_role(self, role: str) -> List[Message]:
        """按角色获取消息
        
        Args:
            role: 消息角色
        
        Returns:
            指定角色的消息列表
        """
        return [m for m in self._messages if m.role == role]
    
    def __repr__(self) -> str:
        return f"InMemoryChatMessageHistory(messages={len(self._messages)})"


class FileChatMessageHistory(BaseChatMessageHistory):
    """文件聊天消息历史
    
    将消息持久化到JSON文件中
    
    Attributes:
        file_path: 文件路径
        encoding: 文件编码
    """
    
    def __init__(
        self,
        file_path: str,
        encoding: str = "utf-8",
        auto_save: bool = True
    ):
        """初始化文件消息历史
        
        Args:
            file_path: 文件路径
            encoding: 文件编码
            auto_save: 是否自动保存
        """
        self.file_path = Path(file_path)
        self.encoding = encoding
        self.auto_save = auto_save
        self._messages: List[Message] = []
        
        # 如果文件存在，加载消息
        if self.file_path.exists():
            self._load_messages()
        else:
            # 确保目录存在
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
    
    @property
    def messages(self) -> List[Message]:
        """获取所有消息"""
        return self._messages
    
    def add_message(self, message: Message) -> None:
        """添加消息"""
        self._messages.append(message)
        if self.auto_save:
            self._save_messages()
    
    def clear(self) -> None:
        """清空消息历史"""
        self._messages = []
        if self.auto_save:
            self._save_messages()
    
    def _load_messages(self) -> None:
        """从文件加载消息"""
        try:
            with open(self.file_path, "r", encoding=self.encoding) as f:
                data = json.load(f)
                self._messages = [Message.from_dict(m) for m in data.get("messages", [])]
        except (json.JSONDecodeError, FileNotFoundError):
            self._messages = []
    
    def _save_messages(self) -> None:
        """保存消息到文件"""
        data = {
            "messages": [m.to_dict() for m in self._messages],
            "updated_at": datetime.now().isoformat()
        }
        with open(self.file_path, "w", encoding=self.encoding) as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def save(self) -> None:
        """手动保存消息"""
        self._save_messages()
    
    def reload(self) -> None:
        """重新加载消息"""
        self._load_messages()
    
    def __repr__(self) -> str:
        return f"FileChatMessageHistory(file_path='{self.file_path}', messages={len(self._messages)})"


class SessionChatMessageHistory(BaseChatMessageHistory):
    """会话聊天消息历史
    
    支持多会话管理的消息历史
    
    Attributes:
        session_id: 会话ID
        store: 消息存储字典（会话ID -> 消息列表）
    """
    
    # 类级别的存储，所有实例共享
    _store: Dict[str, List[Message]] = {}
    
    def __init__(self, session_id: str):
        """初始化会话消息历史
        
        Args:
            session_id: 会话ID
        """
        self.session_id = session_id
        if session_id not in self._store:
            self._store[session_id] = []
    
    @property
    def messages(self) -> List[Message]:
        """获取当前会话的所有消息"""
        return self._store.get(self.session_id, [])
    
    def add_message(self, message: Message) -> None:
        """添加消息到当前会话"""
        if self.session_id not in self._store:
            self._store[self.session_id] = []
        self._store[self.session_id].append(message)
    
    def clear(self) -> None:
        """清空当前会话的消息历史"""
        self._store[self.session_id] = []
    
    @classmethod
    def get_all_sessions(cls) -> List[str]:
        """获取所有会话ID
        
        Returns:
            会话ID列表
        """
        return list(cls._store.keys())
    
    @classmethod
    def clear_all_sessions(cls) -> None:
        """清空所有会话"""
        cls._store.clear()
    
    @classmethod
    def delete_session(cls, session_id: str) -> bool:
        """删除指定会话
        
        Args:
            session_id: 会话ID
        
        Returns:
            是否删除成功
        """
        if session_id in cls._store:
            del cls._store[session_id]
            return True
        return False
    
    def __repr__(self) -> str:
        return f"SessionChatMessageHistory(session_id='{self.session_id}', messages={len(self.messages)})"


def get_chat_history(
    history_type: str = "memory",
    **kwargs
) -> BaseChatMessageHistory:
    """获取聊天消息历史实例的工厂函数
    
    Args:
        history_type: 历史类型 (memory/file/session)
        **kwargs: 传递给具体类的参数
    
    Returns:
        BaseChatMessageHistory实例
    
    Raises:
        ValueError: 不支持的历史类型
    
    Examples:
        >>> # 内存历史
        >>> history = get_chat_history("memory")
        
        >>> # 文件历史
        >>> history = get_chat_history("file", file_path="chat.json")
        
        >>> # 会话历史
        >>> history = get_chat_history("session", session_id="user123")
    """
    history_map = {
        "memory": InMemoryChatMessageHistory,
        "file": FileChatMessageHistory,
        "session": SessionChatMessageHistory,
    }
    
    history_class = history_map.get(history_type.lower())
    if not history_class:
        raise ValueError(
            f"不支持的历史类型: {history_type}. "
            f"支持的类型: {list(history_map.keys())}"
        )
    
    return history_class(**kwargs)
# genAI_main_end
