"""
缓冲区记忆实现（短期记忆）
提供多种缓冲区记忆策略：完整缓冲、滑动窗口、Token限制等
"""

# genAI_main_start
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import copy

from .base_memory import (
    BaseMemory, BaseChatMessageHistory,
    MemoryVariables, MemoryType
)
from ..messages import BaseMessage, HumanMessage, AIMessage, Message
from .chat_history import InMemoryChatMessageHistory


class ConversationBufferMemory(BaseMemory):
    """对话缓冲区记忆
    
    存储完整的对话历史，适用于短对话场景
    
    Attributes:
        chat_history: 聊天消息历史
        input_key: 输入键名
        output_key: 输出键名
    """
    
    def __init__(
        self,
        chat_history: Optional[BaseChatMessageHistory] = None,
        memory_key: str = "history",
        input_key: str = "input",
        output_key: str = "output",
        return_messages: bool = False,
        human_prefix: str = "Human",
        ai_prefix: str = "AI",
        verbose: bool = False
    ):
        """初始化对话缓冲区记忆
        
        Args:
            chat_history: 聊天消息历史（可选，默认使用内存历史）
            memory_key: 记忆键名
            input_key: 输入键名
            output_key: 输出键名
            return_messages: 是否返回消息列表
            human_prefix: 人类消息前缀
            ai_prefix: AI消息前缀
            verbose: 是否输出详细日志
        """
        super().__init__(
            memory_type=MemoryType.SHORT_TERM,
            memory_key=memory_key,
            return_messages=return_messages,
            human_prefix=human_prefix,
            ai_prefix=ai_prefix,
            verbose=verbose
        )
        self.chat_history = chat_history or InMemoryChatMessageHistory()
        self.input_key = input_key
        self.output_key = output_key
    
    @property
    def memory_variables(self) -> List[str]:
        """返回记忆变量名列表"""
        return [self.memory_key]
    
    @property
    def buffer(self) -> List[Message]:
        """获取消息缓冲区"""
        return self.chat_history.messages
    
    @property
    def buffer_as_str(self) -> str:
        """获取格式化的缓冲区字符串"""
        return self.format_messages(self.buffer)
    
    def load_memory_variables(self, inputs: Optional[Dict[str, Any]] = None) -> MemoryVariables:
        """加载记忆变量"""
        if self.return_messages:
            return MemoryVariables(
                messages=copy.deepcopy(self.buffer),
                history=self.buffer_as_str
            )
        return MemoryVariables(
            history=self.buffer_as_str,
            messages=copy.deepcopy(self.buffer)
        )
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """保存对话上下文"""
        input_str = inputs.get(self.input_key, "")
        output_str = outputs.get(self.output_key, "")
        
        if input_str:
            self.chat_history.add_user_message(input_str)
            if self.verbose:
                print(f"[Memory] 保存用户消息: {input_str[:50]}...")
        
        if output_str:
            self.chat_history.add_ai_message(output_str)
            if self.verbose:
                print(f"[Memory] 保存AI消息: {output_str[:50]}...")
    
    def clear(self) -> None:
        """清空记忆"""
        self.chat_history.clear()
        if self.verbose:
            print("[Memory] 记忆已清空")
    
    def __repr__(self) -> str:
        return f"ConversationBufferMemory(messages={len(self.buffer)})"


class ConversationBufferWindowMemory(BaseMemory):
    """滑动窗口对话记忆
    
    只保留最近k轮对话，适用于长对话场景
    
    Attributes:
        k: 保留的对话轮数
        chat_history: 聊天消息历史
    """
    
    def __init__(
        self,
        k: int = 5,
        chat_history: Optional[BaseChatMessageHistory] = None,
        memory_key: str = "history",
        input_key: str = "input",
        output_key: str = "output",
        return_messages: bool = False,
        human_prefix: str = "Human",
        ai_prefix: str = "AI",
        verbose: bool = False
    ):
        """初始化滑动窗口记忆
        
        Args:
            k: 保留的对话轮数（一轮 = 一问一答）
            chat_history: 聊天消息历史
            memory_key: 记忆键名
            input_key: 输入键名
            output_key: 输出键名
            return_messages: 是否返回消息列表
            human_prefix: 人类消息前缀
            ai_prefix: AI消息前缀
            verbose: 是否输出详细日志
        """
        super().__init__(
            memory_type=MemoryType.SHORT_TERM,
            memory_key=memory_key,
            return_messages=return_messages,
            human_prefix=human_prefix,
            ai_prefix=ai_prefix,
            verbose=verbose
        )
        self.k = k
        self.chat_history = chat_history or InMemoryChatMessageHistory()
        self.input_key = input_key
        self.output_key = output_key
    
    @property
    def memory_variables(self) -> List[str]:
        """返回记忆变量名列表"""
        return [self.memory_key]
    
    @property
    def buffer(self) -> List[Message]:
        """获取窗口内的消息"""
        # 保留最后k轮对话（2k条消息）
        messages = self.chat_history.messages
        window_size = self.k * 2
        return messages[-window_size:] if len(messages) > window_size else messages
    
    @property
    def buffer_as_str(self) -> str:
        """获取格式化的缓冲区字符串"""
        return self.format_messages(self.buffer)
    
    def load_memory_variables(self, inputs: Optional[Dict[str, Any]] = None) -> MemoryVariables:
        """加载记忆变量"""
        if self.return_messages:
            return MemoryVariables(
                messages=copy.deepcopy(self.buffer),
                history=self.buffer_as_str
            )
        return MemoryVariables(
            history=self.buffer_as_str,
            messages=copy.deepcopy(self.buffer)
        )
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """保存对话上下文"""
        input_str = inputs.get(self.input_key, "")
        output_str = outputs.get(self.output_key, "")
        
        if input_str:
            self.chat_history.add_user_message(input_str)
        
        if output_str:
            self.chat_history.add_ai_message(output_str)
        
        if self.verbose:
            total = len(self.chat_history.messages)
            window = len(self.buffer)
            print(f"[Memory] 总消息: {total}, 窗口内: {window}")
    
    def clear(self) -> None:
        """清空记忆"""
        self.chat_history.clear()
    
    def __repr__(self) -> str:
        return f"ConversationBufferWindowMemory(k={self.k}, buffer_size={len(self.buffer)})"


class ConversationTokenBufferMemory(BaseMemory):
    """Token限制对话记忆
    
    根据token数量限制保留消息，适用于需要精确控制上下文长度的场景
    
    Attributes:
        max_token_limit: 最大token数量限制
        token_counter: token计数函数
    """
    
    def __init__(
        self,
        max_token_limit: int = 2000,
        token_counter: Optional[Callable[[str], int]] = None,
        chat_history: Optional[BaseChatMessageHistory] = None,
        memory_key: str = "history",
        input_key: str = "input",
        output_key: str = "output",
        return_messages: bool = False,
        human_prefix: str = "Human",
        ai_prefix: str = "AI",
        verbose: bool = False
    ):
        """初始化Token限制记忆
        
        Args:
            max_token_limit: 最大token数量
            token_counter: token计数函数（默认按字符数估算）
            chat_history: 聊天消息历史
            memory_key: 记忆键名
            input_key: 输入键名
            output_key: 输出键名
            return_messages: 是否返回消息列表
            human_prefix: 人类消息前缀
            ai_prefix: AI消息前缀
            verbose: 是否输出详细日志
        """
        super().__init__(
            memory_type=MemoryType.SHORT_TERM,
            memory_key=memory_key,
            return_messages=return_messages,
            human_prefix=human_prefix,
            ai_prefix=ai_prefix,
            verbose=verbose
        )
        self.max_token_limit = max_token_limit
        self.token_counter = token_counter or self._default_token_counter
        self.chat_history = chat_history or InMemoryChatMessageHistory()
        self.input_key = input_key
        self.output_key = output_key
    
    @staticmethod
    def _default_token_counter(text: str) -> int:
        """默认token计数器（简单估算）
        
        粗略估算：中文约1.5字符/token，英文约4字符/token
        这里使用简单的字符数 / 3 作为估算
        """
        return len(text) // 3 + 1
    
    @property
    def memory_variables(self) -> List[str]:
        """返回记忆变量名列表"""
        return [self.memory_key]
    
    @property
    def buffer(self) -> List[Message]:
        """获取在token限制内的消息"""
        messages = self.chat_history.messages
        if not messages:
            return []
        
        # 从最新消息开始，累计token直到超过限制
        selected = []
        current_tokens = 0
        
        for msg in reversed(messages):
            msg_tokens = self.token_counter(msg.content)
            if current_tokens + msg_tokens > self.max_token_limit:
                break
            selected.insert(0, msg)
            current_tokens += msg_tokens
        
        return selected
    
    @property
    def buffer_as_str(self) -> str:
        """获取格式化的缓冲区字符串"""
        return self.format_messages(self.buffer)
    
    @property
    def current_token_count(self) -> int:
        """获取当前缓冲区的token数量"""
        return sum(self.token_counter(m.content) for m in self.buffer)
    
    def load_memory_variables(self, inputs: Optional[Dict[str, Any]] = None) -> MemoryVariables:
        """加载记忆变量"""
        if self.return_messages:
            return MemoryVariables(
                messages=copy.deepcopy(self.buffer),
                history=self.buffer_as_str
            )
        return MemoryVariables(
            history=self.buffer_as_str,
            messages=copy.deepcopy(self.buffer)
        )
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """保存对话上下文"""
        input_str = inputs.get(self.input_key, "")
        output_str = outputs.get(self.output_key, "")
        
        if input_str:
            self.chat_history.add_user_message(input_str)
        
        if output_str:
            self.chat_history.add_ai_message(output_str)
        
        if self.verbose:
            print(f"[Memory] Token使用: {self.current_token_count}/{self.max_token_limit}")
    
    def clear(self) -> None:
        """清空记忆"""
        self.chat_history.clear()
    
    def __repr__(self) -> str:
        return f"ConversationTokenBufferMemory(tokens={self.current_token_count}/{self.max_token_limit})"
# genAI_main_end
