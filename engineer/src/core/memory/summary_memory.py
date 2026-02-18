"""
摘要记忆实现（短期记忆）
通过LLM生成对话摘要来压缩历史信息
"""

# genAI_main_start
from typing import List, Dict, Any, Optional, Callable, TYPE_CHECKING
from datetime import datetime
import copy

from .base_memory import (
    BaseMemory, BaseChatMessageHistory,
    MemoryVariables, MemoryType
)
from ..messages import BaseMessage, HumanMessage, AIMessage, Message
from .chat_history import InMemoryChatMessageHistory

if TYPE_CHECKING:
    from ..language_models import BaseChatModel


# 默认摘要提示词模板
DEFAULT_SUMMARY_PROMPT = """请将以下对话内容总结为简洁的摘要。保留关键信息和上下文。

当前摘要：
{existing_summary}

新的对话内容：
{new_lines}

请生成更新后的摘要："""


class ConversationSummaryMemory(BaseMemory):
    """对话摘要记忆
    
    使用LLM生成对话摘要，适用于非常长的对话场景
    
    Attributes:
        llm: 用于生成摘要的语言模型
        summary: 当前摘要
        summary_prompt: 摘要提示词模板
        buffer: 未摘要的消息缓冲区
        max_buffer_size: 触发摘要的缓冲区大小
    """
    
    def __init__(
        self,
        llm: Optional["BaseChatModel"] = None,
        summary_prompt: str = DEFAULT_SUMMARY_PROMPT,
        max_buffer_size: int = 6,
        chat_history: Optional[BaseChatMessageHistory] = None,
        memory_key: str = "history",
        input_key: str = "input",
        output_key: str = "output",
        return_messages: bool = False,
        human_prefix: str = "Human",
        ai_prefix: str = "AI",
        verbose: bool = False
    ):
        """初始化摘要记忆
        
        Args:
            llm: 用于生成摘要的语言模型（可选，不提供则使用简单截断）
            summary_prompt: 摘要提示词模板
            max_buffer_size: 触发摘要生成的缓冲区消息数量
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
        self.llm = llm
        self.summary_prompt = summary_prompt
        self.max_buffer_size = max_buffer_size
        self.chat_history = chat_history or InMemoryChatMessageHistory()
        self.input_key = input_key
        self.output_key = output_key
        self._summary: str = ""
        self._pending_messages: List[Message] = []
    
    @property
    def memory_variables(self) -> List[str]:
        """返回记忆变量名列表"""
        return [self.memory_key, "summary"]
    
    @property
    def summary(self) -> str:
        """获取当前摘要"""
        return self._summary
    
    @property
    def buffer(self) -> List[Message]:
        """获取未摘要的消息缓冲区"""
        return self._pending_messages
    
    @property
    def buffer_as_str(self) -> str:
        """获取格式化的缓冲区字符串"""
        return self.format_messages(self.buffer)
    
    def load_memory_variables(self, inputs: Optional[Dict[str, Any]] = None) -> MemoryVariables:
        """加载记忆变量"""
        # 组合摘要和当前缓冲区
        history_parts = []
        if self._summary:
            history_parts.append(f"[历史摘要]\n{self._summary}")
        if self._pending_messages:
            history_parts.append(f"[最近对话]\n{self.buffer_as_str}")
        
        history = "\n\n".join(history_parts)
        
        return MemoryVariables(
            history=history,
            summary=self._summary,
            messages=copy.deepcopy(self._pending_messages)
        )
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """保存对话上下文"""
        input_str = inputs.get(self.input_key, "")
        output_str = outputs.get(self.output_key, "")
        
        if input_str:
            msg = Message(role="user", content=input_str)
            self.chat_history.add_message(msg)
            self._pending_messages.append(msg)
        
        if output_str:
            msg = Message(role="assistant", content=output_str)
            self.chat_history.add_message(msg)
            self._pending_messages.append(msg)
        
        # 检查是否需要生成摘要
        if len(self._pending_messages) >= self.max_buffer_size:
            self._generate_summary()
    
    def _generate_summary(self) -> None:
        """生成摘要"""
        if not self._pending_messages:
            return
        
        new_lines = self.format_messages(self._pending_messages)
        
        if self.llm:
            # 使用LLM生成摘要
            prompt = self.summary_prompt.format(
                existing_summary=self._summary or "无",
                new_lines=new_lines
            )
            
            try:
                response = self.llm.invoke(prompt)
                self._summary = response.content
                if self.verbose:
                    print(f"[Memory] 生成摘要: {self._summary[:100]}...")
            except Exception as e:
                if self.verbose:
                    print(f"[Memory] 摘要生成失败: {e}")
                # 降级：使用简单截断
                self._simple_summarize(new_lines)
        else:
            # 无LLM时使用简单截断
            self._simple_summarize(new_lines)
        
        # 清空待处理消息
        self._pending_messages = []
    
    def _simple_summarize(self, new_lines: str) -> None:
        """简单摘要（无LLM时的降级方案）"""
        # 保留最近的对话作为摘要
        if self._summary:
            # 截断旧摘要
            max_summary_len = 500
            if len(self._summary) > max_summary_len:
                self._summary = self._summary[-max_summary_len:] + "..."
            self._summary = f"{self._summary}\n---\n{new_lines[-500:]}"
        else:
            self._summary = new_lines[-500:]
    
    def force_summarize(self) -> None:
        """强制生成摘要（不等待缓冲区满）"""
        self._generate_summary()
    
    def clear(self) -> None:
        """清空记忆"""
        self.chat_history.clear()
        self._summary = ""
        self._pending_messages = []
    
    def __repr__(self) -> str:
        return f"ConversationSummaryMemory(summary_len={len(self._summary)}, buffer={len(self._pending_messages)})"


class ConversationSummaryBufferMemory(BaseMemory):
    """摘要+缓冲区混合记忆
    
    结合摘要和滑动窗口，保留最近对话的同时维护历史摘要
    
    Attributes:
        llm: 用于生成摘要的语言模型
        max_token_limit: 最大token限制
        moving_summary_buffer: 摘要缓冲区
    """
    
    def __init__(
        self,
        llm: Optional["BaseChatModel"] = None,
        max_token_limit: int = 2000,
        token_counter: Optional[Callable[[str], int]] = None,
        summary_prompt: str = DEFAULT_SUMMARY_PROMPT,
        chat_history: Optional[BaseChatMessageHistory] = None,
        memory_key: str = "history",
        input_key: str = "input",
        output_key: str = "output",
        return_messages: bool = False,
        human_prefix: str = "Human",
        ai_prefix: str = "AI",
        verbose: bool = False
    ):
        """初始化摘要缓冲区记忆
        
        Args:
            llm: 用于生成摘要的语言模型
            max_token_limit: 缓冲区最大token数量
            token_counter: token计数函数
            summary_prompt: 摘要提示词模板
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
        self.llm = llm
        self.max_token_limit = max_token_limit
        self.token_counter = token_counter or (lambda x: len(x) // 3 + 1)
        self.summary_prompt = summary_prompt
        self.chat_history = chat_history or InMemoryChatMessageHistory()
        self.input_key = input_key
        self.output_key = output_key
        self._summary: str = ""
    
    @property
    def memory_variables(self) -> List[str]:
        """返回记忆变量名列表"""
        return [self.memory_key, "summary"]
    
    @property
    def summary(self) -> str:
        """获取当前摘要"""
        return self._summary
    
    @property
    def buffer(self) -> List[Message]:
        """获取在token限制内的消息"""
        messages = self.chat_history.messages
        if not messages:
            return []
        
        # 预留摘要的token空间
        summary_tokens = self.token_counter(self._summary) if self._summary else 0
        available_tokens = self.max_token_limit - summary_tokens
        
        # 从最新消息开始选择
        selected = []
        current_tokens = 0
        
        for msg in reversed(messages):
            msg_tokens = self.token_counter(msg.content)
            if current_tokens + msg_tokens > available_tokens:
                break
            selected.insert(0, msg)
            current_tokens += msg_tokens
        
        return selected
    
    @property
    def buffer_as_str(self) -> str:
        """获取格式化的缓冲区字符串"""
        return self.format_messages(self.buffer)
    
    def load_memory_variables(self, inputs: Optional[Dict[str, Any]] = None) -> MemoryVariables:
        """加载记忆变量"""
        history_parts = []
        if self._summary:
            history_parts.append(f"[历史摘要]\n{self._summary}")
        
        buffer_str = self.buffer_as_str
        if buffer_str:
            history_parts.append(f"[最近对话]\n{buffer_str}")
        
        history = "\n\n".join(history_parts)
        
        return MemoryVariables(
            history=history,
            summary=self._summary,
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
        
        # 检查是否需要修剪和生成摘要
        self._prune_buffer()
    
    def _prune_buffer(self) -> None:
        """修剪缓冲区，将超出的部分摘要化"""
        messages = self.chat_history.messages
        if not messages:
            return
        
        # 计算当前总token
        total_tokens = sum(self.token_counter(m.content) for m in messages)
        summary_tokens = self.token_counter(self._summary) if self._summary else 0
        
        if total_tokens + summary_tokens <= self.max_token_limit:
            return
        
        # 需要摘要的消息
        buffer = self.buffer
        to_summarize = messages[:-len(buffer)] if buffer else messages
        
        if to_summarize and self.llm:
            # 生成这些消息的摘要
            new_lines = self.format_messages(to_summarize)
            prompt = self.summary_prompt.format(
                existing_summary=self._summary or "无",
                new_lines=new_lines
            )
            
            try:
                response = self.llm.invoke(prompt)
                self._summary = response.content
                if self.verbose:
                    print(f"[Memory] 更新摘要: {self._summary[:100]}...")
            except Exception as e:
                if self.verbose:
                    print(f"[Memory] 摘要更新失败: {e}")
    
    def clear(self) -> None:
        """清空记忆"""
        self.chat_history.clear()
        self._summary = ""
    
    def __repr__(self) -> str:
        return f"ConversationSummaryBufferMemory(buffer={len(self.buffer)}, summary_len={len(self._summary)})"
# genAI_main_end
