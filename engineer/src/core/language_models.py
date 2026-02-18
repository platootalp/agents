"""
语言模型基础类和接口
对齐 LangChain 的命名规范和架构设计
"""

# genAI_main_start
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Iterator, AsyncIterator
from enum import Enum
from pydantic import BaseModel, Field
import time
import asyncio

# 从统一的消息模块导入
from .messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    FunctionMessage,
    ToolMessage,
    messages_to_dict,
    messages_from_dict,
    get_buffer_string
)


# ==================== 枚举类型 ====================

class ModelType(Enum):
    """模型类型枚举"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    QWEN = "qwen"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"


# ==================== 配置类 ====================

class ChatModelConfig(BaseModel):
    """聊天模型配置类 (使用 Pydantic BaseModel)
    
    用于配置模型的各种参数，对齐 LangChain 的配置方式
    
    Attributes:
        model_name: 模型名称
        model_type: 模型类型
        api_key: API 密钥
        base_url: API 基础 URL
        temperature: 温度参数，控制输出随机性，范围 0-2
        max_tokens: 最大生成 token 数
        top_p: 核采样参数
        top_k: Top-K 采样参数
        frequency_penalty: 频率惩罚
        presence_penalty: 存在惩罚
        timeout: 请求超时时间（秒）
        max_retries: 最大重试次数
        streaming: 是否启用流式输出
        callbacks: 回调函数列表
        metadata: 额外的元数据
    """
    
    model_name: str = Field(description="模型名称")
    model_type: ModelType = Field(description="模型类型")
    api_key: Optional[str] = Field(default=None, description="API 密钥")
    base_url: Optional[str] = Field(default=None, description="API 基础 URL")
    
    # 生成参数
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="温度参数")
    max_tokens: int = Field(default=2000, gt=0, description="最大生成 token 数")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="核采样参数")
    top_k: Optional[int] = Field(default=None, description="Top-K 采样参数")
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="频率惩罚")
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="存在惩罚")
    
    # 请求参数
    timeout: int = Field(default=60, gt=0, description="请求超时时间")
    max_retries: int = Field(default=3, ge=0, description="最大重试次数")
    
    # 高级参数
    streaming: bool = Field(default=False, description="是否启用流式输出")
    callbacks: List[Any] = Field(default_factory=list, description="回调函数列表")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="额外的元数据")
    
    class Config:
        arbitrary_types_allowed = True


# ==================== 响应类 ====================

class ChatResult(BaseModel):
    """聊天结果类 (对齐 LangChain 的 ChatResult)
    
    封装模型的响应信息
    
    Attributes:
        message: AI 消息对象
        generation_info: 生成信息（token 使用量、finish_reason 等）
        llm_output: 模型原始输出
    """
    
    message: AIMessage = Field(description="AI 消息对象")
    generation_info: Dict[str, Any] = Field(default_factory=dict, description="生成信息")
    llm_output: Optional[Dict[str, Any]] = Field(default=None, description="模型原始输出")
    
    class Config:
        arbitrary_types_allowed = True
    
    @property
    def content(self) -> str:
        """获取消息内容"""
        return self.message.content
    
    @property
    def usage(self) -> Dict[str, int]:
        """获取 token 使用情况"""
        return self.generation_info.get("usage", {})
    
    @property
    def finish_reason(self) -> Optional[str]:
        """获取结束原因"""
        return self.generation_info.get("finish_reason")


class LLMResult(BaseModel):
    """LLM 结果类 (对齐 LangChain 的 LLMResult)
    
    用于批量调用时的结果封装
    
    Attributes:
        generations: 生成结果列表
        llm_output: 模型原始输出
    """
    
    generations: List[List[ChatResult]] = Field(description="生成结果列表")
    llm_output: Optional[Dict[str, Any]] = Field(default=None, description="模型原始输出")
    
    class Config:
        arbitrary_types_allowed = True


# ==================== 性能指标类 ====================

class ModelMetrics(BaseModel):
    """模型性能指标类
    
    记录模型的性能统计数据
    """
    
    model_name: str = Field(description="模型名称")
    avg_latency: float = Field(default=0.0, description="平均延迟（秒）")
    avg_tokens_per_second: float = Field(default=0.0, description="平均生成速度")
    success_rate: float = Field(default=1.0, description="成功率")
    total_calls: int = Field(default=0, description="总调用次数")
    error_count: int = Field(default=0, description="错误次数")
    last_updated: float = Field(default_factory=time.time, description="最后更新时间")


# ==================== 基础聊天模型类 ====================

class BaseChatModel(ABC):
    """基础聊天模型类 (对齐 LangChain 的 BaseChatModel)
    
    所有聊天模型必须继承此类并实现抽象方法
    提供统一的接口用于模型调用、流式输出、批量处理等
    
    Attributes:
        config: 模型配置对象
        metrics: 性能指标对象
    """
    
    def __init__(self, config: ChatModelConfig):
        """初始化聊天模型实例
        
        Args:
            config: 模型配置对象
        """
        self.config = config
        self.metrics = ModelMetrics(model_name=config.model_name)
    
    # ==================== 抽象方法（子类必须实现）====================
    
    @abstractmethod
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> ChatResult:
        """生成响应（内部方法，子类必须实现）
        
        Args:
            messages: 消息列表
            stop: 停止词列表
            **kwargs: 额外参数
        
        Returns:
            ChatResult: 聊天结果对象
        """
        pass
    
    @abstractmethod
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> Iterator[str]:
        """流式生成响应（内部方法，子类必须实现）
        
        Args:
            messages: 消息列表
            stop: 停止词列表
            **kwargs: 额外参数
        
        Yields:
            str: 响应文本片段
        """
        pass
    
    # ==================== 公共接口方法（对齐 LangChain）====================
    
    def invoke(
        self,
        messages: Union[List[BaseMessage], str],
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> AIMessage:
        """调用模型生成响应 (对齐 LangChain 的 invoke 方法)
        
        Args:
            messages: 消息列表或单个字符串
            stop: 停止词列表
            **kwargs: 额外参数
        
        Returns:
            AIMessage: AI 消息对象
        
        Examples:
            >>> model = ChatOpenAI(...)
            >>> response = model.invoke("你好")
            >>> response = model.invoke([HumanMessage(content="你好")])
        """
        # 转换输入格式
        if isinstance(messages, str):
            messages = [HumanMessage(content=messages)]
        
        start_time = time.time()
        
        try:
            result = self._generate(messages, stop=stop, **kwargs)
            latency = time.time() - start_time
            
            # 更新性能指标
            tokens = result.usage.get("total_tokens", 0)
            self._update_metrics(latency, tokens, success=True)
            
            return result.message
        
        except Exception as e:
            latency = time.time() - start_time
            self._update_metrics(latency, 0, success=False)
            raise
    
    def stream(
        self,
        messages: Union[List[BaseMessage], str],
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> Iterator[str]:
        """流式调用模型生成响应 (对齐 LangChain 的 stream 方法)
        
        Args:
            messages: 消息列表或单个字符串
            stop: 停止词列表
            **kwargs: 额外参数
        
        Yields:
            str: 响应文本片段
        
        Examples:
            >>> model = ChatOpenAI(...)
            >>> for chunk in model.stream("讲个笑话"):
            ...     print(chunk, end='', flush=True)
        """
        # 转换输入格式
        if isinstance(messages, str):
            messages = [HumanMessage(content=messages)]
        
        yield from self._stream(messages, stop=stop, **kwargs)
    
    def batch(
        self,
        messages_list: List[Union[List[BaseMessage], str]],
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> List[AIMessage]:
        """批量调用模型 (对齐 LangChain 的 batch 方法)
        
        Args:
            messages_list: 消息列表的列表
            stop: 停止词列表
            **kwargs: 额外参数
        
        Returns:
            List[AIMessage]: AI 消息列表
        
        Examples:
            >>> model = ChatOpenAI(...)
            >>> responses = model.batch(["你好", "再见", "谢谢"])
        """
        results = []
        for messages in messages_list:
            result = self.invoke(messages, stop=stop, **kwargs)
            results.append(result)
        return results
    
    # ==================== 异步方法 ====================
    
    async def ainvoke(
        self,
        messages: Union[List[BaseMessage], str],
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> AIMessage:
        """异步调用模型 (对齐 LangChain 的 ainvoke 方法)
        
        Args:
            messages: 消息列表或单个字符串
            stop: 停止词列表
            **kwargs: 额外参数
        
        Returns:
            AIMessage: AI 消息对象
        """
        # 默认实现：在线程池中执行同步方法
        return await asyncio.to_thread(self.invoke, messages, stop=stop, **kwargs)
    
    async def astream(
        self,
        messages: Union[List[BaseMessage], str],
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> AsyncIterator[str]:
        """异步流式调用模型 (对齐 LangChain 的 astream 方法)
        
        Args:
            messages: 消息列表或单个字符串
            stop: 停止词列表
            **kwargs: 额外参数
        
        Yields:
            str: 响应文本片段
        """
        # 默认实现：在线程池中执行同步流式方法
        for chunk in self.stream(messages, stop=stop, **kwargs):
            yield chunk
            await asyncio.sleep(0)  # 让出控制权
    
    async def abatch(
        self,
        messages_list: List[Union[List[BaseMessage], str]],
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> List[AIMessage]:
        """异步批量调用模型 (对齐 LangChain 的 abatch 方法)
        
        Args:
            messages_list: 消息列表的列表
            stop: 停止词列表
            **kwargs: 额外参数
        
        Returns:
            List[AIMessage]: AI 消息列表
        """
        tasks = [
            self.ainvoke(messages, stop=stop, **kwargs)
            for messages in messages_list
        ]
        return await asyncio.gather(*tasks)
    
    # ==================== 向后兼容方法 ====================
    
    def predict(self, text: str, **kwargs: Any) -> str:
        """预测方法（向后兼容）
        
        Args:
            text: 输入文本
            **kwargs: 额外参数
        
        Returns:
            str: 模型输出文本
        """
        response = self.invoke(text, **kwargs)
        return response.content
    
    def predict_messages(
        self,
        messages: List[BaseMessage],
        **kwargs: Any
    ) -> AIMessage:
        """预测消息方法（向后兼容）
        
        Args:
            messages: 消息列表
            **kwargs: 额外参数
        
        Returns:
            AIMessage: AI 消息对象
        """
        return self.invoke(messages, **kwargs)
    
    # ==================== 性能指标方法 ====================
    
    def _update_metrics(self, latency: float, tokens: int, success: bool):
        """更新性能指标（内部方法）
        
        Args:
            latency: 调用延迟（秒）
            tokens: 使用的 token 数
            success: 是否成功
        """
        self.metrics.total_calls += 1
        if not success:
            self.metrics.error_count += 1
        
        total_calls = self.metrics.total_calls
        self.metrics.avg_latency = (
            (self.metrics.avg_latency * (total_calls - 1) + latency) / total_calls
        )
        
        if tokens > 0 and latency > 0:
            tps = tokens / latency
            self.metrics.avg_tokens_per_second = (
                (self.metrics.avg_tokens_per_second * (total_calls - 1) + tps) / total_calls
            )
        
        self.metrics.success_rate = (
            (total_calls - self.metrics.error_count) / total_calls
        )
        self.metrics.last_updated = time.time()
    
    def get_metrics(self) -> ModelMetrics:
        """获取性能指标
        
        Returns:
            ModelMetrics: 性能指标对象
        """
        return self.metrics
    
    def reset_metrics(self):
        """重置性能指标"""
        self.metrics = ModelMetrics(model_name=self.config.model_name)
    
    # ==================== 工具方法 ====================
    
    @property
    def _llm_type(self) -> str:
        """返回 LLM 类型（用于序列化和日志）"""
        return self.config.model_type.value
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """返回标识参数（用于缓存和日志）"""
        return {
            "model_name": self.config.model_name,
            "model_type": self.config.model_type.value,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
    
    def __repr__(self) -> str:
        """字符串表示"""
        return f"{self.__class__.__name__}(model_name='{self.config.model_name}')"


# genAI_main_end
