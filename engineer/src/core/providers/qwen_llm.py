"""
通义千问 LLM提供商实现
支持阿里云通义千问系列LLM的调用，包括Qwen-Turbo、Qwen-Plus、Qwen-Max等
使用OpenAI兼容API接口
"""

import time
from typing import List, Union, Generator

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# genAI_main_start */
from ..language_models import (
    BaseChatModel, ChatModelConfig, BaseMessage,
    HumanMessage, AIMessage, ChatResult
)
from typing import Iterator, Optional, Any
# genAI_main_end */


# genAI_main_start */
class QwenLLM(BaseChatModel):
    """通义千问 LLM提供商
    
    支持阿里云通义千问系列LLM的调用，包括Qwen-Turbo、Qwen-Plus、Qwen-Max等
    使用OpenAI兼容API接口
    
    Attributes:
        client: OpenAI客户端实例（用于调用通义千问API）
        config: LLM配置对象
    """

    def __init__(self, config: ChatModelConfig):
        """初始化通义千问LLM实例
        
        Args:
            config: LLM配置对象，必须包含api_key
        
        Raises:
            RuntimeError: 未安装openai库
        """
        if OpenAI is None:
            raise RuntimeError("请安装openai库: pip install openai")
        
        super().__init__(config)
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
            timeout=config.timeout,
            max_retries=config.max_retries
        )
# genAI_main_end */

    # genAI_main_start */
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> ChatResult:
        """生成响应（内部方法）
        
        Args:
            messages: 消息列表
            stop: 停止词列表
            **kwargs: 额外参数，如temperature、max_tokens等
        
        Returns:
            ChatResult对象，包含响应内容和使用信息
        
        Raises:
            RuntimeError: API调用失败
        """
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=self._convert_messages_to_dict(messages),
                temperature=kwargs.get("temperature", self.config.temperature),
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                top_p=kwargs.get("top_p", self.config.top_p),
                stop=stop,
                **kwargs
            )

            tokens = response.usage.total_tokens if response.usage else 0

            return ChatResult(
                message=AIMessage(content=response.choices[0].message.content or ""),
                generation_info={
                    "model": response.model,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                        "total_tokens": tokens
                    },
                    "finish_reason": response.choices[0].finish_reason
                }
            )
        except Exception as e:
            raise RuntimeError(f"通义千问API调用失败: {str(e)}")
    # genAI_main_end */

    # genAI_main_start */
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> Iterator[str]:
        """流式生成响应（内部方法）
        
        Args:
            messages: 消息列表
            stop: 停止词列表
            **kwargs: 额外参数
        
        Yields:
            str: 流式响应的文本片段
        
        Raises:
            RuntimeError: 流式调用失败
        """
        try:
            stream = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=self._convert_messages_to_dict(messages),
                temperature=kwargs.get("temperature", self.config.temperature),
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                top_p=kwargs.get("top_p", self.config.top_p),
                stop=stop,
                stream=True,
                **kwargs
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            raise RuntimeError(f"通义千问流式调用失败: {str(e)}")
    # genAI_main_end */

    # genAI_main_start */
    def _convert_messages_to_dict(self, messages: List[BaseMessage]) -> List[dict]:
        """将消息列表转换为字典格式
        
        Args:
            messages: 消息列表
        
        Returns:
            包含role和content的字典列表
        """
        return [m.to_dict() for m in messages]
    # genAI_main_end */
