"""
Anthropic Claude LLM提供商实现
支持Anthropic Claude系列LLM的调用，包括Claude-3、Claude-2等
支持标准API调用和流式响应
"""

import time
from typing import List, Union, Generator

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

# genAI_main_start */
from ..language_models import (
    BaseChatModel, ChatModelConfig, BaseMessage,
    HumanMessage, AIMessage, ChatResult
)
from typing import Iterator, Optional, Any
# genAI_main_end */


# genAI_main_start */
class AnthropicLLM(BaseChatModel):
    """Anthropic Claude LLM提供商
    
    支持Anthropic Claude系列LLM的调用，包括Claude-3、Claude-2等
    支持标准API调用和流式响应
    
    Attributes:
        client: Anthropic客户端实例
        config: LLM配置对象
    """

    def __init__(self, config: ChatModelConfig):
        """初始化Anthropic LLM实例
        
        Args:
            config: LLM配置对象，必须包含api_key
        
        Raises:
            RuntimeError: 未安装anthropic库
        """
        if Anthropic is None:
            raise RuntimeError("请安装anthropic库: pip install anthropic")

        super().__init__(config)
        self.client = Anthropic(
            api_key=config.api_key,
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
            response = self.client.messages.create(
                model=self.config.model_name,
                messages=self._convert_messages_to_dict(messages),
                temperature=kwargs.get("temperature", self.config.temperature),
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                top_p=kwargs.get("top_p", self.config.top_p),
                stop_sequences=stop,
                **kwargs
            )

            tokens = response.usage.input_tokens + response.usage.output_tokens

            return ChatResult(
                message=AIMessage(content=response.content[0].text),
                generation_info={
                    "model": response.model,
                    "usage": {
                        "prompt_tokens": response.usage.input_tokens,
                        "completion_tokens": response.usage.output_tokens,
                        "total_tokens": tokens
                    },
                    "finish_reason": response.stop_reason
                }
            )
        except Exception as e:
            raise RuntimeError(f"Anthropic API调用失败: {str(e)}")
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
            with self.client.messages.stream(
                    model=self.config.model_name,
                    messages=self._convert_messages_to_dict(messages),
                    temperature=kwargs.get("temperature", self.config.temperature),
                    max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                    stop_sequences=stop,
                    **kwargs
            ) as stream:
                for text in stream.text_stream:
                    yield text
        except Exception as e:
            raise RuntimeError(f"Anthropic流式调用失败: {str(e)}")
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
