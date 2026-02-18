"""
Ollama本地LLM提供商实现
支持通过Ollama在本地部署和运行开源大语言LLM
使用OpenAI兼容API接口与Ollama服务通信
支持列出和管理本地安装的Ollama LLM
"""

import time
import requests
from typing import List, Generator

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
class OllamaLLM(BaseChatModel):
    """Ollama本地LLM提供商
    
    支持通过Ollama在本地部署和运行开源大语言LLM
    使用OpenAI兼容API接口与Ollama服务通信
    支持列出和管理本地安装的Ollama LLM
    
    Attributes:
        client: OpenAI客户端实例（用于调用Ollama API）
        config: LLM配置对象
    """

    def __init__(self, config: ChatModelConfig):
        """初始化Ollama LLM实例
        
        连接到本地Ollama服务（默认为http://localhost:11434）
        
        Args:
            config: LLM配置对象，model_name为Ollama LLM名称
        
        Raises:
            RuntimeError: 未安装openai库
        """
        if OpenAI is None:
            raise RuntimeError("请安装openai库: pip install openai")
        
        super().__init__(config)
        self.client = OpenAI(
            api_key="ollama",
            base_url=config.base_url or "http://localhost:11434/v1",
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
                messages=[m.to_dict() for m in messages],
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
            raise RuntimeError(f"Ollama API调用失败: {str(e)}")
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
                messages=[m.to_dict() for m in messages],
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
            raise RuntimeError(f"Ollama流式调用失败: {str(e)}")
    # genAI_main_end */

    def list_models(self) -> List[str]:
        """列出所有可用的Ollama LLM
        
        通过Ollama API获取本地已安装的LLM列表
        
        Returns:
            List[str]: LLM名称列表
        
        Raises:
            RuntimeError: 获取LLM列表失败
        """
        try:
            response = requests.get(f"{self.client.base_url.replace('/v1', '')}/api/tags", timeout=10)
            response.raise_for_status()
            result = response.json()
            return [model['name'] for model in result.get('models', [])]
        except Exception as e:
            raise RuntimeError(f"获取Ollama LLM列表失败: {str(e)}")
