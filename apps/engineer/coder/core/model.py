from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Sequence, Union

from .message import Message, UserMessage
from .providers import BaseProvider, GenerationResult, OpenAIProvider, ProviderConfig


class Model:
    def __init__(
        self,
        provider: Union[str, BaseProvider] = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ):
        if isinstance(provider, str):
            config = ProviderConfig(
                api_key=api_key,
                base_url=base_url,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if provider == "openai":
                self.provider = OpenAIProvider(config)
            else:
                raise ValueError(f"Unknown provider: {provider}")
        else:
            self.provider = provider

    @classmethod
    def from_env(cls, provider: str = "openai", **overrides) -> "Model":
        """从环境变量加载配置并创建 Model 实例

        这是推荐的初始化方式，会自动读取环境变量：
        - {PROVIDER}_API_KEY: API 密钥
        - {PROVIDER}_BASE_URL: 可选的基础 URL
        - {PROVIDER}_MODEL: 模型名称，默认 gpt-4o
        - {PROVIDER}_TEMPERATURE: 温度参数，默认 0.7
        - {PROVIDER}_MAX_TOKENS: 最大 token 数

        Args:
            provider: Provider 名称，如 "openai", "anthropic"
            **overrides: 覆盖环境变量的参数

        Returns:
            配置好的 Model 实例

        Example:
            # 从 OPENAI_API_KEY 环境变量加载
            model = Model.from_env("openai")

            # 覆盖特定参数
            model = Model.from_env("openai", model="gpt-4o-mini", temperature=0.5)
        """
        config = ProviderConfig.from_env(provider, **overrides)
        return cls(
            provider=provider,
            api_key=config.api_key,
            base_url=config.base_url,
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    def complete(
        self,
        messages: Union[str, Sequence[Message]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """生成完整响应，返回包含 usage、finish_reason 等元数据的结果

        这是底层方法，返回完整的 GenerationResult。如需仅获取消息内容，
        请使用 generate() 方法。

        Args:
            messages: 消息列表或单个字符串
            tools: 可选的工具定义
            **kwargs: 额外的生成参数

        Returns:
            GenerationResult 包含 message, usage, finish_reason, raw_response
        """
        if isinstance(messages, str):
            messages = [UserMessage(content=messages)]
        return self.provider.generate(messages, tools, **kwargs)

    def generate(
        self,
        messages: Union[str, Sequence[Message]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Message:
        """生成响应，仅返回消息内容（简化版）"""
        result = self.complete(messages, tools, **kwargs)
        return result.message

    def stream(
        self,
        messages: Union[str, Sequence[Message]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        if isinstance(messages, str):
            messages = [UserMessage(content=messages)]
        yield from self.provider.stream(messages, tools, **kwargs)

    async def acomplete(
        self,
        messages: Union[str, Sequence[Message]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """异步生成完整响应，返回包含元数据的结果"""
        if isinstance(messages, str):
            messages = [UserMessage(content=messages)]
        return await self.provider.agenerate(messages, tools, **kwargs)

    async def agenerate(
        self,
        messages: Union[str, Sequence[Message]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Message:
        """异步生成响应，仅返回消息内容（简化版）"""
        result = await self.acomplete(messages, tools, **kwargs)
        return result.message

    async def astream(
        self,
        messages: Union[str, Sequence[Message]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        if isinstance(messages, str):
            messages = [UserMessage(content=messages)]
        async for chunk in self.provider.astream(messages, tools, **kwargs):
            yield chunk
