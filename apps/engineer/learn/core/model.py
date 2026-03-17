from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Sequence, Union

from .message import Message, UserMessage
from .providers import BaseProvider, OpenAIProvider, ProviderConfig


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

    def generate(
        self,
        messages: Union[str, Sequence[Message]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Message:
        if isinstance(messages, str):
            messages = [UserMessage(content=messages)]
        result = self.provider.generate(messages, tools, **kwargs)
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

    async def agenerate(
        self,
        messages: Union[str, Sequence[Message]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Message:
        if isinstance(messages, str):
            messages = [UserMessage(content=messages)]
        result = await self.provider.agenerate(messages, tools, **kwargs)
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
