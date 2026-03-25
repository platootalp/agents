from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Sequence
from pydantic import BaseModel, Field

from .config import ProviderConfig
from .message import Message


class GenerationResult(BaseModel):
    message: Message
    usage: Dict[str, Any] = Field(default_factory=dict)
    finish_reason: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None


class BaseProvider(ABC):
    def __init__(self, config: ProviderConfig):
        self.config = config

    @abstractmethod
    def generate(
        self,
        messages: Sequence[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> GenerationResult:
        pass

    @abstractmethod
    def stream(
        self,
        messages: Sequence[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        pass

    @abstractmethod
    async def agenerate(
        self,
        messages: Sequence[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> GenerationResult:
        pass

    @abstractmethod
    def astream(
        self,
        messages: Sequence[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        pass


class OpenAIProvider(BaseProvider):
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        try:
            from openai import OpenAI

            self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        except ImportError:
            raise ImportError("openai package required")

    def generate(
        self,
        messages: Sequence[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> GenerationResult:
        params = {
            "model": self.config.model,
            "messages": [m.to_openai_dict() for m in messages],
            "temperature": self.config.temperature,
        }
        if self.config.max_tokens:
            params["max_tokens"] = self.config.max_tokens
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**params)
        choice = response.choices[0]

        return GenerationResult(
            message=Message(
                role="assistant",
                content=choice.message.content or "",
                tool_calls=[t.model_dump() for t in choice.message.tool_calls]
                if choice.message.tool_calls
                else None,
            ),
            usage=response.usage.model_dump() if response.usage else {},
            finish_reason=choice.finish_reason,
            raw_response=response.model_dump(),
        )

    def stream(
        self,
        messages: Sequence[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        params = {
            "model": self.config.model,
            "messages": [m.to_openai_dict() for m in messages],
            "temperature": self.config.temperature,
            "stream": True,
        }
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"

        for chunk in self.client.chat.completions.create(**params):
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def agenerate(
        self,
        messages: Sequence[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> GenerationResult:
        import asyncio

        return await asyncio.to_thread(self.generate, messages, tools, **kwargs)

    async def astream(
        self,
        messages: Sequence[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        import asyncio

        for chunk in self.stream(messages, tools, **kwargs):
            yield chunk
            await asyncio.sleep(0)
