"""LLM Provider for OpenAI with tool support and streaming."""

from collections.abc import AsyncGenerator
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from .tools import Tool


class Message(BaseModel):
    """A chat message."""

    role: str = Field(..., description="Role: system, user, assistant, or tool")
    content: str | None = Field(None, description="Message content")
    tool_calls: list[dict] | None = Field(None, description="Tool calls from assistant")
    tool_call_id: str | None = Field(None, description="Tool call ID for tool responses")
    name: str | None = Field(None, description="Tool name for tool responses")

    def to_dict(self) -> dict[str, Any]:
        """Convert to OpenAI API format."""
        result: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            result["content"] = self.content
        if self.tool_calls is not None:
            result["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            result["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            result["name"] = self.name
        return result


class LLMResponse(BaseModel):
    """Response from LLM."""

    content: str | None = Field(None, description="Text content")
    tool_calls: list[dict] = Field(default_factory=list, description="Tool calls to execute")
    finish_reason: str = Field("stop", description="Why generation stopped")


class OpenAILLM:
    """OpenAI LLM provider with streaming and tool support."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.default_model = "Kimi-K2.5"

    async def chat_stream(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        model: str | None = None,
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        """Stream chat completion with real-time output.

        Yields text chunks as they arrive. For tool calls, yields
        special markers that the caller can handle.
        """
        model = model or self.default_model
        openai_messages = [m.to_dict() for m in messages]

        tools_param = None
        if tools:
            tools_param = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in tools
            ]

        stream = await self.client.chat.completions.create(
            model=model,
            messages=openai_messages,
            tools=tools_param,
            tool_choice="auto" if tools else None,
            temperature=temperature,
            stream=True,
        )

        async for chunk in stream:
            # Safety check: some chunks have empty choices
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # Handle content streaming
            if delta.content:
                yield delta.content

            # Handle tool calls - yield special marker
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    if tc.function and tc.function.name:
                        yield f"__TOOL_CALL_START__:{tc.function.name}"
                    if tc.function and tc.function.arguments:
                        yield f"__TOOL_CALL_ARGS__:{tc.function.arguments}"

    async def chat(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        model: str | None = None,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Non-streaming chat completion with optional tool calling."""
        model = model or self.default_model
        openai_messages = [m.to_dict() for m in messages]

        tools_param = None
        if tools:
            tools_param = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in tools
            ]

        response = await self.client.chat.completions.create(
            model=model,
            messages=openai_messages,
            tools=tools_param,
            tool_choice="auto" if tools else None,
            temperature=temperature,
        )

        choice = response.choices[0]
        message = choice.message

        content = message.content
        tool_calls = []

        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
        )
