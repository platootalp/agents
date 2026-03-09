"""Shared type definitions for AI applications."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class Role(StrEnum):
    """Message roles in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """A message in a conversation.

    Attributes:
        role: The role of the message sender
        content: The content of the message
        metadata: Optional metadata for the message
    """

    role: Role
    content: str
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "role": self.role.value,
            "content": self.content,
            **({"metadata": self.metadata} if self.metadata else {}),
        }


@dataclass
class TokenUsage:
    """Token usage information for an LLM call.

    Attributes:
        prompt_tokens: Number of tokens in the prompt
        completion_tokens: Number of tokens in the completion
        total_tokens: Total number of tokens used
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class ChatResult:
    """Result from a chat completion.

    Attributes:
        content: The generated content
        usage: Token usage information
        model: The model used for generation
        metadata: Additional metadata
    """

    content: str
    usage: TokenUsage | None = None
    model: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class ToolCall:
    """A tool call requested by the model.

    Attributes:
        id: Unique identifier for the tool call
        name: Name of the tool to call
        arguments: Arguments to pass to the tool
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    """Result from a tool execution.

    Attributes:
        tool_call_id: ID of the corresponding tool call
        content: Result content
        error: Error message if execution failed
    """

    tool_call_id: str
    content: str
    error: str | None = None
