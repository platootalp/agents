from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4
from pydantic import BaseModel, Field, field_validator


class Message(BaseModel):
    role: str = Field(..., description="system/user/assistant/tool")
    content: str = Field(default="")
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    message_id: str = Field(default_factory=lambda: str(uuid4()))

    @field_validator("role")
    @classmethod
    def check_role(cls, v: str) -> str:
        if v not in ("system", "user", "assistant", "tool"):
            raise ValueError(f"Invalid role: {v}")
        return v

    def to_openai_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"role": self.role}
        if self.content:
            result["content"] = self.content
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        return result


class UserMessage(Message):
    def __init__(self, content: str, **kwargs: Any):
        super().__init__(role="user", content=content, **kwargs)


class AssistantMessage(Message):
    def __init__(self, content: str = "", tool_calls: Optional[List[Dict]] = None, **kwargs: Any):
        super().__init__(role="assistant", content=content, tool_calls=tool_calls, **kwargs)


class SystemMessage(Message):
    def __init__(self, content: str, **kwargs: Any):
        super().__init__(role="system", content=content, **kwargs)


class ToolMessage(Message):
    def __init__(self, content: str, tool_call_id: str, **kwargs: Any):
        super().__init__(role="tool", content=content, tool_call_id=tool_call_id, **kwargs)
