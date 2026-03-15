"""Tool system for function calling."""

import asyncio
import inspect
import json
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Result of a tool execution."""

    success: bool = Field(True, description="Whether execution succeeded")
    result: Any = Field(None, description="The result data")
    error: str | None = Field(None, description="Error message if failed")

    def to_string(self) -> str:
        """Convert result to string for LLM."""
        if not self.success:
            return f"Error: {self.error}"
        if isinstance(self.result, (dict, list)):
            return json.dumps(self.result, ensure_ascii=False, indent=2)
        return str(self.result)


class Tool(BaseModel):
    """A tool that can be called by the LLM."""

    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="JSON schema for parameters"
    )
    func: Callable | None = Field(None, exclude=True, description="The function to execute")

    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given arguments."""
        if self.func is None:
            return ToolResult(success=False, error="Tool has no executable function")

        try:
            if asyncio.iscoroutinefunction(self.func):
                result = await self.func(**kwargs)
            else:
                result = self.func(**kwargs)
            return ToolResult(success=True, result=result)
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    @classmethod
    def from_function(
        cls,
        func: Callable,
        name: str | None = None,
        description: str | None = None,
    ) -> "Tool":
        """Create a Tool from a Python function."""
        tool_name = name or func.__name__
        tool_description = description or func.__doc__ or f"Execute {tool_name}"

        # Inspect function signature for parameters
        sig = inspect.signature(func)
        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            # Skip self/cls for methods
            if param_name in ("self", "cls"):
                continue

            param_type = param.annotation
            if param_type == inspect.Parameter.empty:
                json_type = "string"
            elif param_type in (str,):
                json_type = "string"
            elif param_type in (int, float):
                json_type = "number"
            elif param_type is bool:
                json_type = "boolean"
            elif param_type in (list,):
                json_type = "array"
            elif param_type in (dict,):
                json_type = "object"
            else:
                json_type = "string"

            properties[param_name] = {"type": json_type}
            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        parameters = {
            "type": "object",
            "properties": properties,
        }
        if required:
            parameters["required"] = required

        return cls(
            name=tool_name,
            description=tool_description,
            parameters=parameters,
            func=func,
        )


class ToolManager:
    """Manages available tools."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def register_function(
        self,
        func: Callable,
        name: str | None = None,
        description: str | None = None,
    ) -> Tool:
        """Register a function as a tool."""
        tool = Tool.from_function(func, name, description)
        self.register(tool)
        return tool

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        """List all registered tools."""
        return list(self._tools.values())

    def clear(self) -> None:
        """Clear all tools."""
        self._tools.clear()

    async def execute_tool_call(self, tool_call: dict) -> ToolResult:
        """Execute a tool call from LLM."""
        function_data = tool_call.get("function", {})
        tool_name = function_data.get("name", "")
        arguments_str = function_data.get("arguments", "{}")

        tool = self.get(tool_name)
        if not tool:
            return ToolResult(success=False, error=f"Tool '{tool_name}' not found")

        try:
            arguments = json.loads(arguments_str) if arguments_str else {}
        except json.JSONDecodeError:
            return ToolResult(success=False, error=f"Invalid arguments JSON: {arguments_str}")

        return await tool.execute(**arguments)
