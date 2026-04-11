"""
Agent工具体系基础模块

提供工具基类、结果封装和装饰器支持
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union
import asyncio
import json
import time

from pydantic import BaseModel, Field


class ToolCallbackType(Enum):
    """工具回调类型"""

    ON_TOOL_START = "on_tool_start"
    ON_TOOL_END = "on_tool_end"
    ON_TOOL_ERROR = "on_tool_error"


@dataclass
class ToolResult:
    """标准化的工具执行结果"""

    output: Union[str, Dict[str, Any]] = ""
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0

    @classmethod
    def success_result(cls, output: Any, **metadata) -> "ToolResult":
        """创建成功结果"""
        return cls(output=output, success=True, metadata=metadata)

    @classmethod
    def error_result(cls, error: str, **metadata) -> "ToolResult":
        """创建错误结果"""
        return cls(output="", success=False, error=error, metadata=metadata)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "output": self.output,
            "success": self.success,
            "error": self.error,
            "metadata": self.metadata,
            "elapsed_ms": self.elapsed_ms,
        }

    def __str__(self) -> str:
        if self.success:
            return str(self.output)
        return f"Error: {self.error}"


class BaseTool(ABC):
    """
    工具基类

    所有工具的抽象基类，定义标准接口
    """

    def __init__(
        self,
        name: str,
        description: str,
        args_schema: Optional[Type[BaseModel]] = None,
        return_direct: bool = False,
    ):
        self.name = name
        self.description = description
        self.args_schema = args_schema
        self.return_direct = return_direct
        self._callbacks: Dict[ToolCallbackType, List[Callable]] = {
            ToolCallbackType.ON_TOOL_START: [],
            ToolCallbackType.ON_TOOL_END: [],
            ToolCallbackType.ON_TOOL_ERROR: [],
        }

    @abstractmethod
    def _run(self, *args, **kwargs) -> Any:
        """同步执行逻辑 - 子类必须实现"""
        pass

    async def _arun(self, *args, **kwargs) -> Any:
        """异步执行逻辑 - 默认使用线程池"""
        return await asyncio.to_thread(self._run, *args, **kwargs)

    def run(self, **kwargs) -> ToolResult:
        """
        执行工具（同步）

        Args:
            **kwargs: 工具参数

        Returns:
            ToolResult: 执行结果
        """
        start_time = time.time()

        # 触发开始回调
        self._trigger_callback(ToolCallbackType.ON_TOOL_START, kwargs)

        try:
            # 参数验证
            if self.args_schema:
                validated_args = self.args_schema(**kwargs)
                kwargs = validated_args.model_dump()

            # 执行工具
            result = self._run(**kwargs)

            # 封装结果
            elapsed_ms = (time.time() - start_time) * 1000
            tool_result = ToolResult(
                output=result,
                success=True,
                elapsed_ms=elapsed_ms,
            )

            # 触发结束回调
            self._trigger_callback(ToolCallbackType.ON_TOOL_END, tool_result)

            return tool_result

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            error_result = ToolResult(
                output="",
                success=False,
                error=str(e),
                elapsed_ms=elapsed_ms,
            )

            # 触发错误回调
            self._trigger_callback(ToolCallbackType.ON_TOOL_ERROR, e)

            return error_result

    async def arun(self, **kwargs) -> ToolResult:
        """
        执行工具（异步）

        Args:
            **kwargs: 工具参数

        Returns:
            ToolResult: 执行结果
        """
        start_time = time.time()

        # 触发开始回调
        self._trigger_callback(ToolCallbackType.ON_TOOL_START, kwargs)

        try:
            # 参数验证
            if self.args_schema:
                validated_args = self.args_schema(**kwargs)
                kwargs = validated_args.model_dump()

            # 异步执行
            result = await self._arun(**kwargs)

            # 封装结果
            elapsed_ms = (time.time() - start_time) * 1000
            tool_result = ToolResult(
                output=result,
                success=True,
                elapsed_ms=elapsed_ms,
            )

            # 触发结束回调
            self._trigger_callback(ToolCallbackType.ON_TOOL_END, tool_result)

            return tool_result

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            error_result = ToolResult(
                output="",
                success=False,
                error=str(e),
                elapsed_ms=elapsed_ms,
            )

            # 触发错误回调
            self._trigger_callback(ToolCallbackType.ON_TOOL_ERROR, e)

            return error_result

    def register_callback(self, callback_type: ToolCallbackType, callback: Callable):
        """注册回调函数"""
        self._callbacks[callback_type].append(callback)

    def _trigger_callback(self, callback_type: ToolCallbackType, data: Any):
        """触发回调"""
        for callback in self._callbacks[callback_type]:
            try:
                callback(data)
            except Exception:
                pass  # 回调错误不应影响工具执行

    def get_parameters_schema(self) -> Dict[str, Any]:
        """获取参数Schema"""
        if self.args_schema:
            return self.args_schema.model_json_schema()
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    def to_openai_tool(self) -> Dict[str, Any]:
        """转换为OpenAI工具格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.get_parameters_schema(),
            },
        }

    def to_anthropic_tool(self) -> Dict[str, Any]:
        """转换为Anthropic工具格式"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.get_parameters_schema(),
        }

    def __call__(self, **kwargs) -> ToolResult:
        """使工具可调用"""
        return self.run(**kwargs)


class Tool(BaseTool):
    """
    通用工具类

    基于函数的工具实现
    """

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable,
        args_schema: Optional[Type[BaseModel]] = None,
        return_direct: bool = False,
    ):
        super().__init__(name, description, args_schema, return_direct)
        self.func = func

    def _run(self, *args, **kwargs) -> Any:
        """执行包装的函数"""
        if args and not kwargs:
            # 如果只有位置参数，尝试作为单个参数传递
            return self.func(*args)
        return self.func(**kwargs)


class StructuredTool(BaseTool):
    """
    结构化工具类

    使用Pydantic模型进行参数验证的工具
    """

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable,
        args_schema: Type[BaseModel],
        return_direct: bool = False,
    ):
        super().__init__(name, description, args_schema, return_direct)
        self.func = func

    def _run(self, *args, **kwargs) -> Any:
        """执行并验证参数"""
        # 参数已通过args_schema验证
        if args and not kwargs:
            return self.func(*args)
        return self.func(**kwargs)


def tool(name: Optional[str] = None, description: Optional[str] = None):
    """
    工具装饰器

    将函数转换为工具的便捷装饰器

    Usage:
        @tool()
        def my_function(query: str) -> str:
            return f"Result: {query}"

        @tool(name="search", description="搜索工具")
        def search(query: str) -> str:
            return f"Search: {query}"
    """

    def decorator(func: Callable) -> Tool:
        tool_name = name or func.__name__
        tool_description = description or func.__doc__ or f"Tool: {tool_name}"

        return Tool(
            name=tool_name,
            description=tool_description,
            func=func,
        )

    return decorator


def structured_tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    args_schema: Optional[Type[BaseModel]] = None,
):
    """
    结构化工具装饰器

    使用Pydantic模型进行参数验证的工具装饰器

    Usage:
        class SearchInput(BaseModel):
            query: str
            max_results: int = 10

        @structured_tool(args_schema=SearchInput)
        def search(input: SearchInput) -> str:
            return f"Search: {input.query}"
    """

    def decorator(func: Callable) -> StructuredTool:
        tool_name = name or func.__name__
        tool_description = description or func.__doc__ or f"Tool: {tool_name}"

        return StructuredTool(
            name=tool_name,
            description=tool_description,
            func=func,
            args_schema=args_schema or BaseModel,
        )

    return decorator


def async_tool(name: Optional[str] = None, description: Optional[str] = None):
    """
    异步工具装饰器

    将异步函数转换为工具的便捷装饰器

    Usage:
        @async_tool()
        async def my_async_function(query: str) -> str:
            await asyncio.sleep(1)
            return f"Result: {query}"
    """

    def decorator(func: Callable) -> Tool:
        tool_name = name or func.__name__
        tool_description = description or func.__doc__ or f"Tool: {tool_name}"

        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)

        return Tool(
            name=tool_name,
            description=tool_description,
            func=wrapper,
        )

    return decorator
