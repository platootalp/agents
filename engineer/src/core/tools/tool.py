"""
通用工具类实现
提供灵活的工具创建方式，支持函数式和结构化工具
"""

# genAI_main_start
from typing import Optional, Callable, Any, Type, Dict
from pydantic import BaseModel
from .base_tool import BaseTool


class Tool(BaseTool):
    """通用工具类
    
    允许通过传入函数快速创建工具实例
    适合简单的工具场景，无需创建完整的工具类
    
    Attributes:
        func: 工具执行函数
        coroutine: 异步工具执行函数（可选）
    
    Examples:
        >>> def search(query: str) -> str:
        ...     return f"搜索结果: {query}"
        >>> 
        >>> search_tool = Tool(
        ...     name="search",
        ...     description="搜索互联网信息",
        ...     func=search
        ... )
        >>> result = search_tool.run(query="Python教程")
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        func: Callable,
        args_schema: Optional[Type[BaseModel]] = None,
        return_direct: bool = False,
        verbose: bool = False,
        coroutine: Optional[Callable] = None
    ):
        """初始化通用工具
        
        Args:
            name: 工具名称
            description: 工具描述
            func: 工具执行函数（同步）
            args_schema: 参数模式（Pydantic模型类）
            return_direct: 是否直接返回结果
            verbose: 是否输出详细日志
            coroutine: 异步工具执行函数（可选）
        """
        super().__init__(
            name=name,
            description=description,
            args_schema=args_schema,
            return_direct=return_direct,
            verbose=verbose
        )
        self.func = func
        self.coroutine = coroutine
    
    def _run(self, *args, **kwargs) -> Any:
        """执行工具函数
        
        Args:
            *args: 位置参数
            **kwargs: 关键字参数
        
        Returns:
            工具函数的返回值
        """
        return self.func(*args, **kwargs)
    
    async def _arun(self, *args, **kwargs) -> Any:
        """异步执行工具函数
        
        如果提供了coroutine，使用异步函数；否则在线程中执行同步函数
        
        Args:
            *args: 位置参数
            **kwargs: 关键字参数
        
        Returns:
            工具函数的返回值
        """
        if self.coroutine:
            return await self.coroutine(*args, **kwargs)
        return await super()._arun(*args, **kwargs)


class StructuredTool(BaseTool):
    """结构化工具类
    
    强制使用Pydantic模型进行参数验证的工具
    适合参数复杂、需要严格验证的场景
    
    Attributes:
        func: 工具执行函数
        coroutine: 异步工具执行函数（可选）
    
    Examples:
        >>> from pydantic import BaseModel, Field
        >>> 
        >>> class CalculatorInput(BaseModel):
        ...     expression: str = Field(description="数学表达式")
        >>> 
        >>> def calculate(expression: str) -> str:
        ...     return str(eval(expression))
        >>> 
        >>> calc_tool = StructuredTool(
        ...     name="calculator",
        ...     description="计算数学表达式",
        ...     func=calculate,
        ...     args_schema=CalculatorInput
        ... )
        >>> result = calc_tool.run(expression="2 + 2")
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        func: Callable,
        args_schema: Type[BaseModel],
        return_direct: bool = False,
        verbose: bool = False,
        coroutine: Optional[Callable] = None
    ):
        """初始化结构化工具
        
        Args:
            name: 工具名称
            description: 工具描述
            func: 工具执行函数（同步）
            args_schema: 参数模式（Pydantic模型类，必需）
            return_direct: 是否直接返回结果
            verbose: 是否输出详细日志
            coroutine: 异步工具执行函数（可选）
        
        Raises:
            ValueError: args_schema不是BaseModel子类
        """
        if not issubclass(args_schema, BaseModel):
            raise ValueError("args_schema必须是Pydantic BaseModel的子类")
        
        super().__init__(
            name=name,
            description=description,
            args_schema=args_schema,
            return_direct=return_direct,
            verbose=verbose
        )
        self.func = func
        self.coroutine = coroutine
    
    def _run(self, *args, **kwargs) -> Any:
        """执行工具函数
        
        Args:
            *args: 位置参数
            **kwargs: 关键字参数
        
        Returns:
            工具函数的返回值
        """
        return self.func(*args, **kwargs)
    
    async def _arun(self, *args, **kwargs) -> Any:
        """异步执行工具函数
        
        如果提供了coroutine，使用异步函数；否则在线程中执行同步函数
        
        Args:
            *args: 位置参数
            **kwargs: 关键字参数
        
        Returns:
            工具函数的返回值
        """
        if self.coroutine:
            return await self.coroutine(*args, **kwargs)
        return await super()._arun(*args, **kwargs)


def create_tool_from_function(
    func: Callable,
    name: Optional[str] = None,
    description: Optional[str] = None,
    args_schema: Optional[Type[BaseModel]] = None,
    return_direct: bool = False,
    verbose: bool = False
) -> Tool:
    """从函数创建工具
    
    便捷函数，从普通Python函数快速创建工具实例
    自动从函数中提取名称和文档字符串
    
    Args:
        func: 要包装的函数
        name: 工具名称（未提供时使用函数名）
        description: 工具描述（未提供时使用函数文档字符串）
        args_schema: 参数模式（Pydantic模型类）
        return_direct: 是否直接返回结果
        verbose: 是否输出详细日志
    
    Returns:
        Tool实例
    
    Examples:
        >>> def greet(name: str) -> str:
        ...     '''向用户问候'''
        ...     return f"你好, {name}!"
        >>> 
        >>> greet_tool = create_tool_from_function(greet)
        >>> result = greet_tool.run(name="小明")
    """
    tool_name = name or func.__name__
    tool_description = description or func.__doc__ or f"执行{tool_name}函数"
    
    return Tool(
        name=tool_name,
        description=tool_description,
        func=func,
        args_schema=args_schema,
        return_direct=return_direct,
        verbose=verbose
    )
# genAI_main_end

