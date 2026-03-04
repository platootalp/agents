"""
工具装饰器
提供声明式的工具创建方式，简化工具定义流程
"""

# genAI_main_start
from typing import Optional, Callable, Type
from functools import wraps
from pydantic import BaseModel
from .tool import Tool, StructuredTool


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    args_schema: Optional[Type[BaseModel]] = None,
    return_direct: bool = False,
    verbose: bool = False
):
    """工具装饰器
    
    使用装饰器语法快速创建工具
    自动从函数中提取名称和文档字符串
    
    Args:
        name: 工具名称（未提供时使用函数名）
        description: 工具描述（未提供时使用函数文档字符串）
        args_schema: 参数模式（Pydantic模型类）
        return_direct: 是否直接返回结果
        verbose: 是否输出详细日志
    
    Returns:
        装饰后的工具实例
    
    Examples:
        >>> @tool(description="搜索互联网信息")
        ... def search(query: str) -> str:
        ...     return f"搜索结果: {query}"
        >>> 
        >>> result = search.run(query="Python")
        
        >>> @tool(name="web_search", return_direct=True)
        ... def my_search(query: str) -> str:
        ...     '''在网络上搜索信息'''
        ...     return f"搜索: {query}"
    """
    def decorator(func: Callable) -> Tool:
        tool_name = name or func.__name__
        tool_description = description or func.__doc__ or f"执行{tool_name}函数"
        
        # 如果有args_schema，创建StructuredTool
        if args_schema:
            return StructuredTool(
                name=tool_name,
                description=tool_description,
                func=func,
                args_schema=args_schema,
                return_direct=return_direct,
                verbose=verbose
            )
        
        # 否则创建普通Tool
        return Tool(
            name=tool_name,
            description=tool_description,
            func=func,
            args_schema=None,
            return_direct=return_direct,
            verbose=verbose
        )
    
    return decorator


def async_tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    args_schema: Optional[Type[BaseModel]] = None,
    return_direct: bool = False,
    verbose: bool = False
):
    """异步工具装饰器
    
    使用装饰器语法创建支持异步执行的工具
    
    Args:
        name: 工具名称（未提供时使用函数名）
        description: 工具描述（未提供时使用函数文档字符串）
        args_schema: 参数模式（Pydantic模型类）
        return_direct: 是否直接返回结果
        verbose: 是否输出详细日志
    
    Returns:
        装饰后的工具实例
    
    Examples:
        >>> import asyncio
        >>> 
        >>> @async_tool(description="异步搜索信息")
        ... async def async_search(query: str) -> str:
        ...     await asyncio.sleep(1)  # 模拟网络请求
        ...     return f"异步搜索结果: {query}"
        >>> 
        >>> # 异步调用
        >>> result = await async_search.arun(query="Python")
    """
    def decorator(func: Callable) -> Tool:
        tool_name = name or func.__name__
        tool_description = description or func.__doc__ or f"执行{tool_name}函数"
        
        # 创建同步包装函数（用于同步调用场景）
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            return loop.run_until_complete(func(*args, **kwargs))
        
        # 如果有args_schema，创建StructuredTool
        if args_schema:
            return StructuredTool(
                name=tool_name,
                description=tool_description,
                func=sync_wrapper,
                args_schema=args_schema,
                return_direct=return_direct,
                verbose=verbose,
                coroutine=func
            )
        
        # 否则创建普通Tool
        return Tool(
            name=tool_name,
            description=tool_description,
            func=sync_wrapper,
            args_schema=None,
            return_direct=return_direct,
            verbose=verbose,
            coroutine=func
        )
    
    return decorator


def structured_tool(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    return_direct: bool = False,
    verbose: bool = False
):
    """结构化工具装饰器
    
    装饰器版本的StructuredTool，要求函数的第一个参数必须是Pydantic模型
    
    Args:
        func: 被装饰的函数（自动传入）
        name: 工具名称（未提供时使用函数名）
        description: 工具描述（未提供时使用函数文档字符串）
        return_direct: 是否直接返回结果
        verbose: 是否输出详细日志
    
    Returns:
        装饰后的结构化工具实例
    
    Examples:
        >>> from pydantic import BaseModel, Field
        >>> 
        >>> class SearchInput(BaseModel):
        ...     query: str = Field(description="搜索查询")
        ...     max_results: int = Field(default=10, description="最大结果数")
        >>> 
        >>> @structured_tool
        ... def search(input: SearchInput) -> str:
        ...     '''搜索互联网信息'''
        ...     return f"搜索'{input.query}'，返回{input.max_results}条结果"
        >>> 
        >>> result = search.run(query="Python", max_results=5)
    """
    def decorator(f: Callable) -> StructuredTool:
        tool_name = name or f.__name__
        tool_description = description or f.__doc__ or f"执行{tool_name}函数"
        
        # 尝试从函数签名中提取参数模式
        import inspect
        sig = inspect.signature(f)
        
        # 假设第一个参数是Pydantic模型
        params = list(sig.parameters.values())
        if not params:
            raise ValueError(f"函数{f.__name__}必须至少有一个参数（Pydantic模型）")
        
        first_param = params[0]
        args_schema = first_param.annotation
        
        if not (isinstance(args_schema, type) and issubclass(args_schema, BaseModel)):
            raise ValueError(
                f"函数{f.__name__}的第一个参数必须标注为Pydantic BaseModel类型"
            )
        
        # 创建包装函数，将关键字参数转换为模型实例
        @wraps(f)
        def wrapper(**kwargs):
            model_instance = args_schema(**kwargs)
            return f(model_instance)
        
        return StructuredTool(
            name=tool_name,
            description=tool_description,
            func=wrapper,
            args_schema=args_schema,
            return_direct=return_direct,
            verbose=verbose
        )
    
    # 支持无参数调用：@structured_tool
    if func is not None:
        return decorator(func)
    
    # 支持带参数调用：@structured_tool(name="...")
    return decorator
# genAI_main_end

