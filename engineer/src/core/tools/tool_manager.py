"""
工具管理器
提供工具注册、查找、调用等管理功能
"""

# genAI_main_start
from typing import List, Dict, Optional, Any, Union
from .base_tool import BaseTool, ToolResult


class ToolManager:
    """工具管理器
    
    负责管理多个工具的注册、查找和调用
    提供统一的工具管理接口
    
    Attributes:
        tools: 工具字典，键为工具名称，值为工具实例
        verbose: 是否输出详细日志
    
    Examples:
        >>> manager = ToolManager()
        >>> manager.register_tool(calculator)
        >>> manager.register_tool(search_tool)
        >>> result = manager.run_tool("calculator", expression="2+2")
    """
    
    def __init__(self, tools: Optional[List[BaseTool]] = None, verbose: bool = False):
        """初始化工具管理器
        
        Args:
            tools: 初始工具列表（可选）
            verbose: 是否输出详细日志
        """
        self.tools: Dict[str, BaseTool] = {}
        self.verbose = verbose
        
        if tools:
            for tool in tools:
                self.register_tool(tool)
    
    def register_tool(self, tool: BaseTool):
        """注册工具
        
        将工具添加到管理器中
        
        Args:
            tool: 要注册的工具实例
        
        Raises:
            ValueError: 工具名称已存在
        """
        if tool.name in self.tools:
            raise ValueError(f"工具'{tool.name}'已存在")
        
        self.tools[tool.name] = tool
        
        if self.verbose:
            print(f"已注册工具: {tool.name}")
    
    def unregister_tool(self, name: str):
        """注销工具
        
        从管理器中移除指定工具
        
        Args:
            name: 工具名称
        
        Raises:
            ValueError: 工具不存在
        """
        if name not in self.tools:
            raise ValueError(f"工具'{name}'不存在")
        
        del self.tools[name]
        
        if self.verbose:
            print(f"已注销工具: {name}")
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """获取工具
        
        根据名称获取工具实例
        
        Args:
            name: 工具名称
        
        Returns:
            工具实例或None
        """
        return self.tools.get(name)
    
    def has_tool(self, name: str) -> bool:
        """检查工具是否存在
        
        Args:
            name: 工具名称
        
        Returns:
            工具是否存在
        """
        return name in self.tools
    
    def list_tools(self) -> List[str]:
        """列出所有工具名称
        
        Returns:
            工具名称列表
        """
        return list(self.tools.keys())
    
    def get_all_tools(self) -> List[BaseTool]:
        """获取所有工具实例
        
        Returns:
            工具实例列表
        """
        return list(self.tools.values())
    
    def run_tool(self, name: str, *args, **kwargs) -> ToolResult:
        """运行工具（同步）
        
        根据名称查找并执行工具
        
        Args:
            name: 工具名称
            *args: 位置参数
            **kwargs: 关键字参数
        
        Returns:
            ToolResult对象
        
        Raises:
            ValueError: 工具不存在
        """
        tool = self.get_tool(name)
        if tool is None:
            raise ValueError(f"工具'{name}'不存在")
        
        if self.verbose:
            print(f"正在运行工具: {name}")
        
        return tool.run(*args, **kwargs)
    
    async def arun_tool(self, name: str, *args, **kwargs) -> ToolResult:
        """运行工具（异步）
        
        根据名称查找并异步执行工具
        
        Args:
            name: 工具名称
            *args: 位置参数
            **kwargs: 关键字参数
        
        Returns:
            ToolResult对象
        
        Raises:
            ValueError: 工具不存在
        """
        tool = self.get_tool(name)
        if tool is None:
            raise ValueError(f"工具'{name}'不存在")
        
        if self.verbose:
            print(f"正在异步运行工具: {name}")
        
        return await tool.arun(*args, **kwargs)
    
    def to_openai_tools(self) -> List[Dict[str, Any]]:
        """转换为OpenAI工具格式
        
        将所有工具转换为OpenAI函数调用格式
        
        Returns:
            OpenAI工具格式的字典列表
        """
        return [tool.to_openai_tool() for tool in self.tools.values()]
    
    def to_anthropic_tools(self) -> List[Dict[str, Any]]:
        """转换为Anthropic工具格式
        
        将所有工具转换为Anthropic工具使用格式
        
        Returns:
            Anthropic工具格式的字典列表
        """
        return [tool.to_anthropic_tool() for tool in self.tools.values()]
    
    def get_tool_descriptions(self) -> str:
        """获取所有工具的描述信息
        
        Returns:
            格式化的工具描述字符串
        """
        if not self.tools:
            return "没有可用的工具"
        
        result = "=== 可用工具列表 ===\n\n"
        for i, tool in enumerate(self.tools.values(), 1):
            result += f"{i}. {tool.name}\n"
            result += f"   描述: {tool.description}\n"
            if tool.args_schema:
                result += f"   参数: {tool.args_schema.schema()['title']}\n"
            result += "\n"
        
        return result
    
    def clear(self):
        """清空所有工具"""
        self.tools.clear()
        
        if self.verbose:
            print("已清空所有工具")
    
    def __len__(self) -> int:
        """获取工具数量
        
        Returns:
            工具数量
        """
        return len(self.tools)
    
    def __contains__(self, name: str) -> bool:
        """检查工具是否存在（支持in操作符）
        
        Args:
            name: 工具名称
        
        Returns:
            工具是否存在
        """
        return name in self.tools
    
    def __repr__(self) -> str:
        """管理器的字符串表示
        
        Returns:
            格式化的管理器信息字符串
        """
        return f"ToolManager(tools={len(self.tools)})"


class ToolExecutor:
    """工具执行器
    
    提供更高级的工具执行功能，支持工具链、条件执行等
    
    Attributes:
        manager: 工具管理器实例
        verbose: 是否输出详细日志
    """
    
    def __init__(self, manager: ToolManager, verbose: bool = False):
        """初始化工具执行器
        
        Args:
            manager: 工具管理器实例
            verbose: 是否输出详细日志
        """
        self.manager = manager
        self.verbose = verbose
    
    def execute_tool_chain(
        self,
        tool_calls: List[Dict[str, Any]]
    ) -> List[ToolResult]:
        """执行工具链
        
        按顺序执行多个工具调用
        
        Args:
            tool_calls: 工具调用列表，每个元素包含name和args
                例: [{"name": "calculator", "args": {"expression": "2+2"}}]
        
        Returns:
            工具结果列表
        """
        results = []
        
        for i, call in enumerate(tool_calls):
            tool_name = call.get("name")
            tool_args = call.get("args", {})
            
            if self.verbose:
                print(f"执行工具链 [{i+1}/{len(tool_calls)}]: {tool_name}")
            
            try:
                result = self.manager.run_tool(tool_name, **tool_args)
                results.append(result)
                
                # 如果工具执行失败且不是最后一个，可以选择中断
                if not result.success:
                    if self.verbose:
                        print(f"工具{tool_name}执行失败，继续执行下一个工具")
            
            except Exception as e:
                error_result = ToolResult(
                    output="",
                    success=False,
                    error=str(e),
                    metadata={"tool_name": tool_name}
                )
                results.append(error_result)
        
        return results
    
    async def aexecute_tool_chain(
        self,
        tool_calls: List[Dict[str, Any]]
    ) -> List[ToolResult]:
        """异步执行工具链
        
        按顺序异步执行多个工具调用
        
        Args:
            tool_calls: 工具调用列表
        
        Returns:
            工具结果列表
        """
        results = []
        
        for i, call in enumerate(tool_calls):
            tool_name = call.get("name")
            tool_args = call.get("args", {})
            
            if self.verbose:
                print(f"异步执行工具链 [{i+1}/{len(tool_calls)}]: {tool_name}")
            
            try:
                result = await self.manager.arun_tool(tool_name, **tool_args)
                results.append(result)
                
                if not result.success:
                    if self.verbose:
                        print(f"工具{tool_name}执行失败，继续执行下一个工具")
            
            except Exception as e:
                error_result = ToolResult(
                    output="",
                    success=False,
                    error=str(e),
                    metadata={"tool_name": tool_name}
                )
                results.append(error_result)
        
        return results
    
    def execute_parallel(
        self,
        tool_calls: List[Dict[str, Any]]
    ) -> List[ToolResult]:
        """并行执行多个工具
        
        同时执行多个独立的工具调用
        
        Args:
            tool_calls: 工具调用列表
        
        Returns:
            工具结果列表
        """
        import concurrent.futures
        
        results = []
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for call in tool_calls:
                tool_name = call.get("name")
                tool_args = call.get("args", {})
                future = executor.submit(self.manager.run_tool, tool_name, **tool_args)
                futures.append(future)
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    error_result = ToolResult(
                        output="",
                        success=False,
                        error=str(e)
                    )
                    results.append(error_result)
        
        return results
# genAI_main_end

