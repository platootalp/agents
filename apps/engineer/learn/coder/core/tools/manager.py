"""
工具管理器模块

提供工具注册、查找、执行和管理功能
"""

import asyncio
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Type, Union, Tuple

from .base import BaseTool, Tool, ToolResult


class ToolManager:
    """
    工具管理器

    负责工具的注册、查找和管理
    """

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._categories: Dict[str, List[str]] = {}

    def register_tool(self, tool: BaseTool, category: Optional[str] = None) -> None:
        """
        注册工具

        Args:
            tool: 要注册的工具
            category: 工具分类（可选）

        Raises:
            ValueError: 如果工具名称已存在
        """
        if tool.name in self._tools:
            raise ValueError(f"工具 '{tool.name}' 已存在")

        self._tools[tool.name] = tool

        # 添加到分类
        if category:
            if category not in self._categories:
                self._categories[category] = []
            self._categories[category].append(tool.name)

    def register_tools(self, tools: List[BaseTool], category: Optional[str] = None) -> None:
        """
        批量注册工具

        Args:
            tools: 工具列表
            category: 工具分类（可选）
        """
        for tool in tools:
            self.register_tool(tool, category)

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """
        获取工具

        Args:
            name: 工具名称

        Returns:
            BaseTool: 工具实例，如果不存在则返回None
        """
        return self._tools.get(name)

    def has_tool(self, name: str) -> bool:
        """
        检查工具是否存在

        Args:
            name: 工具名称

        Returns:
            bool: 是否存在
        """
        return name in self._tools

    def remove_tool(self, name: str) -> bool:
        """
        移除工具

        Args:
            name: 工具名称

        Returns:
            bool: 是否成功移除
        """
        if name in self._tools:
            del self._tools[name]

            # 从分类中移除
            for category, tools in self._categories.items():
                if name in tools:
                    tools.remove(name)

            return True
        return False

    def list_tools(self) -> List[str]:
        """
        列出所有工具名称

        Returns:
            List[str]: 工具名称列表
        """
        return list(self._tools.keys())

    def get_tools_by_category(self, category: str) -> List[BaseTool]:
        """
        获取分类下的所有工具

        Args:
            category: 分类名称

        Returns:
            List[BaseTool]: 工具列表
        """
        tool_names = self._categories.get(category, [])
        return [self._tools[name] for name in tool_names if name in self._tools]

    def list_categories(self) -> List[str]:
        """
        列出所有分类

        Returns:
            List[str]: 分类列表
        """
        return list(self._categories.keys())

    def get_all_tools(self) -> Dict[str, BaseTool]:
        """
        获取所有工具

        Returns:
            Dict[str, BaseTool]: 工具字典
        """
        return self._tools.copy()

    def run_tool(self, name: str, **kwargs) -> ToolResult:
        """
        执行工具（同步）

        Args:
            name: 工具名称
            **kwargs: 工具参数

        Returns:
            ToolResult: 执行结果
        """
        tool = self.get_tool(name)
        if not tool:
            return ToolResult.error_result(f"工具 '{name}' 不存在")

        return tool.run(**kwargs)

    async def arun_tool(self, name: str, **kwargs) -> ToolResult:
        """
        执行工具（异步）

        Args:
            name: 工具名称
            **kwargs: 工具参数

        Returns:
            ToolResult: 执行结果
        """
        tool = self.get_tool(name)
        if not tool:
            return ToolResult.error_result(f"工具 '{name}' 不存在")

        return await tool.arun(**kwargs)

    def get_openai_tools(self) -> List[Dict[str, Any]]:
        """
        获取所有工具的OpenAI格式

        Returns:
            List[Dict[str, Any]]: OpenAI工具格式列表
        """
        return [tool.to_openai_tool() for tool in self._tools.values()]

    def get_anthropic_tools(self) -> List[Dict[str, Any]]:
        """
        获取所有工具的Anthropic格式

        Returns:
            List[Dict[str, Any]]: Anthropic工具格式列表
        """
        return [tool.to_anthropic_tool() for tool in self._tools.values()]

    def clear(self) -> None:
        """清空所有工具"""
        self._tools.clear()
        self._categories.clear()


class ToolExecutor:
    """
    工具执行器

    提供高级工具执行功能，支持工具链、并行执行等
    """

    def __init__(self, manager: Optional[ToolManager] = None):
        self.manager = manager or ToolManager()

    def execute_tool_chain(
        self,
        tool_calls: List[Dict[str, Any]],
        stop_on_error: bool = True,
        callback: Optional[Callable[[int, ToolResult], None]] = None,
    ) -> List[ToolResult]:
        """
        顺序执行工具链

        Args:
            tool_calls: 工具调用列表，格式：[{"name": "tool1", "args": {...}}, ...]
            stop_on_error: 遇到错误时是否停止
            callback: 每步执行的回调函数

        Returns:
            List[ToolResult]: 执行结果列表
        """
        results = []

        for i, call in enumerate(tool_calls):
            name = call.get("name")
            args = call.get("args", {})

            result = self.manager.run_tool(name, **args)
            results.append(result)

            if callback:
                callback(i, result)

            # 如果出错且需要停止
            if not result.success and stop_on_error:
                break

        return results

    async def aexecute_tool_chain(
        self,
        tool_calls: List[Dict[str, Any]],
        stop_on_error: bool = True,
        callback: Optional[Callable[[int, ToolResult], None]] = None,
    ) -> List[ToolResult]:
        """
        异步顺序执行工具链

        Args:
            tool_calls: 工具调用列表
            stop_on_error: 遇到错误时是否停止
            callback: 每步执行的回调函数

        Returns:
            List[ToolResult]: 执行结果列表
        """
        results = []

        for i, call in enumerate(tool_calls):
            name = call.get("name")
            args = call.get("args", {})

            result = await self.manager.arun_tool(name, **args)
            results.append(result)

            if callback:
                callback(i, result)

            if not result.success and stop_on_error:
                break

        return results

    def execute_parallel(
        self,
        tool_calls: List[Dict[str, Any]],
        max_workers: int = 5,
        callback: Optional[Callable[[int, ToolResult], None]] = None,
    ) -> List[ToolResult]:
        """
        并行执行多个工具

        Args:
            tool_calls: 工具调用列表
            max_workers: 最大并发数
            callback: 每步执行的回调函数

        Returns:
            List[ToolResult]: 执行结果列表
        """
        results = [None] * len(tool_calls)

        def execute_single(index: int, call: Dict[str, Any]) -> None:
            name = call.get("name")
            args = call.get("args", {})
            result = self.manager.run_tool(name, **args)
            results[index] = result

            if callback:
                callback(index, result)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(execute_single, i, call): i for i, call in enumerate(tool_calls)
            }

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    index = futures[future]
                    results[index] = ToolResult.error_result(str(e))

        return results

    async def aexecute_parallel(
        self,
        tool_calls: List[Dict[str, Any]],
        callback: Optional[Callable[[int, ToolResult], None]] = None,
    ) -> List[ToolResult]:
        """
        异步并行执行多个工具

        Args:
            tool_calls: 工具调用列表
            callback: 每步执行的回调函数

        Returns:
            List[ToolResult]: 执行结果列表
        """

        async def execute_single(index: int, call: Dict[str, Any]) -> Tuple[int, ToolResult]:
            name = call.get("name")
            args = call.get("args", {})
            result = await self.manager.arun_tool(name, **args)

            if callback:
                callback(index, result)

            return index, result

        tasks = [execute_single(i, call) for i, call in enumerate(tool_calls)]
        completed = await asyncio.gather(*tasks, return_exceptions=True)

        results = [None] * len(tool_calls)
        for item in completed:
            if isinstance(item, Exception):
                # 找到第一个空位置放入错误结果
                for i, r in enumerate(results):
                    if r is None:
                        results[i] = ToolResult.error_result(str(item))
                        break
            else:
                index, result = item
                results[index] = result

        return results

    def execute_conditional(
        self,
        tool_call: Dict[str, Any],
        condition: Callable[[ToolResult], bool],
        success_tool: Optional[Dict[str, Any]] = None,
        failure_tool: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        条件执行工具

        Args:
            tool_call: 主工具调用
            condition: 条件函数
            success_tool: 成功时执行的工具
            failure_tool: 失败时执行的工具

        Returns:
            Dict: 包含所有执行结果
        """
        # 执行主工具
        name = tool_call.get("name")
        args = tool_call.get("args", {})
        main_result = self.manager.run_tool(name, **args)

        result = {
            "main": main_result,
            "success": None,
            "failure": None,
        }

        # 根据条件执行后续工具
        if condition(main_result):
            if success_tool:
                s_name = success_tool.get("name")
                s_args = success_tool.get("args", {})
                result["success"] = self.manager.run_tool(s_name, **s_args)
        else:
            if failure_tool:
                f_name = failure_tool.get("name")
                f_args = failure_tool.get("args", {})
                result["failure"] = self.manager.run_tool(f_name, **f_args)

        return result


class RetryableToolExecutor:
    """
    支持失败重试的工具执行器

    提供指数退避+抖动的重试机制，增强工具执行可靠性
    """

    def __init__(
        self,
        manager: Optional[ToolManager] = None,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    ):
        """
        初始化重试执行器

        Args:
            manager: 工具管理器，如未提供则创建新实例
            max_retries: 最大重试次数
            base_delay: 基础延迟（秒）
            max_delay: 最大延迟（秒）
            exponential_base: 指数退避基数
            jitter: 是否添加抖动
            retryable_exceptions: 可重试的异常类型元组，默认为所有Exception
        """
        self.manager = manager or ToolManager()
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions or (Exception,)

    def _calculate_delay(self, attempt: int) -> float:
        """
        计算重试延迟（指数退避 + 抖动）

        Args:
            attempt: 当前尝试次数（从0开始）

        Returns:
            float: 延迟秒数
        """
        # 指数退避
        delay = self.base_delay * (self.exponential_base**attempt)
        # 限制最大延迟
        delay = min(delay, self.max_delay)

        # 添加抖动（±25%）
        if self.jitter:
            jitter_factor = 0.75 + random.random() * 0.5
            delay *= jitter_factor

        return delay

    def execute_with_retry(
        self,
        tool_name: str,
        callback: Optional[Callable[[int, ToolResult], None]] = None,
        **kwargs,
    ) -> ToolResult:
        """
        同步执行工具，支持失败重试

        Args:
            tool_name: 工具名称
            callback: 每次尝试的回调函数，接收(attempt_index, result)
            **kwargs: 工具参数

        Returns:
            ToolResult: 最终结果（成功或最后一次失败）
        """
        last_result = None

        for attempt in range(self.max_retries + 1):
            try:
                result = self.manager.run_tool(tool_name, **kwargs)
                last_result = result

                if callback:
                    callback(attempt, result)

                if result.success:
                    return result

                # 如果不是最后一次尝试，等待后重试
                if attempt < self.max_retries:
                    delay = self._calculate_delay(attempt)
                    time.sleep(delay)

            except self.retryable_exceptions as e:
                error_result = ToolResult.error_result(str(e))
                last_result = error_result

                if callback:
                    callback(attempt, error_result)

                if attempt < self.max_retries:
                    delay = self._calculate_delay(attempt)
                    time.sleep(delay)

        # 所有重试都失败了
        if last_result and not last_result.success:
            return ToolResult(
                output=last_result.output,
                success=False,
                error=f"工具 '{tool_name}' 在 {self.max_retries} 次重试后仍失败: {last_result.error}",
                elapsed_ms=last_result.elapsed_ms,
            )

        return last_result or ToolResult.error_result("未知错误")

    async def aexecute_with_retry(
        self,
        tool_name: str,
        callback: Optional[Callable[[int, ToolResult], None]] = None,
        **kwargs,
    ) -> ToolResult:
        """
        异步执行工具，支持失败重试

        Args:
            tool_name: 工具名称
            callback: 每次尝试的回调函数，接收(attempt_index, result)
            **kwargs: 工具参数

        Returns:
            ToolResult: 最终结果（成功或最后一次失败）
        """
        last_result = None

        for attempt in range(self.max_retries + 1):
            try:
                result = await self.manager.arun_tool(tool_name, **kwargs)
                last_result = result

                if callback:
                    callback(attempt, result)

                if result.success:
                    return result

                # 如果不是最后一次尝试，等待后重试
                if attempt < self.max_retries:
                    delay = self._calculate_delay(attempt)
                    await asyncio.sleep(delay)

            except self.retryable_exceptions as e:
                error_result = ToolResult.error_result(str(e))
                last_result = error_result

                if callback:
                    callback(attempt, error_result)

                if attempt < self.max_retries:
                    delay = self._calculate_delay(attempt)
                    await asyncio.sleep(delay)

        # 所有重试都失败了
        if last_result and not last_result.success:
            return ToolResult(
                output=last_result.output,
                success=False,
                error=f"工具 '{tool_name}' 在 {self.max_retries} 次重试后仍失败: {last_result.error}",
                elapsed_ms=last_result.elapsed_ms,
            )

        return last_result or ToolResult.error_result("未知错误")

    def execute_chain_with_retry(
        self,
        tool_calls: List[Dict[str, Any]],
        stop_on_error: bool = True,
        per_tool_callback: Optional[Callable[[int, int, ToolResult], None]] = None,
    ) -> List[ToolResult]:
        """
        顺序执行工具链，每个工具支持重试

        Args:
            tool_calls: 工具调用列表，格式：[{"name": "tool1", "args": {...}}, ...]
            stop_on_error: 遇到错误时是否停止
            per_tool_callback: 回调函数，接收(tool_index, attempt_index, result)

        Returns:
            List[ToolResult]: 执行结果列表
        """
        results = []

        for tool_index, call in enumerate(tool_calls):
            name = call.get("name")
            args = call.get("args", {})

            def callback(attempt: int, result: ToolResult):
                if per_tool_callback:
                    per_tool_callback(tool_index, attempt, result)

            result = self.execute_with_retry(name, callback=callback, **args)
            results.append(result)

            if not result.success and stop_on_error:
                break

        return results

    async def aexecute_chain_with_retry(
        self,
        tool_calls: List[Dict[str, Any]],
        stop_on_error: bool = True,
        per_tool_callback: Optional[Callable[[int, int, ToolResult], None]] = None,
    ) -> List[ToolResult]:
        """
        异步顺序执行工具链，每个工具支持重试

        Args:
            tool_calls: 工具调用列表
            stop_on_error: 遇到错误时是否停止
            per_tool_callback: 回调函数，接收(tool_index, attempt_index, result)

        Returns:
            List[ToolResult]: 执行结果列表
        """
        results = []

        for tool_index, call in enumerate(tool_calls):
            name = call.get("name")
            args = call.get("args", {})

            def callback(attempt: int, result: ToolResult):
                if per_tool_callback:
                    per_tool_callback(tool_index, attempt, result)

            result = await self.aexecute_with_retry(name, callback=callback, **args)
            results.append(result)

            if not result.success and stop_on_error:
                break

        return results
