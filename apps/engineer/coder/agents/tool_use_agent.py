"""
ToolUseAgent - 基础工具使用Agent

架构设计:
=========

层级结构:
    BaseAgent (learn.core.agent)
        ↓ 继承
    ToolUseAgent (当前文件)
        ↓ 继承
    [McpAgent, SkillsUseAgent, TaskAgent, ...]

职责分离:
    - BaseAgent: 提供基础Agent接口 (name, description, model, max_steps)
    - ToolUseAgent: 整合BaseAgent + ToolManager，作为工具使用Agent的基类
    - ToolManager (coder.core.tools.manager): 工具注册和管理
    - ToolExecutor (coder.core.tools.manager): 工具执行

核心组件:
    1. ToolManager (来自 coder.core.tools.manager)
       - 工具注册、查找和管理
       - get_openai_tools(): 获取OpenAI格式的工具定义
       - run_tool()/arun_tool(): 同步/异步执行工具

    2. ToolExecutor (来自 coder.core.tools.manager)
       - execute_tool_chain(): 顺序执行工具链
       - execute_parallel(): 并行执行工具
       - execute_conditional(): 条件执行

    3. BaseTool (来自 coder.core.tools.base)
       - 所有工具的基类
       - 支持同步(_run)和异步(_arun)执行
       - 提供工具回调机制

    4. MessageBuilder (来自 learn.core.utils)
       - 构建标准消息格式 (system/user/assistant/tool)
       - 处理工具调用和响应的转换

设计模式:
    - 组合优于继承: ToolUseAgent 组合 ToolManager 和 ToolExecutor
    - 模板方法: 子类可重写 _build_system_prompt() 自定义行为
    - 依赖注入: 通过构造函数注入 model 和 tools
    - 向后兼容: 支持旧的 Tool 类（自动转换为 BaseTool）

扩展指南:
    子类化时可以:
    1. 重写 _build_system_prompt() - 自定义系统提示
    2. 重写 run() - 实现更复杂的执行逻辑
    3. 在 __init__ 中 self.tool_manager.register_tool() - 添加额外工具
    4. 重写 invoke() - 自定义调用流程

工作流程:
    User Input → _build_system_prompt() → LLM
                                        ↓
    Final Answer ← Tool Results ← Tool Execution ← Tool Calls
"""

import sys
import os

# 添加项目根目录到路径，支持从任意位置运行
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
)
sys.path.insert(0, project_root)

import json
from typing import Any, Dict, List, Optional, Union

# 新的工具系统导入
try:
    from apps.engineer.coder.core.tools.base import BaseTool, Tool, ToolResult
    from apps.engineer.coder.core.tools.manager import ToolManager, ToolExecutor
except ImportError:
    from apps.engineer.coder.core.tools.base import BaseTool, Tool, ToolResult
    from apps.engineer.coder.core.tools.manager import ToolManager, ToolExecutor

# Agent基础导入
try:
    from apps.engineer.coder.core.agent import BaseAgent
    from apps.engineer.coder.core.model import Model
    from apps.engineer.coder.core.utils import MessageBuilder
except ImportError:
    from apps.engineer.coder.core.agent import BaseAgent
    from apps.engineer.coder.core.model import Model
    from apps.engineer.coder.core.utils import MessageBuilder


class ToolUseAgent(BaseAgent):
    """
    基础工具使用Agent - 所有工具使用Agent的基类

    架构:
        继承 BaseAgent 提供: name, description, model, max_steps, run(), arun()
        组合 ToolManager 提供: 工具注册、查找、管理
        组合 ToolExecutor 提供: 工具链执行、并行执行

    职责:
        1. 管理工具列表 (通过 ToolManager)
        2. 提供标准系统提示词构建
        3. 封装同步/异步调用接口
        4. 工具调用执行和结果处理

    关键方法:
        _build_system_prompt(): 构建系统提示，子类可重写
        invoke(): 同步调用入口
        stream(): 流式调用入口
        ainvoke()/astream(): 异步接口
        call_tool(): 直接调用指定工具

    向后兼容:
        - 支持传入旧的 Tool 对象（自动包装为 BaseTool）
        - 支持传入新的 BaseTool 对象
        - 支持混合传入

    子类化示例:
        class MyAgent(ToolUseAgent):
            def _build_system_prompt(self) -> str:
                return "自定义提示词..."

            def __init__(self, ...):
                super().__init__(...)
                # 注册新工具
                self.tool_manager.register_tool(MyCustomTool())
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are a helpful assistant that can use tools to help answer user queries. "
        "Use the available tools when needed, and provide a clear final answer."
    )

    def __init__(
        self,
        name: str,
        description: str = "",
        model: Optional[Model] = None,
        tools: Optional[List[Union[BaseTool, Any]]] = None,
        max_steps: int = 10,
    ):
        """
        初始化 ToolUseAgent

        Args:
            name: Agent名称
            description: Agent描述
            model: LLM模型实例
            tools: 工具列表（支持BaseTool或旧版Tool）
            max_steps: 最大执行步数
        """
        BaseAgent.__init__(self, name, description, model, max_steps)

        # 初始化工具管理器和执行器
        self.tool_manager = ToolManager()
        self.tool_executor = ToolExecutor(self.tool_manager)

        # 消息历史
        self.message_history: List[Dict[str, Any]] = []

        # 注册工具（处理向后兼容）
        if tools:
            for tool in tools:
                self._register_tool_compat(tool)

    def _register_tool_compat(self, tool: Union[BaseTool, Any]) -> None:
        """
        注册工具（支持向后兼容）

        支持:
        - 新的 BaseTool 子类: 直接注册
        - 旧的 Tool 对象: 自动包装为 BaseTool
        - @tool 装饰器创建的工具: 直接注册
        """
        if isinstance(tool, BaseTool):
            # 新的工具类，直接注册
            self.tool_manager.register_tool(tool)
        elif hasattr(tool, "name") and hasattr(tool, "func"):
            # 旧的 Tool 对象或 @tool 装饰器创建的工具，包装为新的 BaseTool
            # 注意：新的 Tool 类使用 args_schema 而不是 parameters
            wrapped = Tool(
                name=tool.name,
                description=getattr(tool, "description", ""),
                func=tool.func,
                args_schema=getattr(tool, "args_schema", None),
            )
            self.tool_manager.register_tool(wrapped)
        elif callable(tool):
            # 尝试作为简单函数包装
            wrapped = Tool(
                name=getattr(tool, "__name__", "unknown_tool"),
                description=getattr(tool, "__doc__", ""),
                func=tool,
            )
            self.tool_manager.register_tool(wrapped)
        else:
            raise ValueError(f"不支持的工具类型: {type(tool)}")

    def _build_system_prompt(self) -> str:
        """构建系统提示词 - 子类可重写"""
        return self.DEFAULT_SYSTEM_PROMPT

    def _get_openai_tools(self) -> Optional[List[Dict[str, Any]]]:
        """
        获取 OpenAI 格式的工具定义

        Returns:
            OpenAI 工具格式列表，如果没有工具则返回 None
        """
        tools = self.tool_manager.get_openai_tools()
        return tools if tools else None

    def _init_conversation(self, input: str, system_prompt: str, reset: bool = False) -> None:
        """
        初始化对话历史

        Args:
            input: 用户输入
            system_prompt: 系统提示
            reset: 是否重置历史
        """
        if reset or not self.message_history:
            self.message_history = [
                MessageBuilder.build_system_message(system_prompt),
                MessageBuilder.build_user_message(input),
            ]
        else:
            self.message_history.append(MessageBuilder.build_user_message(input))

    def call_tool(self, tool_name: str, tool_args: Union[str, Dict[str, Any]]) -> str:
        """
        调用指定工具

        Args:
            tool_name: 工具名称
            tool_args: 工具参数（JSON字符串或字典）

        Returns:
            工具执行结果字符串
        """
        # 解析参数
        if isinstance(tool_args, str):
            try:
                args = json.loads(tool_args) if tool_args else {}
            except json.JSONDecodeError:
                args = {"query": tool_args}
        else:
            args = tool_args

        # 执行工具
        result = self.tool_manager.run_tool(tool_name, **args)

        if result.success:
            return str(result.output)
        else:
            return f"Error: {result.error}"

    async def acall_tool(self, tool_name: str, tool_args: Union[str, Dict[str, Any]]) -> str:
        """
        异步调用指定工具

        Args:
            tool_name: 工具名称
            tool_args: 工具参数（JSON字符串或字典）

        Returns:
            工具执行结果字符串
        """
        # 解析参数
        if isinstance(tool_args, str):
            try:
                args = json.loads(tool_args) if tool_args else {}
            except json.JSONDecodeError:
                args = {"query": tool_args}
        else:
            args = tool_args

        # 异步执行工具
        result = await self.tool_manager.arun_tool(tool_name, **args)

        if result.success:
            return str(result.output)
        else:
            return f"Error: {result.error}"

    def _execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        执行多个工具调用

        Args:
            tool_calls: 工具调用列表

        Returns:
            执行结果列表
        """
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            tool_id = tool_call["id"]
            tool_args = tool_call["function"]["arguments"]

            # 执行工具
            result_str = self.call_tool(tool_name, tool_args)

            results.append(
                {
                    "tool_call_id": tool_id,
                    "tool_name": tool_name,
                    "result": result_str,
                }
            )

        return results

    def invoke(self, input: str) -> str:
        """
        同步调用 - 使用标准对话循环

        Args:
            input: 用户输入

        Returns:
            Agent的最终响应
        """
        if not self.model:
            return "Error: No model configured."

        # 初始化对话
        self._init_conversation(input, self._build_system_prompt(), reset=True)

        for step in range(self.max_steps):
            # 获取工具定义
            openai_tools = self._get_openai_tools()

            # 调用LLM
            response = self.model.generate(self.message_history, tools=openai_tools)
            message = response.choices[0].message

            # 使用 MessageBuilder 构建助手消息
            tool_calls = MessageBuilder.convert_api_tool_calls(message.tool_calls)
            assistant_msg = MessageBuilder.build_assistant_message(
                message.content or "", tool_calls
            )
            self.message_history.append(assistant_msg)

            # 如果没有工具调用，返回回复内容
            if not tool_calls:
                return message.content.strip() if message.content else ""

            # 执行工具调用
            results = self._execute_tool_calls(tool_calls)

            # 添加工具响应到历史
            for result in results:
                tool_msg = MessageBuilder.build_tool_response_message(
                    result["tool_call_id"], result["result"]
                )
                self.message_history.append(tool_msg)

        return "Reached maximum steps without a final answer."

    async def ainvoke(self, input: str) -> str:
        """异步调用"""
        # 目前使用同步实现，子类可重写为真正的异步
        return self.invoke(input)

    def stream(self, input: str, reset: bool = False) -> str:
        """
        流式调用 - 实时打印输出

        Args:
            input: 用户输入
            reset: 是否重置对话历史

        Returns:
            Agent的最终响应
        """
        if not self.model:
            return "Error: No model configured."

        # 初始化对话
        self._init_conversation(input, self._build_system_prompt(), reset)

        if reset or len(self.message_history) <= 2:
            print("\n🆕 New Conversation\n")

        print(f"👤 User: {input}\n")

        for step in range(self.max_steps):
            # 获取工具定义
            openai_tools = self._get_openai_tools()

            # 流式调用LLM
            stream = self.model.stream(self.message_history, tools=openai_tools)

            # 累积内容和工具调用
            accumulated_content = ""
            accumulated_tool_calls: Dict[int, Dict[str, Any]] = {}

            # 追踪打印状态
            in_thinking = False
            in_content = False

            for chunk in stream:
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                # 处理思考内容
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    if not in_thinking:
                        print("\n💭 Thinking:\n", end="", flush=True)
                        in_thinking = True
                        in_content = False
                    print(delta.reasoning_content, end="", flush=True)

                # 处理内容块
                if delta.content:
                    if not in_content:
                        print("\n📝 Content:\n", end="", flush=True)
                        in_content = True
                        in_thinking = False
                    print(delta.content, end="", flush=True)
                    accumulated_content += delta.content

                # 处理工具调用块
                if delta.tool_calls:
                    accumulated_tool_calls.update(
                        MessageBuilder.accumulate_tool_calls(delta.tool_calls)
                    )

            tool_calls_list = list(accumulated_tool_calls.values())

            # 使用 MessageBuilder 构建助手消息
            assistant_msg = MessageBuilder.build_assistant_message(
                accumulated_content, tool_calls_list
            )
            self.message_history.append(assistant_msg)

            # 如果没有工具调用，返回累积的内容
            if not tool_calls_list:
                return accumulated_content.strip() if accumulated_content else ""

            # 执行工具调用
            print(f"\n🔧 Tool Calls ({len(tool_calls_list)}):")
            for i, tool_call in enumerate(tool_calls_list, 1):
                tool_name = tool_call["function"]["name"]
                args = tool_call["function"]["arguments"]

                print(f"  [{i}] {tool_name}")
                if args:
                    try:
                        args_pretty = json.dumps(json.loads(args), ensure_ascii=False, indent=2)
                        print(f"      Args: {args_pretty[:200]}")
                    except:
                        print(f"      Args: {args[:200]}")

                # 显示执行指示器
                import time

                start_time = time.time()
                print(f"      Executing...", end="", flush=True)

                # 执行工具
                tool_result = self.call_tool(tool_name, args)

                elapsed = (time.time() - start_time) * 1000
                print(f" ✓ Done ({elapsed:.0f}ms)")

                result_display = (
                    tool_result[:300] + "..." if len(tool_result) > 300 else tool_result
                )
                print(f"      Result: {result_display}")

                # 使用 MessageBuilder 构建工具响应
                tool_msg = MessageBuilder.build_tool_response_message(
                    tool_call["id"], str(tool_result)
                )
                self.message_history.append(tool_msg)

            print()  # 工具部分后的空行

        return "Reached maximum steps without a final answer."

    async def astream(self, input: str, reset: bool = False) -> str:
        """异步流式调用"""
        # 目前使用同步实现，子类可重写为真正的异步
        return self.stream(input, reset)


# ============================================================================
# 示例工具和用法
# ============================================================================

import re
from dotenv import load_dotenv


# 使用新的 @tool 装饰器创建工具
from apps.engineer.coder.core.tools.base import tool


@tool()
def search_tool(query: str) -> str:
    """搜索工具示例"""
    query = query.strip()
    if not query:
        return "Empty query"
    return f"Search results for '{query}': [Example result 1, Example result 2]"


@tool()
def calculator_tool(expression: str) -> str:
    """计算器工具示例"""
    expr = expression.strip()
    if not expr:
        return "Empty expression"
    if not re.match(r"^[0-9+\-*/().\s]+$", expr):
        return "Invalid characters in expression"
    try:
        result = eval(expr, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Calc error: {e}"


if __name__ == "__main__":
    load_dotenv()

    # 创建内置工具
    builtin_tools = [
        ReadFileTool(),
        WriteFileTool(),
        EditFileTool(),
        GlobTool(),
        GrepTool(),
        ShellTool(),
        WebSearchTool(),
        WebFetchTool(),
    ]

    agent = ToolUseAgent(
        name="ExampleAgent",
        model=Model(),
        tools=builtin_tools,  # 使用新的工具系统
        max_steps=5,
    )

    # 创建内置工具
    builtin_tools = create_all_builtin_tools()

    agent = ToolUseAgent(
        name="ExampleAgent",
        model=Model(),
        tools=builtin_tools,  # 使用新的工具系统
        max_steps=5,
    )

    result = agent.invoke(
        "What's the population of Paris? Please use the web_search tool to find this information."
    )
    print(result)
