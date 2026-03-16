"""
ToolUseAgent - 基础工具使用Agent

架构设计:
=========

层级结构:
    BaseAgent (core/agent.py)
        ↓ 继承
    ToolUseAgent (当前文件)
        ↓ 继承
    [McpAgent, SkillsUseAgent, TaskAgent, ...]

职责分离:
    - BaseAgent: 提供基础Agent接口 (name, description, model, max_steps)
    - ConversationMixin (core/utils.py): 提供对话循环实现
    - ToolUseAgent: 整合BaseAgent + ConversationMixin，作为工具使用Agent的基类

核心组件:
    1. ConversationMixin (来自 core/utils.py)
       - run_conversation(): 标准对话循环
       - run_conversation_stream(): 流式对话循环
       - _execute_tool_call(): 工具调用执行
       - _init_conversation(): 对话初始化

    2. MessageBuilder (来自 core/utils.py)
       - 构建标准消息格式 (system/user/assistant/tool)
       - 处理工具调用和响应的转换

    3. ToolParser (来自 core/utils.py)
       - 从LLM输出解析工具调用
       - 支持 functions.xxx 和 XML 格式

设计模式:
    - 多重继承: 从 BaseAgent 和 ConversationMixin 组合功能
    - 模板方法: 子类可重写 _build_system_prompt() 自定义行为
    - 依赖注入: 通过构造函数注入 model 和 tools

扩展指南:
    子类化时可以:
    1. 重写 _build_system_prompt() - 自定义系统提示
    2. 重写 run() - 实现更复杂的执行逻辑
    3. 在 __init__ 中 self.tools.append() - 添加额外工具
    4. 重写 invoke() - 自定义调用流程

工作流程:
    User Input → _build_system_prompt() → LLM
                                        ↓
    Final Answer ← Tool Results ← Tool Execution ← Tool Calls
"""

from typing import List, Optional

try:
    from apps.engineer.learn.agent.core.agent import BaseAgent
    from apps.engineer.learn.agent.core.model import Model
    from apps.engineer.learn.agent.core.tool import Tool
    from apps.engineer.learn.agent.core.utils import ConversationMixin
except ImportError:
    from learn.agent.core.agent import BaseAgent
    from learn.agent.core.model import Model
    from learn.agent.core.tool import Tool
    from learn.agent.core.utils import ConversationMixin


class ToolUseAgent(BaseAgent, ConversationMixin):
    """
    基础工具使用Agent - 所有工具使用Agent的基类

    多重继承结构:
        BaseAgent 提供: name, description, model, max_steps, run(), arun()
        ConversationMixin 提供: run_conversation(), run_conversation_stream(),
                                _execute_tool_call(), _init_conversation()

    职责:
        1. 管理工具列表 (self.tools)
        2. 提供标准系统提示词构建
        3. 封装同步/异步调用接口

    关键方法:
        _build_system_prompt(): 构建系统提示，子类可重写
        invoke(): 同步调用入口
        stream(): 流式调用入口
        ainvoke()/astream(): 异步接口

    子类化示例:
        class MyAgent(ToolUseAgent):
            def _build_system_prompt(self) -> str:
                return "自定义提示词..."

            def __init__(self, ...):
                super().__init__(...)
                self.tools.append(my_custom_tool)
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
        tools: Optional[List[Tool]] = None,
        max_steps: int = 10,
    ):
        BaseAgent.__init__(self, name, description, model, max_steps)
        ConversationMixin.__init__(self)
        self.tools = tools or []

    def _build_system_prompt(self) -> str:
        """构建系统提示词 - 子类可重写"""
        return self.DEFAULT_SYSTEM_PROMPT

    def invoke(self, input: str) -> str:
        """
        同步调用 - 使用标准对话循环

        Args:
            input: 用户输入

        Returns:
            Agent的最终响应
        """
        return self.run_conversation(input, self._build_system_prompt())

    def stream(self, input: str, reset: bool = False) -> str:
        """
        流式调用 - 实时打印输出

        Args:
            input: 用户输入
            reset: 是否重置对话历史

        Returns:
            Agent的最终响应
        """
        return self.run_conversation_stream(
            input, self._build_system_prompt(), reset=reset, print_output=True
        )

    async def ainvoke(self, input: str) -> str:
        """异步调用 (当前使用同步实现)"""
        return self.invoke(input)

    async def astream(self, input: str) -> str:
        """异步流式调用 (当前使用同步实现)"""
        return self.stream(input, reset=True)


# ============================================================================
# 示例工具和用法
# ============================================================================

import json
import re
from dotenv import load_dotenv


def search_tool(args: str) -> str:
    """搜索工具示例"""
    try:
        parsed = json.loads(args)
        query = parsed.get("query", args)
    except:
        query = args

    query = query.strip()
    if not query:
        return "Empty query"
    return f"Search results for '{query}': [Example result 1, Example result 2]"


def calculator_tool(args: str) -> str:
    """计算器工具示例"""
    try:
        parsed = json.loads(args)
        expr = parsed.get("expression", args)
    except:
        expr = args

    expr = expr.strip()
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

    tools = [
        Tool(name="search", description="Web search", func=search_tool),
        Tool(name="calculator", description="Simple calculator", func=calculator_tool),
    ]

    agent = ToolUseAgent(
        name="ExampleAgent",
        model=Model(),
        tools=tools,
        max_steps=5,
    )

    result = agent.invoke(
        "What's the population of Paris? Please use the search tool to find this information."
    )
    print(result)
