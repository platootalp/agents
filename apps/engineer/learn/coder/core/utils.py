"""
共享工具类和辅助函数

提供 Agent 开发中常用的工具类：
- MessageBuilder: 构建 LLM 消息
- ConversationMixin: 对话循环复用
- ToolExecutorMixin: 工具执行复用
"""

import json
import re
import time
from typing import Any, Callable
from dataclasses import dataclass, field


@dataclass
class ToolCallResult:
    """工具调用执行结果"""

    tool_call_id: str
    tool_name: str
    result: str
    elapsed_ms: float
    args: str = ""  # JSON string of arguments for display


class MessageBuilder:
    """统一消息构建工具类"""

    @staticmethod
    def build_system_message(content: str) -> dict[str, str]:
        """构建系统消息"""
        return {"role": "system", "content": content}

    @staticmethod
    def build_user_message(content: str) -> dict[str, str]:
        """构建用户消息"""
        return {"role": "user", "content": content}

    @staticmethod
    def build_assistant_message(
        content: str, tool_calls: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """构建助手消息"""
        msg: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        return msg

    @staticmethod
    def build_tool_response_message(tool_call_id: str, content: str) -> dict[str, Any]:
        """构建工具响应消息"""
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": str(content),
        }

    @staticmethod
    def convert_api_tool_calls(api_tool_calls: list[Any]) -> list[dict[str, Any]]:
        """将 API 工具调用对象转换为字典格式"""
        if not api_tool_calls:
            return []
        return [
            {
                "id": tc.id,
                "type": tc.type,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in api_tool_calls
        ]

    @staticmethod
    def accumulate_tool_calls(tool_call_deltas: list[Any]) -> dict[int, dict[str, Any]]:
        """从流式响应中累积工具调用"""
        accumulated: dict[int, dict[str, Any]] = {}
        for tc_delta in tool_call_deltas:
            index = tc_delta.index
            if index not in accumulated:
                accumulated[index] = {
                    "id": tc_delta.id or "",
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                }
            if tc_delta.id:
                accumulated[index]["id"] = tc_delta.id
            if tc_delta.function:
                if tc_delta.function.name:
                    accumulated[index]["function"]["name"] = tc_delta.function.name
                if tc_delta.function.arguments:
                    accumulated[index]["function"]["arguments"] += tc_delta.function.arguments
        return accumulated


class ToolParser:
    """工具调用解析器 - 处理各种格式的工具调用"""

    @staticmethod
    def parse_from_content(content: str) -> list[dict[str, Any]]:
        """
        从内容字符串中解析工具调用

        支持格式:
        1. functions.name:index {json_args}
        2. <function_calls>...</function_calls> XML格式
        """
        if not content:
            return []

        tool_calls = []

        # Pattern 1: functions.name:index {json_args}
        pattern1 = r"functions\.([\w\-]+):(\d+)\s*[\s\S]*?(\{[\s\S]*?\})"
        for match in re.finditer(pattern1, content):
            name = match.group(1)
            index = match.group(2)
            args_str = match.group(3)
            try:
                args = json.loads(args_str)
                tool_calls.append(
                    {
                        "id": f"call_{index}",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": args if isinstance(args, str) else json.dumps(args),
                        },
                    }
                )
            except json.JSONDecodeError:
                continue

        # Pattern 2: <function_calls> XML format
        if not tool_calls:
            func_calls_pattern = r"<function_calls>.*?</function_calls>"
            func_calls_match = re.search(func_calls_pattern, content, re.DOTALL)
            if func_calls_match:
                invoke_pattern = r'<invoke name="(\w+)">(.*?)</invoke>'
                for invoke_match in re.finditer(
                    invoke_pattern, func_calls_match.group(0), re.DOTALL
                ):
                    name = invoke_match.group(1)
                    params_str = invoke_match.group(2)
                    params = {}
                    param_pattern = r'<parameter name="(\w+)">(.*?)</parameter>'
                    for param_match in re.finditer(param_pattern, params_str, re.DOTALL):
                        params[param_match.group(1)] = param_match.group(2)
                    tool_calls.append(
                        {
                            "id": f"call_parsed_{len(tool_calls)}",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(params),
                            },
                        }
                    )

        return tool_calls


class ToolExecutorMixin:
    """工具执行复用 Mixin"""

    def __init__(self):
        self.tools = []

    def _get_openai_tools(self) -> list[dict[str, Any]] | None:
        """将工具转换为 OpenAI 函数调用格式"""
        if not self.tools:
            return None

        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self.tools
        ]

    def call_tool(self, tool_name: str, args: str) -> str:
        """调用工具并返回结果"""
        for tool in self.tools:
            if tool.name.lower() == tool_name.lower():
                if callable(tool.func):
                    try:
                        return tool.func(args)
                    except Exception as e:
                        return f"Tool {tool.name} error: {e}"
                return tool.description or f"No callable for tool {tool.name}"
        return f"Tool {tool_name} not found"

    def _format_tool_args(self, args: str) -> str:
        """格式化工具参数用于显示"""
        if args:
            try:
                return json.dumps(json.loads(args), indent=4)
            except json.JSONDecodeError:
                return args
        return ""

    def _print_tool_execution(self, result: ToolCallResult, index: int = 0) -> None:
        """打印工具执行详情"""
        prefix = f"  [{index}] " if index > 0 else "  "
        print(f"{prefix}{result.tool_name}")

        if result.args:
            args_pretty = self._format_tool_args(result.args)
            if args_pretty:
                print(f"      Args: {args_pretty}")

        print(f"      Executing...", end="", flush=True)
        print(f" ✓ Done ({result.elapsed_ms:.0f}ms)")

        result_display = result.result[:300] + "..." if len(result.result) > 300 else result.result
        print(f"      Result: {result_display}")


class ConversationMixin(ToolExecutorMixin):
    """对话循环复用 Mixin"""

    def __init__(self):
        super().__init__()
        self.message_history: list[dict[str, Any]] = []
        self.max_steps: int = 10

    def _init_conversation(self, input: str, system_prompt: str, reset: bool = False) -> None:
        """初始化对话历史"""
        if reset or not self.message_history:
            self.message_history = [
                MessageBuilder.build_system_message(system_prompt),
            ]
        self.message_history.append(MessageBuilder.build_user_message(input))

    def _execute_tool_call(self, tool_call: dict[str, Any]) -> ToolCallResult:
        """执行单个工具调用"""
        tool_name = tool_call["function"]["name"]
        tool_id = tool_call["id"]
        raw_args = tool_call["function"]["arguments"]

        # 验证参数是否为有效的 JSON
        try:
            json.loads(raw_args)
            args_to_send = raw_args
        except json.JSONDecodeError:
            # 如果参数不是有效的 JSON，尝试包装为查询参数
            args_to_send = json.dumps({"query": raw_args}, ensure_ascii=False)

        start_time = time.time()
        result = self.call_tool(tool_name, args_to_send)
        elapsed_ms = (time.time() - start_time) * 1000

        return ToolCallResult(
            tool_call_id=tool_id,
            tool_name=tool_name,
            result=str(result),
            elapsed_ms=elapsed_ms,
            args=args_to_send,
        )

    def _run_conversation_step(
        self, openai_tools: list[dict[str, Any]] | None, print_output: bool = False
    ) -> tuple[str | None, bool]:
        """
        执行单步对话

        Returns:
            (response_content, has_more_steps): 响应内容和是否还有更多步骤
        """
        from apps.engineer.learn.agent.core.model import Model

        if not self.model:
            return "No model configured.", False

        response = self.model.generate(self.message_history, tools=openai_tools)
        message = response.choices[0].message

        # 转换工具调用
        tool_calls = MessageBuilder.convert_api_tool_calls(message.tool_calls)

        # 如果没有从 API 获取到工具调用，尝试从内容解析
        if not tool_calls and message.content:
            parsed_calls = ToolParser.parse_from_content(message.content)
            if parsed_calls:
                tool_calls = parsed_calls
                if print_output:
                    print(f"\n[Parsed {len(tool_calls)} tool calls from content]")

        # 构建助手消息
        assistant_msg = MessageBuilder.build_assistant_message(message.content or "", tool_calls)
        self.message_history.append(assistant_msg)

        if print_output and message.content:
            print(f"\n🤖 {message.content}")

        # 执行工具调用
        if tool_calls:
            if print_output:
                print(f"\n🔧 Tool Calls ({len(tool_calls)}):")

            for i, tool_call in enumerate(tool_calls, 1):
                result = self._execute_tool_call(tool_call)

                if print_output:
                    self._print_tool_execution(result, i)

                # 添加工具响应到历史
                tool_msg = MessageBuilder.build_tool_response_message(
                    result.tool_call_id, result.result
                )
                self.message_history.append(tool_msg)

            if print_output:
                print()

            return None, True  # 继续执行

        # 返回最终响应
        if message.content:
            return message.content.strip(), False

        return None, False

    def _run_conversation_stream_step(
        self, openai_tools: list[dict[str, Any]] | None, print_output: bool = False
    ) -> tuple[str | None, bool]:
        """
        执行单步流式对话

        Returns:
            (response_content, has_more_steps): 响应内容和是否还有更多步骤
        """
        if not self.model:
            return "No model configured.", False

        accumulated_content = ""
        accumulated_tool_calls: dict[int, dict[str, Any]] = {}

        stream = self.model.stream(self.message_history, tools=openai_tools)

        for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            if delta.content:
                accumulated_content += delta.content
                if print_output:
                    print(delta.content, end="", flush=True)

            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    index = tc_delta.index
                    if index not in accumulated_tool_calls:
                        accumulated_tool_calls[index] = {
                            "id": tc_delta.id or "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    if tc_delta.id:
                        accumulated_tool_calls[index]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            accumulated_tool_calls[index]["function"]["name"] = (
                                tc_delta.function.name
                            )
                        if tc_delta.function.arguments:
                            accumulated_tool_calls[index]["function"]["arguments"] += (
                                tc_delta.function.arguments
                            )

        tool_calls_list = list(accumulated_tool_calls.values())

        # 验证工具调用参数的 JSON 完整性
        valid_tool_calls = []
        for tc in tool_calls_list:
            args = tc.get("function", {}).get("arguments", "")
            try:
                json.loads(args)
                valid_tool_calls.append(tc)
            except json.JSONDecodeError:
                # 如果参数不完整，跳过这个工具调用
                if print_output:
                    print(
                        f"\n[跳过不完整的工具调用: {tc.get('function', {}).get('name', 'unknown')}]"
                    )
                continue

        tool_calls_list = valid_tool_calls

        # 如果没有从 API 获取到工具调用，尝试从内容解析
        if not tool_calls_list and accumulated_content:
            parsed_calls = ToolParser.parse_from_content(accumulated_content)
            if parsed_calls:
                tool_calls_list = parsed_calls
                if print_output:
                    print(f"\n[Parsed {len(tool_calls_list)} tool calls from content]")

        # 构建助手消息
        assistant_msg = MessageBuilder.build_assistant_message(accumulated_content, tool_calls_list)
        self.message_history.append(assistant_msg)

        # 执行工具调用
        if tool_calls_list:
            if print_output:
                print(f"\n🔧 Tool Calls ({len(tool_calls_list)}):")

            for i, tool_call in enumerate(tool_calls_list, 1):
                result = self._execute_tool_call(tool_call)

                if print_output:
                    self._print_tool_execution(result, i)

                tool_msg = MessageBuilder.build_tool_response_message(
                    result.tool_call_id, result.result
                )
                self.message_history.append(tool_msg)

            if print_output:
                print()

            return None, True

        return accumulated_content.strip() if accumulated_content else "", False

    def run_conversation(
        self, input: str, system_prompt: str, reset: bool = False, print_output: bool = False
    ) -> str:
        """
        运行完整对话循环

        Args:
            input: 用户输入
            system_prompt: 系统提示词
            reset: 是否重置对话历史
            print_output: 是否打印输出

        Returns:
            最终响应
        """
        self._init_conversation(input, system_prompt, reset)

        if print_output:
            print(f"\n🆕 New Conversation\n")
            print(f"👤 User: {input}\n")

        for _ in range(self.max_steps):
            openai_tools = self._get_openai_tools()
            result, has_more = self._run_conversation_step(openai_tools, print_output)

            if not has_more:
                return result or ""

        return "Reached maximum steps without a final answer."

    def run_conversation_stream(
        self, input: str, system_prompt: str, reset: bool = False, print_output: bool = False
    ) -> str:
        """
        运行完整流式对话循环

        Args:
            input: 用户输入
            system_prompt: 系统提示词
            reset: 是否重置对话历史
            print_output: 是否打印输出

        Returns:
            最终响应
        """
        self._init_conversation(input, system_prompt, reset)

        if print_output:
            if reset or len(self.message_history) <= 2:
                print("\n🆕 New Conversation\n")
            print(f"👤 User: {input}\n")

        for _ in range(self.max_steps):
            openai_tools = self._get_openai_tools()
            result, has_more = self._run_conversation_stream_step(openai_tools, print_output)

            if not has_more:
                return result or ""

        return "Reached maximum steps without a final answer."
