"""
McpAgent - MCP (Model Context Protocol) Agent

架构设计:
=========

继承关系:
    ToolUseAgent (tool_use_agent.py)
        ↓ 继承
    McpAgent (当前文件)

MCP协议支持:
    McpAgent 通过 FastMCP 客户端连接到 MCP 服务器，支持两种传输方式:
    1. StdioTransport: 通过子进程标准输入输出通信 (最可靠)
    2. HTTP Transport: 通过HTTP连接到远程MCP服务器

核心组件:
    1. FastMCP Client (fastmcp.Client)
       - 连接MCP服务器
       - 列出可用工具
       - 调用远程工具

    2. ToolManager (来自 coder.core.tools.manager)
       - 管理本地工具
       - 提供工具注册和查找

    3. MessageBuilder (来自 coder.core.utils)
       - 构建包含MCP工具信息的系统提示
       - 转换MCP工具格式为OpenAI工具格式

    4. ToolUseAgent (继承)
       - 复用标准对话循环
       - 提供基础Agent功能

数据流:
    1. 初始化: McpAgent → 创建 FastMCP Client → 连接服务器 → 获取工具列表
    2. 调用: User Input → ToolUseAgent.invoke() → MCP工具调用 → 返回结果

关键方法:
    _build_system_prompt(): 构建包含MCP服务器工具信息的系统提示
    _convert_mcp_tools(): 将MCP工具格式转换为OpenAI工具格式
    _call_mcp_tool(): 调用MCP服务器上的工具

依赖:
    - fastmcp: MCP协议客户端库
    - ToolUseAgent: 基础工具使用Agent
    - ToolManager: 新工具系统的工具管理器

使用场景:
    - 连接外部MCP服务器获取工具能力
    - 集成远程服务（如文件系统、数据库、API等）
    - 构建可扩展的工具生态系统
"""

import asyncio
import json
from typing import Optional, List, Dict, Any, AsyncIterator
from dataclasses import dataclass, field

from dotenv import load_dotenv

# 新工具系统导入
try:
    from apps.engineer.coder.core.tools.base import BaseTool, Tool, ToolResult
    from apps.engineer.coder.core.tools.manager import ToolManager
except ImportError:
    from apps.engineer.coder.core.tools.base import BaseTool, Tool, ToolResult
    from apps.engineer.coder.core.tools.manager import ToolManager

# Agent基础导入
try:
    from apps.engineer.coder.core.model import Model
    from apps.engineer.coder.agents.tool_use_agent import ToolUseAgent
    from apps.engineer.coder.core.utils import MessageBuilder
except ImportError:
    from apps.engineer.coder.core.model import Model
    from apps.engineer.coder.agents.tool_use_agent import ToolUseAgent
    from apps.engineer.coder.core.utils import MessageBuilder

try:
    from fastmcp import Client
    from fastmcp.client.transports import PythonStdioTransport

    FASTMCP_AVAILABLE = True
except ImportError:
    FASTMCP_AVAILABLE = False


@dataclass
class StreamChunk:
    """Represents a chunk of streamed response."""

    content: str = ""
    reasoning: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    is_complete: bool = False


class McpAgent(ToolUseAgent):
    """
    MCP协议Agent - 连接MCP服务器并使用远程工具

    继承自 ToolUseAgent，添加MCP协议支持:
        - 连接MCP服务器 (stdio 或 HTTP)
        - 动态获取服务器工具列表
        - 调用远程MCP工具
        - 复用 ToolUseAgent 的对话循环

    传输方式:
        1. StdioTransport (推荐): 通过子进程通信
           - 最可靠，支持所有MCP功能
           - 使用 command + args 参数

        2. HTTP Transport: 连接远程HTTP服务器
           - 适合云服务部署
           - 使用 server_url 参数

    工具集成:
        - 连接时自动从服务器获取工具列表
        - 转换为OpenAI工具格式供LLM使用
        - 工具调用自动路由到MCP服务器

    关键方法:
        __init__(): 配置传输方式，连接服务器
        _build_system_prompt(): 构建包含MCP工具的系统提示
        _call_mcp_tool(): 调用MCP服务器的工具

    示例:
        ```python
        # Stdio transport (本地Python脚本作为MCP服务器)
        coder = McpAgent(
            name="MCP Agent",
            model=Model(),
            command="python",
            args=["mcp_server.py"],
        )

        # HTTP transport (远程MCP服务)
        coder = McpAgent(
            name="MCP Agent",
            model=Model(),
            server_url="http://localhost:8000/mcp",
        )
        ```
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        model: Optional[Model] = None,
        tools: Optional[List[BaseTool]] = None,
        max_steps: int = 10,
        # HTTP transport
        server_url: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        # Stdio transport
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
    ):
        """Initialize the MCP coder.

        Args:
            name: Agent name
            description: Agent description
            model: LLM model instance
            tools: Additional local tools (optional) - 使用新的 BaseTool
            max_steps: Maximum reasoning steps
            server_url: URL for HTTP transport (e.g., "http://localhost:8000/mcp")
            headers: Custom HTTP headers for remote MCP services
            command: Command for stdio transport (e.g., "python")
            args: Arguments for the command
            env: Environment variables
        """
        super().__init__(name, description, model, tools, max_steps)

        if not FASTMCP_AVAILABLE:
            raise ImportError("FastMCP not installed. Install with: uv add fastmcp")

        if server_url:
            # HTTP transport with optional headers
            if headers:
                try:
                    from fastmcp.client.transports import StreamableHttpTransport

                    transport = StreamableHttpTransport(
                        url=server_url,
                        headers=headers,
                    )
                    self._client = Client(transport)
                except ImportError:
                    raise ImportError(
                        "Custom headers require StreamableHTTPTransport. "
                        "Please upgrade FastMCP or use stdio transport."
                    )
            else:
                self._client = Client(server_url)
        elif command and args:
            # Stdio transport using PythonStdioTransport
            script_path = args[0]
            script_args = args[1:] if len(args) > 1 else None
            transport = PythonStdioTransport(
                script_path=script_path,
                args=script_args,
                env=env,
                python_cmd=command,
            )
            self._client = Client(transport)
        else:
            raise ValueError("Must provide either server_url or command+args")

        self._connected = False

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the coder."""
        return (
            "You are a helpful assistant with access to tools. "
            "\n\n## ABSOLUTE RULES (VIOLATION WILL CAUSE ERRORS):\n"
            "1. When you need to use a tool, you MUST use the tool_calls field ONLY. "
            "NEVER output tool calls in the 'content' field.\n"
            "\n"
            "2. NEVER output text like 'functions.xxx' or tool calls in content. "
            "These are NOT valid tool calls - they will fail.\n"
            "\n"
            "3. tool_calls MUST be in the dedicated tool_calls field, NOT in content.\n"
            "\n\n## CORRECT OUTPUT FORMAT:\n"
            "- If tools needed: content='', tool_calls=[actual tool call objects]\n"
            "- If no tools needed: content='your answer', tool_calls=[]\n"
            "\n\nRemember: ONLY tool_calls field can trigger tool execution."
        )

    async def connect(self) -> None:
        """Connect to the MCP server."""
        self._connected = True

    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        self._connected = False

    async def _list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from the MCP server."""
        async with self._client as client:
            tools = await client.list_tools()
            return [
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema,
                }
                for tool in tools
            ]

    def _get_openai_tools(self) -> Optional[List[Dict[str, Any]]]:
        """Convert all tools (local + MCP) to OpenAI function calling format."""
        openai_tools = []

        # Add local tools from tool_manager
        if self.tool_manager:
            openai_tools.extend(self.tool_manager.get_openai_tools())

        return openai_tools if openai_tools else None

    async def _get_all_tools(self) -> List[Dict[str, Any]]:
        """Get combined list of local and MCP tools."""
        all_tools = self._get_openai_tools() or []

        if self._connected:
            mcp_tools = await self._list_tools()
            for tool in mcp_tools:
                all_tools.append(
                    {
                        "type": "function",
                        "function": tool,
                    }
                )

        return all_tools

    async def _call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Call a tool on the MCP server."""
        async with self._client as client:
            try:
                result = await client.call_tool(tool_name, arguments)

                content_parts = []
                for item in result.content:
                    if hasattr(item, "text"):
                        content_parts.append(item.text)
                    else:
                        content_parts.append(str(item))

                return "\n".join(content_parts) if content_parts else "No result"

            except Exception as e:
                return f"MCP tool error: {type(e).__name__}: {e}"

    def _call_local_tool(self, tool_name: str, args: Dict[str, Any]) -> Optional[str]:
        """Call a local tool by name using ToolManager."""
        if self.tool_manager:
            result = self.tool_manager.run_tool(tool_name, **args)
            if result.success:
                return str(result.output)
            else:
                return f"Tool error: {result.error}"
        return None

    async def _execute_single_tool(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single tool call."""
        import time

        tool_name = tool_call["function"]["name"]
        tool_id = tool_call["id"]

        try:
            tool_args = json.loads(tool_call["function"]["arguments"])
        except json.JSONDecodeError:
            tool_args = {"query": tool_call["function"]["arguments"]}

        start_time = time.time()

        # Try local tool first
        result = self._call_local_tool(tool_name, tool_args)

        # If not local, try MCP tool
        if result is None:
            result = await self._call_mcp_tool(tool_name, tool_args)

        elapsed_ms = (time.time() - start_time) * 1000

        return {
            "tool_call_id": tool_id,
            "tool_name": tool_name,
            "result": str(result),
            "elapsed_ms": elapsed_ms,
            "args": tool_call["function"]["arguments"],
        }

    async def _execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute multiple tool calls."""
        results = []
        for tool_call in tool_calls:
            result = await self._execute_single_tool(tool_call)
            results.append(result)
        return results

    def _ensure_sync_context(self, method_name: str) -> None:
        """Ensure we're not in an async context for sync methods."""
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                raise RuntimeError(
                    f"Cannot call {method_name}() from an async context. "
                    f"Use the async version instead."
                )
        except RuntimeError:
            pass

    def _validate_setup(self, require_fastmcp: bool = False) -> Optional[str]:
        """Validate coder setup and return error message if invalid."""
        if require_fastmcp and not FASTMCP_AVAILABLE:
            return "FastMCP not available. Install with: uv add fastmcp"
        if not self.model:
            return "No model configured."
        return None

    async def _run_conversation_loop(
        self,
        input: str,
        streaming: bool = False,
        print_output: bool = False,
    ) -> str:
        """Core conversation loop implementation."""
        if not self._connected:
            await self.connect()

        all_tools = await self._get_all_tools()

        # Initialize conversation
        self._init_conversation(input, self._build_system_prompt())

        for step in range(self.max_steps):
            if streaming:
                result = await self._run_streaming_step(all_tools, print_output)
            else:
                result = await self._run_non_streaming_step(all_tools)

            if result is not None:
                return result

        return "Reached maximum steps without a final answer."

    async def _run_streaming_step(
        self, all_tools: List[Dict[str, Any]], print_output: bool
    ) -> Optional[str]:
        """Run one step of the conversation in streaming mode."""
        accumulated_content = ""
        accumulated_tool_calls: Dict[int, Dict[str, Any]] = {}

        stream = self.model.stream(self.message_history, tools=all_tools or None)

        for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            if delta.content:
                accumulated_content += delta.content
                if print_output:
                    print(delta.content, end="", flush=True)

            if delta.tool_calls:
                accumulated_tool_calls.update(
                    MessageBuilder.accumulate_tool_calls(delta.tool_calls)
                )

        tool_calls_list = list(accumulated_tool_calls.values())

        assistant_msg = MessageBuilder.build_assistant_message(accumulated_content, tool_calls_list)
        self.message_history.append(assistant_msg)

        if tool_calls_list:
            await self._execute_and_record_tools(tool_calls_list, print_output)
            return None

        return accumulated_content.strip() if accumulated_content else ""

    async def _run_non_streaming_step(self, all_tools: List[Dict[str, Any]]) -> Optional[str]:
        """Run one step of the conversation in non-streaming mode."""
        response = self.model.generate(self.message_history, tools=all_tools or None)
        message = response.choices[0].message

        tool_calls = MessageBuilder.convert_api_tool_calls(message.tool_calls)

        assistant_msg = MessageBuilder.build_assistant_message(message.content or "", tool_calls)
        self.message_history.append(assistant_msg)

        if tool_calls:
            await self._execute_and_record_tools(tool_calls, print_output=False)
            return None

        if message.content:
            return message.content.strip()

        return None

    async def _execute_and_record_tools(
        self, tool_calls: List[Dict[str, Any]], print_output: bool
    ) -> None:
        """Execute tool calls and record results in history."""
        if print_output:
            print(f"\n🔧 Tool Calls ({len(tool_calls)}):")

        results = await self._execute_tool_calls(tool_calls)

        for i, result in enumerate(results, 1):
            if print_output:
                print(f"  [{i}] {result['tool_name']}")
                print(f"      Executing... ✓ Done ({result['elapsed_ms']:.0f}ms)")
                result_display = (
                    result["result"][:300] + "..."
                    if len(result["result"]) > 300
                    else result["result"]
                )
                print(f"      Result: {result_display}")

            tool_msg = MessageBuilder.build_tool_response_message(
                result["tool_call_id"], result["result"]
            )
            self.message_history.append(tool_msg)

        if print_output:
            print()

    def invoke(self, input: str) -> str:
        """Process user input using MCP tools and LLM (synchronous)."""
        if error := self._validate_setup(require_fastmcp=True):
            return error

        self._ensure_sync_context("invoke")
        return asyncio.run(self._run_conversation_loop(input, streaming=False))

    async def ainvoke(self, input: str) -> str:
        """Async version of invoke."""
        if error := self._validate_setup(require_fastmcp=True):
            return error

        return await self._run_conversation_loop(input, streaming=False)

    def stream(self, input: str, reset: bool = False) -> str:
        """Stream the response from the model (synchronous)."""
        if error := self._validate_setup():
            return error

        self._ensure_sync_context("stream")

        self._init_conversation(input, self._build_system_prompt(), reset)

        if reset or len(self.message_history) <= 2:
            print("\n🆕 New Conversation\n")

        print(f"👤 User: {input}\n")

        return asyncio.run(self._run_conversation_loop(input, streaming=True, print_output=True))

    async def astream(self, input: str, reset: bool = False) -> AsyncIterator[StreamChunk]:
        """Async streaming response generator."""
        if error := self._validate_setup(require_fastmcp=True):
            yield StreamChunk(content=error, is_complete=True)
            return

        self._init_conversation(input, self._build_system_prompt(), reset)

        if not self._connected:
            await self.connect()

        all_tools = await self._get_all_tools()

        for _ in range(self.max_steps):
            async for chunk in self._astream_response_chunk(all_tools):
                yield chunk

            last_msg = self.message_history[-1] if self.message_history else {}
            tool_calls = last_msg.get("tool_calls", [])
            tool_calls_list = tool_calls if isinstance(tool_calls, list) else []
            has_tool_calls = len(tool_calls_list) > 0

            if has_tool_calls:
                async for chunk in self._astream_tool_results(tool_calls_list):
                    yield chunk
            else:
                yield StreamChunk(is_complete=True)
                return

        yield StreamChunk(
            content="Reached maximum steps without a final answer.",
            is_complete=True,
        )

    async def _astream_response_chunk(
        self, all_tools: List[Dict[str, Any]]
    ) -> AsyncIterator[StreamChunk]:
        """Stream one response chunk and accumulate tool calls."""
        accumulated_content = ""
        accumulated_reasoning = ""
        accumulated_tool_calls: Dict[int, Dict[str, Any]] = {}

        stream = self.model.stream(self.message_history, tools=all_tools or None)

        for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            if delta.content:
                accumulated_content += delta.content
                yield StreamChunk(content=delta.content)

            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                accumulated_reasoning += delta.reasoning_content
                yield StreamChunk(reasoning=delta.reasoning_content)

            if delta.tool_calls:
                accumulated_tool_calls.update(
                    MessageBuilder.accumulate_tool_calls(delta.tool_calls)
                )

        tool_calls_list = list(accumulated_tool_calls.values())

        assistant_msg = MessageBuilder.build_assistant_message(accumulated_content, tool_calls_list)
        self.message_history.append(assistant_msg)

    async def _astream_tool_results(
        self, tool_calls: List[Dict[str, Any]]
    ) -> AsyncIterator[StreamChunk]:
        """Execute tool calls and yield results."""
        results = await self._execute_tool_calls(tool_calls)

        for result in results:
            yield StreamChunk(
                tool_calls=[
                    {
                        "name": result["tool_name"],
                        "result": result["result"],
                        "elapsed_ms": result["elapsed_ms"],
                    }
                ]
            )

            tool_msg = MessageBuilder.build_tool_response_message(
                result["tool_call_id"], result["result"]
            )
            self.message_history.append(tool_msg)


# =============================================================================
# Example
# =============================================================================


async def example():
    """Example: Connect McpAgent to a FastMCP server via stdio."""
    import tempfile
    import os

    server_code = '''
from fastmcp import FastMCP

mcp = FastMCP("Demo Server")

@mcp.tool
def calculator(operation: str, a: float, b: float) -> str:
    """Perform basic math operations."""
    if operation == "add":
        return f"Result: {a + b}"
    elif operation == "subtract":
        return f"Result: {a - b}"
    elif operation == "multiply":
        return f"Result: {a * b}"
    elif operation == "divide":
        if b == 0:
            return "Error: Division by zero"
        return f"Result: {a / b}"
    return "Error: Unknown operation"

@mcp.tool
def weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: Sunny, 25°C, Humidity: 60%"

if __name__ == "__main__":
    mcp.run()
'''

    load_dotenv()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(server_code)
        server_path = f.name

    try:
        print("=" * 50)
        print("🚀 FastMCP Example (stdio transport)")
        print("=" * 50)

        print("\n1️⃣  Creating McpAgent (stdio transport)...")
        agent = McpAgent(
            name="DemoAgent",
            model=Model(),
            command="python",
            args=[server_path],
        )
        print("   ✓ Agent created")

        print("\n2️⃣  Testing tool calls:")
        result = await agent._call_mcp_tool("calculator", {"operation": "multiply", "a": 5, "b": 3})
        print(f"   Calculator (5 × 3): {result}")

        result = await agent._call_mcp_tool("weather", {"city": "Beijing"})
        print(f"   Weather (Beijing): {result}")

        print("\n" + "=" * 50)
        print("✅ Example completed!")
        print("=" * 50)

    finally:
        os.unlink(server_path)


if __name__ == "__main__":
    asyncio.run(example())
