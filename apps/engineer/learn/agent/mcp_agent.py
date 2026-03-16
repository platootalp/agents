"""
McpAgent - MCP (Model Context Protocol) Agent

架构设计:
=========

继承关系:
    ToolUseAgent (01.tool_use_agent.py)
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

    2. MessageBuilder (来自 core/utils.py)
       - 构建包含MCP工具信息的系统提示
       - 转换MCP工具格式为OpenAI工具格式

    3. ToolCallResult (来自 core/utils.py)
       - 统一工具调用结果格式

    4. ConversationMixin (继承自 ToolUseAgent)
       - 复用标准对话循环

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
    - MessageBuilder: 消息构建工具

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

try:
    from apps.engineer.learn.agent.core.model import Model
    from apps.engineer.learn.agent.core.tool import Tool
    from apps.engineer.learn.agent.tool_use_agent import ToolUseAgent
    from apps.engineer.learn.agent.core.utils import (
        ToolCallResult,
        MessageBuilder,
        ConversationMixin,
    )
except ImportError:
    from learn.agent.core.model import Model
    from learn.agent.core.tool import Tool
    from learn.agent.tool_use_agent import ToolUseAgent
    from learn.agent.core.utils import (
        ToolCallResult,
        MessageBuilder,
        ConversationMixin,
    )

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
        agent = McpAgent(
            name="MCP Agent",
            model=Model(),
            command="python",
            args=["mcp_server.py"],
        )

        # HTTP transport (远程MCP服务)
        agent = McpAgent(
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
        tools: Optional[List[Tool]] = None,
        max_steps: int = 10,
        # HTTP transport
        server_url: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        # Stdio transport
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
    ):
        """Initialize the MCP agent.

        Args:
            name: Agent name
            description: Agent description
            model: LLM model instance
            tools: Additional local tools (optional)
            max_steps: Maximum reasoning steps
            server_url: URL for HTTP transport (e.g., "http://localhost:8000/mcp")
            headers: Custom HTTP headers for remote MCP services (e.g., {"CONTEXT7_API_KEY": "xxx"})
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
                # Use StreamableHTTPTransport for custom headers
                try:
                    from fastmcp.client.transports import StreamableHttpTransport

                    transport = StreamableHttpTransport(
                        url=server_url,
                        headers=headers,
                    )
                    self._client = Client(transport)
                except ImportError:
                    # Fallback: try httpx transport if available
                    raise ImportError(
                        "Custom headers require StreamableHTTPTransport. "
                        "Please upgrade FastMCP or use stdio transport."
                    )
            else:
                # Simple HTTP transport without headers
                self._client = Client(server_url)
        elif command and args:
            # Stdio transport using PythonStdioTransport
            # First arg is the script path, rest are script arguments
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
        """Build the system prompt for the agent.

        Returns:
            System prompt string with strict output format constraints
        """
        return (
            "You are a helpful assistant with access to tools. "
            "\n\n## ABSOLUTE RULES (VIOLATION WILL CAUSE ERRORS):\n"
            "1. When you need to use a tool, you MUST use the tool_calls field ONLY. "
            "NEVER output tool calls in the 'content' field.\n"
            "\n"
            "2. NEVER output text like 'functions.xxx' or '我来帮你查询' or '让我调用工具'. "
            "These are NOT valid tool calls - they will fail.\n"
            "\n"
            "3. tool_calls MUST be in the dedicated tool_calls field, NOT in content. "
            "If you output tool calls in content, the system cannot execute them.\n"
            "\n\n## CORRECT OUTPUT FORMAT:\n"
            "- If tools needed: content='', tool_calls=[actual tool call objects]\n"
            "- If no tools needed: content='your answer', tool_calls=[]\n"
            "\n\n## WRONG FORMATS (WILL FAIL):\n"
            "❌ content='functions.weather:0 {...}' → NOT EXECUTABLE\n"
            "❌ content='我来帮你查询天气' → NOT EXECUTABLE\n"
            "❌ content='<function_calls>...' → NOT EXECUTABLE\n"
            "❌ content包含任何工具调用格式 → NOT EXECUTABLE\n"
            "\n"
            "✅ content='', tool_calls=[{name:'weather',arguments:{city:'Beijing'}}] → EXECUTABLE\n"
            "\n\n## EXECUTION FLOW:\n"
            "1. User asks question\n"
            "2. You decide if tool is needed\n"
            "3. If YES: output empty content + tool_calls with actual parameters\n"
            "4. System executes tool_calls\n"
            "5. You receive tool results\n"
            "6. You output final answer in content\n"
            "\n\nRemember: ONLY tool_calls field can trigger tool execution. Content field is NEVER parsed for tools."
        )

    async def connect(self) -> None:
        """Connect to the MCP server.

        For FastMCP, connection is handled automatically via async context manager.
        This method is kept for API compatibility.
        """
        self._connected = True

    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        self._connected = False

    async def _list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from the MCP server.

        Returns:
            List of tool definitions
        """
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

        # Add local tools
        if self.tools:
            for tool in self.tools:
                openai_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.parameters,
                        },
                    }
                )

        return openai_tools if openai_tools else None

    async def _get_all_tools(self) -> List[Dict[str, Any]]:
        """Get combined list of local and MCP tools.

        Returns:
            Combined list of all available tools
        """
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
        """Call a tool on the MCP server.

        Args:
            tool_name: Name of the MCP tool to call
            arguments: Tool arguments as a dictionary

        Returns:
            Tool result as a string
        """
        async with self._client as client:
            try:
                result = await client.call_tool(tool_name, arguments)

                # result is CallToolResult with content list
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
        """Call a local tool by name.

        Args:
            tool_name: Name of the tool
            args: Tool arguments

        Returns:
            Tool result or None if tool not found locally
        """
        for t in self.tools:
            if t.name.lower() == tool_name.lower():
                if callable(t.func):
                    try:
                        query = args.get("query", "") if isinstance(args, dict) else str(args)
                        return t.func(query)
                    except Exception as e:
                        return f"Tool {t.name} error: {e}"
                return t.description or f"No callable for tool {t.name}"
        return None

    async def _execute_single_tool(self, tool_call: Dict[str, Any]) -> ToolCallResult:
        """Execute a single tool call.

        Args:
            tool_call: Tool call dictionary with 'id', 'function' keys

        Returns:
            ToolCallResult with execution details
        """
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

        return ToolCallResult(
            tool_call_id=tool_id,
            tool_name=tool_name,
            result=str(result),
            elapsed_ms=elapsed_ms,
            args=tool_call["function"]["arguments"],
        )

    async def _execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[ToolCallResult]:
        """Execute multiple tool calls.

        Args:
            tool_calls: List of tool call dictionaries

        Returns:
            List of ToolCallResult
        """
        results = []
        for tool_call in tool_calls:
            result = await self._execute_single_tool(tool_call)
            results.append(result)
        return results

    def _ensure_sync_context(self, method_name: str) -> None:
        """Ensure we're not in an async context for sync methods.

        Args:
            method_name: Name of the calling method for error message

        Raises:
            RuntimeError: If called from within an async context
        """
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
        """Validate agent setup and return error message if invalid.

        Args:
            require_fastmcp: Whether FastMCP is required for this operation

        Returns:
            Error message if setup is invalid, None otherwise
        """
        if require_fastmcp and not FASTMCP_AVAILABLE:
            return "FastMCP not available. Install with: uv add fastmcp"
        if not self.model:
            return "No model configured."
        return None

    # =========================================================================
    # Conversation Loop Core
    # =========================================================================

    async def _run_conversation_loop(
        self,
        input: str,
        streaming: bool = False,
        print_output: bool = False,
    ) -> str:
        """Core conversation loop implementation.

        Args:
            input: User input
            streaming: Whether to use streaming mode
            print_output: Whether to print output to console

        Returns:
            Final response string
        """
        if not self._connected:
            await self.connect()

        all_tools = await self._get_all_tools()

        # Initialize conversation using shared mixin method
        ConversationMixin._init_conversation(self, input, self._build_system_prompt())

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
        """Run one step of the conversation in streaming mode.

        Args:
            all_tools: Available tools
            print_output: Whether to print output

        Returns:
            Final response if done, None to continue
        """
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

        # Build assistant message using shared MessageBuilder
        assistant_msg = MessageBuilder.build_assistant_message(accumulated_content, tool_calls_list)
        self.message_history.append(assistant_msg)

        # Execute tools if any, otherwise return final answer
        if tool_calls_list:
            await self._execute_and_record_tools(tool_calls_list, print_output)
            return None

        return accumulated_content.strip() if accumulated_content else ""

    async def _run_non_streaming_step(self, all_tools: List[Dict[str, Any]]) -> Optional[str]:
        """Run one step of the conversation in non-streaming mode.

        Args:
            all_tools: Available tools

        Returns:
            Final response if done, None to continue
        """
        response = self.model.generate(self.message_history, tools=all_tools or None)
        message = response.choices[0].message

        tool_calls = MessageBuilder.convert_api_tool_calls(message.tool_calls)

        # Build assistant message using shared MessageBuilder
        assistant_msg = MessageBuilder.build_assistant_message(message.content or "", tool_calls)
        self.message_history.append(assistant_msg)

        # Execute tools if any, otherwise return final answer
        if tool_calls:
            await self._execute_and_record_tools(tool_calls, print_output=False)
            return None

        if message.content:
            return message.content.strip()

        return None

    async def _execute_and_record_tools(
        self, tool_calls: List[Dict[str, Any]], print_output: bool
    ) -> None:
        """Execute tool calls and record results in history.

        Args:
            tool_calls: List of tool calls to execute
            print_output: Whether to print execution details
        """
        if print_output:
            print(f"\n🔧 Tool Calls ({len(tool_calls)}):")

        results = await self._execute_tool_calls(tool_calls)

        for i, result in enumerate(results, 1):
            if print_output:
                self._print_tool_execution(result, i)

            # Use shared MessageBuilder for tool response
            tool_msg = MessageBuilder.build_tool_response_message(
                result.tool_call_id, result.result
            )
            self.message_history.append(tool_msg)

        if print_output:
            print()

    # =========================================================================
    # Public API
    # =========================================================================

    def invoke(self, input: str) -> str:
        """Process user input using MCP tools and LLM (synchronous).

        Args:
            input: User query string

        Returns:
            Final response from the agent
        """
        if error := self._validate_setup(require_fastmcp=True):
            return error

        self._ensure_sync_context("invoke")
        return asyncio.run(self._run_conversation_loop(input, streaming=False))

    async def ainvoke(self, input: str) -> str:
        """Async version of invoke.

        Args:
            input: User query string

        Returns:
            Final response from the agent
        """
        if error := self._validate_setup(require_fastmcp=True):
            return error

        return await self._run_conversation_loop(input, streaming=False)

    def stream(self, input: str, reset: bool = False) -> str:
        """Stream the response from the model (synchronous).

        Note: This method should only be called from synchronous contexts.
        When in an async context, use 'astream()' instead.

        Args:
            input: User input message
            reset: If True, reset conversation history before processing

        Returns:
            Final accumulated response
        """
        if error := self._validate_setup():
            return error

        self._ensure_sync_context("stream")

        # Initialize conversation using shared mixin method
        ConversationMixin._init_conversation(self, input, self._build_system_prompt(), reset)

        if reset or len(self.message_history) <= 2:
            print("\n🆕 New Conversation\n")

        print(f"👤 User: {input}\n")

        return asyncio.run(self._run_conversation_loop(input, streaming=True, print_output=True))

    async def astream(self, input: str, reset: bool = False) -> AsyncIterator[StreamChunk]:
        """Async streaming response generator.

        Args:
            input: User input message
            reset: If True, reset conversation history before processing

        Yields:
            StreamChunk objects containing content, reasoning, and tool calls
        """
        if error := self._validate_setup(require_fastmcp=True):
            yield StreamChunk(content=error, is_complete=True)
            return

        # Initialize conversation using shared mixin method
        ConversationMixin._init_conversation(self, input, self._build_system_prompt(), reset)

        if not self._connected:
            await self.connect()

        all_tools = await self._get_all_tools()

        for _ in range(self.max_steps):
            async for chunk in self._astream_response_chunk(all_tools):
                yield chunk

            # Check if we're done (no tool calls in the last assistant message)
            last_msg = self.message_history[-1] if self.message_history else {}
            tool_calls = last_msg.get("tool_calls", [])
            tool_calls_list = tool_calls if isinstance(tool_calls, list) else []
            has_tool_calls = len(tool_calls_list) > 0

            # Execute tools if any, otherwise we're done
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
        """Stream one response chunk and accumulate tool calls.

        Args:
            all_tools: Available tools

        Yields:
            StreamChunk objects with content and reasoning
        """
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

        # Build assistant message using shared MessageBuilder
        assistant_msg = MessageBuilder.build_assistant_message(accumulated_content, tool_calls_list)
        self.message_history.append(assistant_msg)

    async def _astream_tool_results(
        self, tool_calls: List[Dict[str, Any]]
    ) -> AsyncIterator[StreamChunk]:
        """Execute tool calls and yield results.

        Args:
            tool_calls: List of tool calls to execute

        Yields:
            StreamChunk objects with tool call results
        """
        results = await self._execute_tool_calls(tool_calls)

        for result in results:
            yield StreamChunk(
                tool_calls=[
                    {
                        "name": result.tool_name,
                        "result": result.result,
                        "elapsed_ms": result.elapsed_ms,
                    }
                ]
            )

            # Use shared MessageBuilder for tool response
            tool_msg = MessageBuilder.build_tool_response_message(
                result.tool_call_id, result.result
            )
            self.message_history.append(tool_msg)


# =============================================================================
# Example MCP Server (using FastMCP)
# =============================================================================

"""
Save this as 'simple_server.py':

------------------------------------------------------------------------------
from fastmcp import FastMCP

mcp = FastMCP("Demo Server")

@mcp.tool
def calculator(operation: str, a: float, b: float) -> str:
    '''Perform basic math operations.'''
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
    '''Get weather for a city.'''
    return f"Weather in {city}: Sunny, 25°C"

if __name__ == "__main__":
    mcp.run()  # Uses stdio transport by default
------------------------------------------------------------------------------

Run this example:
    cd apps/engineer
    uv run python learn/agent/03.mcp_agent.py
"""


async def example():
    """Example: Connect McpAgent to a FastMCP server via stdio."""
    import tempfile
    import os

    # Create a temporary MCP server script using stdio transport
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
    mcp.run()  # Uses stdio transport by default
'''

    # Write server script to temp file
    load_dotenv()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(server_code)
        server_path = f.name

    try:
        print("=" * 50)
        print("🚀 FastMCP Example (stdio transport)")
        print("=" * 50)

        # Create McpAgent with stdio transport
        print("\n1️⃣  Creating McpAgent (stdio transport)...")
        agent = McpAgent(
            name="DemoAgent",
            model=Model(),
            command="python",
            args=[server_path],
        )
        print("   ✓ Agent created")

        # Test direct tool call
        print("\n2️⃣  Testing tool calls:")
        result = await agent._call_mcp_tool("calculator", {"operation": "multiply", "a": 5, "b": 3})
        print(f"   Calculator (5 × 3): {result}")

        result = await agent._call_mcp_tool("weather", {"city": "Beijing"})
        print(f"   Weather (Beijing): {result}")

        # Test with LLM if available
        print("\n3️⃣  Natural language query:")
        try:
            print(f"   Query: 'What is 125 multiply 301?'")
            print(f"   Response: ", end="", flush=True)

            async for chunk in agent.astream("What is 125 multiply 301?"):
                if chunk.reasoning:
                    print(chunk.reasoning, end="", flush=True)
                if chunk.content:
                    print(chunk.content, end="", flush=True)
            print()  # Newline after response
        except Exception as e:
            print(f"   ⚠️  Skipped: {e}")

        print("\n" + "=" * 50)
        print("✅ Example completed!")
        print("=" * 50)

    finally:
        os.unlink(server_path)


if __name__ == "__main__":
    asyncio.run(example())
