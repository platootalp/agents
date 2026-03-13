"""McpAgent - An agent that connects to MCP servers using FastMCP.

This module provides McpAgent which connects to MCP servers using the FastMCP
library (PrefectHQ/fastmcp) - a simplified, Pythonic MCP implementation.

Install: uv add fastmcp
"""

import asyncio
import json
import re
import time
from typing import Optional, List, Dict, Any, AsyncIterator
from dataclasses import dataclass, field

from dotenv import load_dotenv
from fastmcp.client import SSETransport

from apps.engineer.learn.agent.core.model import Model
from apps.engineer.learn.agent.core.tool import Tool
from apps.engineer.learn.agent.tool_use_agent import ToolUseAgent

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


@dataclass
class ToolCallResult:
    """Result of a tool call execution."""

    tool_call_id: str
    tool_name: str
    result: str
    elapsed_ms: float
    args: str = ""  # JSON string of arguments for display


class McpAgent(ToolUseAgent):
    """An agent that connects to MCP servers and uses their tools via FastMCP.

    This agent extends ToolUseAgent to support MCP protocol using FastMCP's
    simplified Client API.

    Example:
        ```python
        # Stdio transport (default, most reliable)
        agent = McpAgent(
            name="MCP Agent",
            model=Model(),
            command="python",
            args=["mcp_server.py"],
        )

        # Or HTTP transport
        agent = McpAgent(
            name="MCP Agent",
            model=Model(),
            server_url="http://localhost:8000/mcp",
        )

        # Synchronous usage
        result = agent.invoke("What can you help me with?")

        # Streaming usage
        result = agent.stream("Tell me a story")

        # Async usage
        result = await agent.ainvoke("What can you help me with?")

        # Async streaming
        async for chunk in agent.astream("Tell me a story"):
            print(chunk.content, end="")
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

    def _format_tool_args(self, args: str) -> str:
        """Format tool arguments for display.

        Args:
            args: JSON string of arguments

        Returns:
            Formatted string
        """
        if args:
            try:
                return json.dumps(json.loads(args), indent=4)
            except:
                return args
        return ""

    def _print_tool_execution(self, result: ToolCallResult, index: int = 0) -> None:
        """Print tool execution details.

        Args:
            result: ToolCallResult to display
            index: Optional index for display
        """
        prefix = f"  [{index}] " if index > 0 else "  "
        print(f"{prefix}{result.tool_name}")

        # Try to format arguments if available
        if result.args:
            args_pretty = self._format_tool_args(result.args)
            if args_pretty:
                print(f"      Args: {args_pretty}")

        print(f"      Executing...", end="", flush=True)
        print(f" ✓ Done ({result.elapsed_ms:.0f}ms)")

        result_display = result.result[:300] + "..." if len(result.result) > 300 else result.result
        print(f"      Result: {result_display}")

    def _convert_api_tool_calls(self, api_tool_calls: Optional[List[Any]]) -> List[Dict[str, Any]]:
        """Convert API tool call objects to dict format.

        Args:
            api_tool_calls: List of tool call objects from API response

        Returns:
            List of tool call dicts
        """
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

    def _parse_tool_calls_from_content(self, content: Optional[str]) -> List[Dict[str, Any]]:
        """Parse tool calls from content when LLM outputs them in text format.

        Some models output tool calls in content field like:
        'functions.weather:0 {"city": "Beijing"}'
        or '<function_calls>...</function_calls>'

        Args:
            content: The content string to parse

        Returns:
            List of parsed tool call dicts
        """
        if not content:
            return []

        tool_calls = []

        # Pattern 1: functions.name:index {json_args}
        # Example: functions.weather:0 {"city": "Beijing"}
        # Also handles: functions.resolve-library-id:0 <marker> {"libraryName": "..."} <marker>
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
        # Example: <function_calls><invoke name="weather"><parameter name="city">Beijing</parameter></invoke></function_calls>
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
                    # Parse parameters
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

    def _build_assistant_message(
        self, content: str, tool_calls: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Build an assistant message for conversation history.

        Args:
            content: Message content
            tool_calls: Optional list of tool calls

        Returns:
            Assistant message dict
        """
        msg: Dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        return msg

    def _build_tool_response_message(self, tool_call_id: str, content: str) -> Dict[str, Any]:
        """Build a tool response message for conversation history.

        Args:
            tool_call_id: ID of the tool call
            content: Response content

        Returns:
            Tool response message dict
        """
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": str(content),
        }

    def _accumulate_tool_calls(self, tool_call_deltas: List[Any]) -> Dict[int, Dict[str, Any]]:
        """Accumulate tool call deltas into complete tool call objects.

        Args:
            tool_call_deltas: List of tool call delta objects from streaming

        Returns:
            Dictionary mapping index to accumulated tool call dict
        """
        accumulated: Dict[int, Dict[str, Any]] = {}
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

    def _init_conversation(self, input: str, reset: bool = False) -> None:
        """Initialize conversation history.

        Args:
            input: User input
            reset: Whether to force reset the conversation
        """
        if reset or not hasattr(self, "message_history") or not self.message_history:
            self.message_history = [
                {"role": "system", "content": self._build_system_prompt()},
            ]

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
        self._init_conversation(input)
        self.message_history.append({"role": "user", "content": input})

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
        final_content = ""
        accumulated_tool_calls: Dict[int, Dict[str, Any]] = {}

        stream = self.model.stream(self.message_history, tools=all_tools or None)

        for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            if delta.content:
                final_content += delta.content
                if print_output:
                    print(delta.content, end="", flush=True)

            if delta.tool_calls:
                accumulated_tool_calls.update(self._accumulate_tool_calls(delta.tool_calls))

        tool_calls_list = list(accumulated_tool_calls.values())

        # If no tool_calls from API, try parsing from content
        if not tool_calls_list and final_content:
            parsed_calls = self._parse_tool_calls_from_content(final_content)
            if parsed_calls:
                tool_calls_list = parsed_calls
                if print_output:
                    print(f"\n[Parsed {len(tool_calls_list)} tool calls from content]")

        assistant_msg = self._build_assistant_message(final_content, tool_calls_list)
        self.message_history.append(assistant_msg)

        # Execute tools if any, otherwise return final answer
        if tool_calls_list:
            await self._execute_and_record_tools(tool_calls_list, print_output)
            return None

        return final_content.strip() if final_content else ""

    async def _run_non_streaming_step(self, all_tools: List[Dict[str, Any]]) -> Optional[str]:
        """Run one step of the conversation in non-streaming mode.

        Args:
            all_tools: Available tools

        Returns:
            Final response if done, None to continue
        """
        response = self.model.generate(self.message_history, tools=all_tools or None)
        message = response.choices[0].message

        tool_calls = self._convert_api_tool_calls(message.tool_calls)

        # If no tool_calls from API, try parsing from content
        if not tool_calls and message.content:
            parsed_calls = self._parse_tool_calls_from_content(message.content)
            if parsed_calls:
                tool_calls = parsed_calls

        assistant_msg = self._build_assistant_message(message.content or "", tool_calls)
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

            tool_msg = self._build_tool_response_message(result.tool_call_id, result.result)
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
        self._init_conversation(input, reset)

        if reset or not self.message_history[1:]:
            print("\n🆕 New Conversation\n")

        self.message_history.append({"role": "user", "content": input})
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

        self._init_conversation(input, reset)
        self.message_history.append({"role": "user", "content": input})

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
                accumulated_tool_calls.update(self._accumulate_tool_calls(delta.tool_calls))

        tool_calls_list = list(accumulated_tool_calls.values())

        # If no tool_calls from API, try parsing from content
        if not tool_calls_list and accumulated_content:
            parsed_calls = self._parse_tool_calls_from_content(accumulated_content)
            if parsed_calls:
                tool_calls_list = parsed_calls

        assistant_msg = self._build_assistant_message(accumulated_content, tool_calls_list)
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

            tool_msg = self._build_tool_response_message(result.tool_call_id, result.result)
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
    uv run python learn/agent/mcp_agent.py
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
