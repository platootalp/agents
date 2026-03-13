"""McpAgent - An agent that connects to MCP servers using FastMCP.

This module provides McpAgent which connects to MCP servers using the FastMCP
library (PrefectHQ/fastmcp) - a simplified, Pythonic MCP implementation.

Install: uv add fastmcp
"""

import asyncio
import json
import time
from typing import Optional, List, Dict, Any, AsyncIterator
from dataclasses import dataclass, field

from dotenv import load_dotenv

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
            command: Command for stdio transport (e.g., "python")
            args: Arguments for the command
            env: Environment variables
        """
        super().__init__(name, description, model, tools, max_steps)

        if not FASTMCP_AVAILABLE:
            raise ImportError("FastMCP not installed. Install with: uv add fastmcp")

        if server_url:
            # HTTP transport
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
            System prompt string
        """
        return (
            "You are a helpful assistant that can use tools from MCP servers "
            "to help answer user queries. Use the available tools when needed, "
            "and provide a clear final answer."
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

        # Get all available tools
        all_tools = await self._get_all_tools()

        # Initialize history
        self.message_history = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": input},
        ]

        step = 0
        while step < self.max_steps:
            if streaming:
                # Streaming mode
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

                # Add assistant message
                assistant_msg = self._build_assistant_message(final_content, tool_calls_list)
                self.message_history.append(assistant_msg)

                if not tool_calls_list:
                    return final_content.strip() if final_content else ""

                # Execute tool calls
                if print_output:
                    print(f"\n🔧 Tool Calls ({len(tool_calls_list)}):")

                results = await self._execute_tool_calls(tool_calls_list)

                for i, result in enumerate(results, 1):
                    if print_output:
                        self._print_tool_execution(result, i)

                    # Add tool response to history
                    tool_msg = self._build_tool_response_message(result.tool_call_id, result.result)
                    self.message_history.append(tool_msg)

                if print_output:
                    print()

            else:
                # Non-streaming mode
                response = self.model.generate(self.message_history, tools=all_tools or None)
                message = response.choices[0].message

                # Add assistant message
                tool_calls = self._convert_api_tool_calls(message.tool_calls)
                assistant_msg = self._build_assistant_message(message.content or "", tool_calls)
                self.message_history.append(assistant_msg)

                # Check if model wants to call tools
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        tool_name = tool_call.function.name
                        try:
                            tool_args = json.loads(tool_call.function.arguments)
                        except json.JSONDecodeError:
                            tool_args = {"query": tool_call.function.arguments}

                        result = self._call_local_tool(tool_name, tool_args)

                        if result is None:
                            result = await self._call_mcp_tool(tool_name, tool_args)

                        # Add tool response
                        tool_msg = self._build_tool_response_message(tool_call.id, str(result))
                        self.message_history.append(tool_msg)

                    step += 1
                    continue

                # Return final answer
                if message.content:
                    return message.content.strip()

            step += 1

        return "Reached maximum steps without a final answer."

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
        if not FASTMCP_AVAILABLE:
            return "FastMCP not available. Install with: uv add fastmcp"

        if not self.model:
            return "No model configured."

        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                raise RuntimeError(
                    "Cannot call invoke() from an async context. "
                    "Use 'await agent.ainvoke(input)' instead."
                )
        except RuntimeError:
            pass

        return asyncio.run(self._run_conversation_loop(input, streaming=False))

    async def ainvoke(self, input: str) -> str:
        """Async version of invoke.

        Args:
            input: User query string

        Returns:
            Final response from the agent
        """
        if not FASTMCP_AVAILABLE:
            return "FastMCP not available. Install with: uv add fastmcp"

        if not self.model:
            return "No model configured."

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
        if not self.model:
            return "No model configured."

        # Check if we're in an async context
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                raise RuntimeError(
                    "Cannot call stream() from an async context. "
                    "Use 'async for chunk in agent.astream(input)' instead."
                )
        except RuntimeError:
            pass

        # Reset or initialize history
        if reset or not hasattr(self, "message_history") or not self.message_history:
            self.message_history = [
                {"role": "system", "content": self._build_system_prompt()},
            ]
            print("\n🆕 New Conversation\n")

        # Append user message
        self.message_history.append({"role": "user", "content": input})
        print(f"👤 User: {input}\n")

        # Run streaming loop with output printing
        return asyncio.run(self._run_conversation_loop(input, streaming=True, print_output=True))

    async def astream(self, input: str, reset: bool = False) -> AsyncIterator[StreamChunk]:
        """Async streaming response generator.

        Args:
            input: User input message
            reset: If True, reset conversation history before processing

        Yields:
            StreamChunk objects containing content, reasoning, and tool calls
        """
        if not self.model:
            yield StreamChunk(content="No model configured.", is_complete=True)
            return

        if not FASTMCP_AVAILABLE:
            yield StreamChunk(
                content="FastMCP not available. Install with: uv add fastmcp",
                is_complete=True,
            )
            return

        # Reset or initialize history
        if reset or not hasattr(self, "message_history") or not self.message_history:
            self.message_history = [
                {"role": "system", "content": self._build_system_prompt()},
            ]

        self.message_history.append({"role": "user", "content": input})

        if not self._connected:
            await self.connect()

        all_tools = await self._get_all_tools()

        step = 0
        while step < self.max_steps:
            accumulated_content = ""
            accumulated_reasoning = ""
            accumulated_tool_calls: Dict[int, Dict[str, Any]] = {}

            stream = self.model.stream(self.message_history, tools=all_tools or None)

            for chunk in stream:
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                # Handle content
                if delta.content:
                    accumulated_content += delta.content
                    yield StreamChunk(content=delta.content)

                # Handle reasoning
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    accumulated_reasoning += delta.reasoning_content
                    yield StreamChunk(reasoning=delta.reasoning_content)

                # Handle tool calls
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

            # Add assistant message to history
            assistant_msg = self._build_assistant_message(accumulated_content, tool_calls_list)
            self.message_history.append(assistant_msg)

            if tool_calls_list:
                # Execute tool calls and yield results
                results = await self._execute_tool_calls(tool_calls_list)

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

                    # Add to history
                    tool_msg = self._build_tool_response_message(result.tool_call_id, result.result)
                    self.message_history.append(tool_msg)

                step += 1
            else:
                # No tool calls, we're done
                yield StreamChunk(is_complete=True)
                return

        yield StreamChunk(
            content="Reached maximum steps without a final answer.",
            is_complete=True,
        )


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
