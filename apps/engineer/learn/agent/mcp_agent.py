"""McpAgent - An agent that can connect to MCP servers and use their tools.

MCP (Model Context Protocol) is a protocol for standardizing AI model interactions
with external tools and data sources. This agent extends ToolUseAgent to support
connecting to MCP servers and utilizing their tools.
"""

import asyncio
import json
from typing import Optional, List, Dict, Any
from contextlib import AsyncExitStack

from apps.engineer.learn.agent.core.model import Model
from apps.engineer.learn.agent.core.tool import Tool
from apps.engineer.learn.agent.tool_use_agent import ToolUseAgent

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.types import TextContent, Tool as MCPTool

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


class McpAgent(ToolUseAgent):
    """An agent that connects to MCP servers and uses their tools.

    This agent extends ToolUseAgent to support MCP protocol, allowing it to:
    - Connect to MCP servers via stdio or other transports
    - Discover and use tools exposed by MCP servers
    - Interact with LLMs using MCP-provided tools

    Example:
        ```python
        # Connect to an MCP server via stdio
        agent = McpAgent(
            name="MCP Agent",
            model=Model(),
            command="python",
            args=["mcp_server.py"],
        )

        result = agent.invoke("What can you help me with?")
        ```
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        model: Optional[Model] = None,
        tools: Optional[List[Tool]] = None,
        max_steps: int = 10,
        # MCP-specific parameters
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
            command: Command to start MCP server (e.g., "python", "npx")
            args: Arguments for the command (e.g., ["server.py"])
            env: Environment variables for the MCP server
        """
        super().__init__(name, description, model, tools, max_steps)

        if not MCP_AVAILABLE:
            raise ImportError("MCP SDK not installed. Install with: uv add mcp")

        self.command = command
        self.args = args or []
        self.env = env

        # MCP client state
        self._session: Optional[ClientSession] = None
        self._exit_stack: Optional[AsyncExitStack] = None
        self._mcp_tools: List[MCPTool] = []

        # System prompt updated for MCP context
        self.SYSTEM_PROMPT = (
            "You are a helpful assistant that can use tools from MCP servers "
            "to help answer user queries. Use the available tools when needed, "
            "and provide a clear final answer."
        )

    async def connect(self) -> None:
        """Connect to the MCP server.

        This method establishes a connection to the MCP server using stdio
        transport and initializes the session.
        """
        if not self.command:
            raise ValueError("No command specified for MCP server connection")

        self._exit_stack = AsyncExitStack()

        # Create server parameters
        server_params = StdioServerParameters(
            command=self.command,
            args=self.args,
            env=self.env,
        )

        # Connect to the server
        stdio_transport = await self._exit_stack.enter_async_context(stdio_client(server_params))
        read_stream, write_stream = stdio_transport

        # Create session
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )

        # Initialize the session
        await self._session.initialize()

        # Load available tools from MCP server
        await self._load_mcp_tools()

    async def _load_mcp_tools(self) -> None:
        """Load tools from the MCP server."""
        if not self._session:
            raise RuntimeError("Not connected to MCP server")

        tools_result = await self._session.list_tools()
        self._mcp_tools = tools_result.tools

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

        # Add MCP tools
        for mcp_tool in self._mcp_tools:
            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": mcp_tool.name,
                        "description": mcp_tool.description or "",
                        "parameters": mcp_tool.inputSchema,
                    },
                }
            )

        return openai_tools if openai_tools else None

    async def _call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Call a tool on the MCP server.

        Args:
            tool_name: Name of the MCP tool to call
            arguments: Tool arguments as a dictionary

        Returns:
            Tool result as a string
        """
        if not self._session:
            raise RuntimeError("Not connected to MCP server")

        try:
            result = await self._session.call_tool(tool_name, arguments)

            # Extract text content from result
            content_parts = []
            for content in result.content:
                if isinstance(content, TextContent):
                    content_parts.append(content.text)
                else:
                    content_parts.append(str(content))

            return "\n".join(content_parts) if content_parts else "No result"

        except Exception as e:
            return f"MCP tool error: {type(e).__name__}: {e}"

    def invoke(self, input: str) -> str:
        """Process user input using MCP tools and LLM.

        Args:
            input: User query string

        Returns:
            Final response from the agent
        """
        if not MCP_AVAILABLE:
            return "MCP SDK not available. Install with: uv add mcp"

        if not self.model:
            return "No model configured."

        # Run the async workflow
        return asyncio.run(self._invoke_async(input))

    async def _invoke_async(self, input: str) -> str:
        """Async implementation of invoke."""
        # Connect if not already connected
        if not self._session:
            await self.connect()

        # Initialize history
        self.message_history = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": input},
        ]

        step = 0
        while step < self.max_steps:
            # Get available tools in OpenAI format
            openai_tools = self._get_openai_tools()

            # Generate response with tool support
            response = self.model.generate(self.message_history, tools=openai_tools)
            message = response.choices[0].message

            # Add assistant message to history
            assistant_msg: Dict[str, Any] = {
                "role": "assistant",
                "content": message.content or "",
            }
            if message.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ]
            self.message_history.append(assistant_msg)

            # Check if model wants to call tools
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        tool_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        tool_args = {"query": tool_call.function.arguments}

                    # Check if it's a local tool or MCP tool
                    tool_result = self._call_local_tool(tool_name, tool_args)

                    if tool_result is None:
                        # Try MCP tool
                        tool_result = await self._call_mcp_tool(tool_name, tool_args)

                    # Add tool response to history
                    self.message_history.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(tool_result),
                        }
                    )

                step += 1
                continue

            # Return the final answer
            if message.content:
                return message.content.strip()

            step += 1

        return "Reached maximum steps without a final answer."

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
                        # Extract query from args or use full args
                        query = args.get("query", "") if isinstance(args, dict) else str(args)
                        return t.func(query)
                    except Exception as e:
                        return f"Tool {t.name} error: {e}"
                return t.description or f"No callable for tool {t.name}"
        return None

    async def disconnect(self) -> None:
        """Disconnect from the MCP server and cleanup resources."""
        if self._exit_stack:
            await self._exit_stack.aclose()
            self._exit_stack = None
            self._session = None
            self._mcp_tools = []

    def __del__(self):
        """Cleanup when the agent is garbage collected."""
        if self._session or self._exit_stack:
            try:
                asyncio.run(self.disconnect())
            except Exception:
                pass  # Ignore cleanup errors during garbage collection

    def stream(self, input: str) -> str:
        """Stream processing is not yet implemented for MCP agent."""
        # For now, delegate to invoke
        return self.invoke(input)

    def get_mcp_tools(self) -> List[MCPTool]:
        """Get the list of MCP tools available on the connected server.

        Returns:
            List of MCP tool definitions
        """
        return self._mcp_tools.copy()

    async def is_connected(self) -> bool:
        """Check if the agent is connected to an MCP server.

        Returns:
            True if connected, False otherwise
        """
        return self._session is not None
