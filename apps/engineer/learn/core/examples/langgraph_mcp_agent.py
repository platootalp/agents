"""LangGraphMcpAgent - An MCP coder built with LangGraph.

This module provides an MCP coder using LangGraph's state machine approach.
LangGraph offers more control over coder workflow compared to AgentExecutor:
- Explicit state management
- Conditional routing
- Human-in-the-loop support
- Better visibility into execution steps

Architecture:
    ┌─────────────┐
    │    Start    │
    └──────┬──────┘
           ▼
    ┌─────────────┐     No tool calls      ┌──────┐
    │  Agent Node │ ─────────────────────► │  End │
    │  (call_llm) │                        └──────┘
    └──────┬──────┘
           │ Has tool calls
           ▼
    ┌─────────────┐
    │  Tools Node │
    │  (ToolNode) │
    └──────┬──────┘
           │
           └──────────────────────────────► (back to Agent)

Install: uv add langgraph langchain-openai mcp
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Literal
from typing_extensions import Annotated

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.managed import IsLastStep
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# =============================================================================
# State Definitions
# =============================================================================


@dataclass
class InputState:
    """Input state for the coder.

    This is the minimal interface for starting a conversation.
    """

    messages: Annotated[Sequence[AnyMessage], add_messages] = field(default_factory=list)
    """Conversation messages history."""


@dataclass
class State(InputState):
    """Complete coder state.

    Extends InputState with additional managed fields.
    """

    is_last_step: IsLastStep = field(default=False)
    """Managed flag indicating if this is the last allowed step."""


# =============================================================================
# MCP Tool Manager
# =============================================================================


class MCPToolManager:
    """Manages MCP server connection and tool conversion.

    Converts MCP tools to LangChain format for use with ToolNode.
    """

    def __init__(
        self,
        command: str = "python",
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
    ):
        self.command = command
        self.args = args or []
        self.env = env
        self._tools: List[Any] = []
        self._session: Optional[ClientSession] = None

    async def initialize(self) -> List[Any]:
        """Connect to MCP server and retrieve tools.

        Returns:
            List of LangChain-compatible tools.
        """
        if self._tools:
            return self._tools

        server_params = StdioServerParameters(
            command=self.command,
            args=self.args,
            env=self.env,
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # Get MCP tools
                mcp_tools = await session.list_tools()

                # Convert to LangChain tools
                self._tools = [self._convert_mcp_tool(session, tool) for tool in mcp_tools.tools]

                return self._tools

    def _convert_mcp_tool(self, session: ClientSession, mcp_tool: Any) -> Any:
        """Convert an MCP tool to LangChain format.

        Args:
            session: Active MCP client session
            mcp_tool: MCP tool definition

        Returns:
            LangChain StructuredTool
        """
        from langchain_core.tools import StructuredTool
        import inspect

        # Build the function dynamically
        async def tool_func(**kwargs):
            """Dynamic tool function."""
            result = await session.call_tool(mcp_tool.name, kwargs)
            # Extract text content from result
            content_parts = []
            for item in result.content:
                if hasattr(item, "text"):
                    content_parts.append(item.text)
                else:
                    content_parts.append(str(item))
            return "\n".join(content_parts) if content_parts else "No result"

        # Set function name and doc
        tool_func.__name__ = mcp_tool.name
        tool_func.__doc__ = mcp_tool.description or f"Call {mcp_tool.name} tool"

        # Create StructuredTool
        return StructuredTool.from_function(
            func=tool_func,
            name=mcp_tool.name,
            description=mcp_tool.description or "",
            args_schema=mcp_tool.inputSchema,
        )


# =============================================================================
# Graph Nodes
# =============================================================================


class AgentNode:
    """Agent node that calls the LLM."""

    def __init__(self, model: Optional[str] = None, temperature: float = 0.0):
        self.llm = ChatOpenAI(
            model=model or "gpt-4o",
            temperature=temperature,
        )
        self.system_message = (
            "You are a helpful assistant with access to tools. "
            "Use tools when needed and provide clear final answers. "
            "When you need to use a tool, call it directly without describing your intention first."
        )

    async def __call__(
        self,
        state: State,
        config: Optional[RunnableConfig] = None,
    ) -> Dict[str, List[AnyMessage]]:
        """Call the LLM with current state.

        Args:
            state: Current conversation state
            config: Runtime configuration

        Returns:
            Dictionary with messages to add to state
        """
        # Prepare messages with system prompt
        messages = [{"role": "system", "content": self.system_message}]
        messages.extend(state.messages)

        # Get LLM response
        response = await self.llm.ainvoke(messages, config)

        # Handle last step case
        if state.is_last_step and isinstance(response, AIMessage) and response.tool_calls:
            return {
                "messages": [
                    AIMessage(
                        id=response.id,
                        content="Sorry, I ran out of steps. Let me provide my best answer based on what I know.",
                    )
                ]
            }

        return {"messages": [response]}


def create_tools_node(tools: List[Any]) -> ToolNode:
    """Create a ToolNode with MCP tools.

    Args:
        tools: List of LangChain tools

    Returns:
        Configured ToolNode
    """
    return ToolNode(tools)


# =============================================================================
# Router
# =============================================================================


def route_agent_output(state: State) -> Literal["__end__", "tools"]:
    """Route based on coder's output.

    Args:
        state: Current state

    Returns:
        Next node name
    """
    last_message = state.messages[-1]

    if not isinstance(last_message, AIMessage):
        raise ValueError(f"Expected AIMessage, got {type(last_message).__name__}")

    # If no tool calls, we're done
    if not last_message.tool_calls:
        return "__end__"

    # Otherwise, execute tools
    return "tools"


# =============================================================================
# Graph Builder
# =============================================================================


def create_mcp_graph(
    tools: List[Any],
    model: Optional[str] = None,
    temperature: float = 0.0,
) -> StateGraph:
    """Create the MCP coder graph.

    Args:
        tools: List of LangChain tools from MCP
        model: Model name
        temperature: Sampling temperature

    Returns:
        Compiled StateGraph
    """
    # Create nodes
    agent_node = AgentNode(model=model, temperature=temperature)
    tools_node = create_tools_node(tools)

    # Build graph
    builder = StateGraph(State, input=InputState)

    # Add nodes
    builder.add_node("coder", agent_node)
    builder.add_node("tools", tools_node)

    # Set entry point
    builder.add_edge("__start__", "coder")

    # Add conditional edges from coder
    builder.add_conditional_edges(
        "coder",
        route_agent_output,
    )

    # Tools always go back to coder
    builder.add_edge("tools", "coder")

    # Compile graph
    return builder.compile()


# =============================================================================
# Main Agent Class
# =============================================================================


class LangGraphMcpAgent:
    """MCP Agent built with LangGraph.

    Example:
        # Initialize with MCP server
        coder = LangGraphMcpAgent(
            args=["mcp_server.py"],
        )

        # Run synchronously
        result = coder.invoke("What is 125 * 301?")

        # Run asynchronously
        result = await coder.ainvoke("Tell me a story")

        # Stream responses
        async for chunk in coder.astream("Hello"):
            print(chunk.content, end="")
    """

    def __init__(
        self,
        name: str = "MCP Agent",
        description: str = "",
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_steps: int = 10,
        command: str = "python",
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
    ):
        """Initialize the LangGraph MCP coder.

        Args:
            name: Agent name
            description: Agent description
            model: LLM model name
            temperature: Sampling temperature
            max_steps: Maximum reasoning steps
            command: Command for MCP server
            args: Arguments for MCP server
            env: Environment variables
        """
        self.name = name
        self.description = description
        self.max_steps = max_steps
        self.model = model
        self.temperature = temperature

        # MCP tool manager
        self._tool_manager = MCPToolManager(
            command=command,
            args=args,
            env=env,
        )

        # Will be initialized on first use
        self._graph: Optional[Any] = None
        self._tools: Optional[List[Any]] = None

    async def _initialize(self) -> None:
        """Initialize tools and graph."""
        if self._graph is not None:
            return

        # Initialize MCP tools
        self._tools = await self._tool_manager.initialize()

        # Create graph
        self._graph = create_mcp_graph(
            tools=self._tools,
            model=self.model,
            temperature=self.temperature,
        )

    def invoke(self, input: str) -> str:
        """Run coder synchronously.

        Args:
            input: User query

        Returns:
            Final response
        """
        return asyncio.run(self.ainvoke(input))

    async def ainvoke(self, input: str) -> str:
        """Run coder asynchronously.

        Args:
            input: User query

        Returns:
            Final response
        """
        await self._initialize()

        # Prepare initial state
        initial_state = InputState(messages=[HumanMessage(content=input)])

        # Run graph
        config = {"recursion_limit": self.max_steps + 5}
        result = await self._graph.ainvoke(initial_state, config=config)

        # Extract final response
        last_message = result["messages"][-1]
        if isinstance(last_message, AIMessage):
            return last_message.content or ""

        return str(last_message.content)

    async def astream(self, input: str):
        """Stream coder execution.

        Args:
            input: User query

        Yields:
            Execution events (messages, tool calls, etc.)
        """
        await self._initialize()

        initial_state = InputState(messages=[HumanMessage(content=input)])

        config = {"recursion_limit": self.max_steps + 5}

        async for event in self._graph.astream(initial_state, config=config):
            yield event

    def stream(self, input: str) -> str:
        """Stream coder execution synchronously.

        Args:
            input: User query

        Returns:
            Final accumulated response
        """
        accumulated = ""

        async def _stream():
            nonlocal accumulated
            async for event in self.astream(input):
                # Handle different event types
                if "coder" in event:
                    messages = event["coder"].get("messages", [])
                    for msg in messages:
                        if isinstance(msg, AIMessage) and msg.content:
                            print(msg.content, end="", flush=True)
                            accumulated += msg.content
                elif "tools" in event:
                    print("\n🔧 Tool execution...", flush=True)

        asyncio.run(_stream())
        return accumulated


# =============================================================================
# Comparison
# =============================================================================

"""
LANGGRAPH VS LANGCHAIN AgentExecutor:

LangGraph version (~200 lines of core coder):
- Explicit state machine with StateGraph
- Clear node definitions (coder, tools)
- Conditional routing logic
- Better visibility into execution
- Easier to add human-in-the-loop
- Support for persistence/checkpointing

LangChain AgentExecutor version (~150 lines):
- Black-box loop execution
- Less control over flow
- Simpler for basic use cases

Original McpAgent (957 lines):
- Manual everything

SAVINGS COMPARED TO ORIGINAL:
- LangGraph: ~75% reduction (957 → ~240 lines total)
- LangChain: ~80% reduction (957 → ~190 lines total)

LANGGRAPH ADVANTAGES:
+ More explicit control
+ Better for complex workflows
+ Human-in-the-loop support
+ State persistence
+ Easier debugging
+ Visual graph representation

LANGCHAIN ADVANTAGES:
+ Simpler for basic cases
+ Less boilerplate
+ More mature ecosystem
"""


# =============================================================================
# Example
# =============================================================================


async def example():
    """Example: Run LangGraph MCP coder."""
    import tempfile
    import os

    # Create temporary MCP server
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
    return f"Weather in {city}: Sunny, 25°C"

if __name__ == "__main__":
    mcp.run()
'''

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(server_code)
        server_path = f.name

    try:
        print("=" * 50)
        print("🚀 LangGraph MCP Example")
        print("=" * 50)

        # Create coder
        print("\n1️⃣  Creating LangGraphMcpAgent...")
        agent = LangGraphMcpAgent(
            name="DemoAgent",
            args=[server_path],
            max_steps=5,
        )
        print("   ✓ Agent created")

        # Test query
        print("\n2️⃣  Testing query:")
        print(f"   Query: 'What is 125 multiplied by 301?'")
        print(f"   Response: ", end="", flush=True)

        result = await agent.ainvoke("What is 125 multiplied by 301?")
        print(result)

        print("\n" + "=" * 50)
        print("✅ Example completed!")
        print("=" * 50)

    finally:
        os.unlink(server_path)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    asyncio.run(example())
