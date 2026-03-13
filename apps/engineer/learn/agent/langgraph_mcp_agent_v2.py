"""LangGraphMcpAgent v2 - Using official langchain-mcp-adapters.

This is the recommended approach using LangGraph's official MCP integration:
- langchain-mcp-adapters package
- MultiServerMCPClient for connection management
- Native ToolNode compatibility

Install: uv add langgraph langchain-openai langchain-mcp-adapters

Official Docs: https://docs.langchain.com/oss/python/langchain/mcp
"""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Literal
from typing_extensions import Annotated

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.managed import IsLastStep
from langchain_mcp_adapters.client import MultiServerMCPClient


# =============================================================================
# State Definitions (TypedDict for LangGraph)
# =============================================================================


@dataclass
class InputState:
    """Input state for the agent."""

    messages: Annotated[Sequence[AnyMessage], add_messages] = field(default_factory=list)


@dataclass
class State(InputState):
    """Complete agent state with managed fields."""

    is_last_step: IsLastStep = field(default=False)


# =============================================================================
# Agent Node
# =============================================================================


class AgentNode:
    """Agent node that calls the LLM with tool binding."""

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
    ):
        self.llm = ChatOpenAI(
            model=model or "gpt-4o",
            temperature=temperature,
        )
        self.system_prompt = system_prompt or (
            "You are a helpful assistant with access to tools. "
            "Use tools when needed and provide clear final answers."
        )

    async def __call__(
        self,
        state: State,
        config: Optional[RunnableConfig] = None,
    ) -> Dict[str, List[AnyMessage]]:
        """Call LLM with current messages and available tools."""
        # Bind tools from config (set during graph compilation)
        tools = config.get("configurable", {}).get("tools", [])
        model_with_tools = self.llm.bind_tools(tools)

        # Prepare messages
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(state.messages)

        # Get response
        response = await model_with_tools.ainvoke(messages, config)

        # Handle last step case
        if state.is_last_step and isinstance(response, AIMessage) and response.tool_calls:
            return {
                "messages": [
                    AIMessage(
                        id=response.id,
                        content="I ran out of steps. Here's my best answer based on available information.",
                    )
                ]
            }

        return {"messages": [response]}


# =============================================================================
# Graph Factory
# =============================================================================


async def create_mcp_agent_graph(
    client: MultiServerMCPClient,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_steps: int = 10,
):
    """Create a LangGraph agent with MCP tools.

    Args:
        client: Configured MultiServerMCPClient
        model: LLM model name
        temperature: Sampling temperature
        max_steps: Maximum recursion steps

    Returns:
        Compiled StateGraph with MCP tools integrated
    """
    # Get tools from MCP servers
    tools = await client.get_tools()

    # Create nodes
    agent_node = AgentNode(model=model, temperature=temperature)
    tools_node = ToolNode(tools)

    # Build graph
    builder = StateGraph(State, input=InputState)

    # Add nodes
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tools_node)

    # Define edges
    builder.add_edge("__start__", "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")

    # Compile with tool configuration
    graph = builder.compile()

    return graph, tools


# =============================================================================
# Main Agent Class
# =============================================================================


class LangGraphMcpAgentV2:
    """MCP Agent using official LangGraph MCP adapters.

    This is the RECOMMENDED implementation using langchain-mcp-adapters.

    Example:
        # Single server
        agent = LangGraphMcpAgentV2({
            "math": {
                "transport": "stdio",
                "command": "python",
                "args": ["math_server.py"],
            }
        })

        # Multiple servers
        agent = LangGraphMcpAgentV2({
            "math": {"transport": "stdio", "command": "python", "args": ["math.py"]},
            "weather": {"transport": "http", "url": "http://localhost:8000/mcp"},
        })

        # Run
        result = await agent.ainvoke("What is 125 * 301?")
    """

    def __init__(
        self,
        servers: Dict[str, Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_steps: int = 10,
    ):
        """Initialize the agent with MCP server configurations.

        Args:
            servers: MCP server configurations, keyed by server name
                Example:
                {
                    "math": {
                        "transport": "stdio",
                        "command": "python",
                        "args": ["math_server.py"],
                    },
                    "weather": {
                        "transport": "http",
                        "url": "http://localhost:8000/mcp",
                    }
                }
            model: LLM model name (default: gpt-4o)
            temperature: Sampling temperature
            max_steps: Maximum reasoning steps
        """
        self.servers = servers
        self.model = model
        self.temperature = temperature
        self.max_steps = max_steps

        self._client: Optional[MultiServerMCPClient] = None
        self._graph: Optional[Any] = None
        self._tools: Optional[List] = None

    async def _initialize(self) -> None:
        """Initialize MCP client and graph."""
        if self._graph is not None:
            return

        # Create MCP client with server configurations
        self._client = MultiServerMCPClient(self.servers)

        # Create graph with MCP tools
        self._graph, self._tools = await create_mcp_agent_graph(
            client=self._client,
            model=self.model,
            temperature=self.temperature,
            max_steps=self.max_steps,
        )

    def invoke(self, input: str) -> str:
        """Run agent synchronously.

        Args:
            input: User query

        Returns:
            Final response string
        """
        return asyncio.run(self.ainvoke(input))

    async def ainvoke(self, input: str) -> str:
        """Run agent asynchronously.

        Args:
            input: User query

        Returns:
            Final response string
        """
        await self._initialize()

        # Prepare state
        initial_state = InputState(messages=[HumanMessage(content=input)])

        # Run graph
        config = {"recursion_limit": self.max_steps + 5}
        result = await self._graph.ainvoke(initial_state, config=config)

        # Extract final message
        last_message = result["messages"][-1]
        if isinstance(last_message, AIMessage):
            return last_message.content or ""
        return str(last_message.content)

    async def astream(self, input: str):
        """Stream agent execution events.

        Args:
            input: User query

        Yields:
            Execution events showing each step
        """
        await self._initialize()

        initial_state = InputState(messages=[HumanMessage(content=input)])
        config = {"recursion_limit": self.max_steps + 5}

        async for event in self._graph.astream(initial_state, config=config):
            yield event

    def stream(self, input: str) -> str:
        """Stream agent execution synchronously.

        Args:
            input: User query

        Returns:
            Final accumulated response
        """
        accumulated = ""

        async def _stream():
            nonlocal accumulated
            async for event in self.astream(input):
                if "agent" in event:
                    messages = event["agent"].get("messages", [])
                    for msg in messages:
                        if isinstance(msg, AIMessage) and msg.content:
                            print(msg.content, end="", flush=True)
                            accumulated += msg.content
                elif "tools" in event:
                    print("\n[Tool execution]", flush=True)

        asyncio.run(_stream())
        return accumulated


# =============================================================================
# Simplified Factory Function
# =============================================================================


async def create_mcp_agent(
    server_config: Dict[str, Dict[str, Any]],
    model: str = "gpt-4o",
    temperature: float = 0.0,
) -> LangGraphMcpAgentV2:
    """Factory function to create an MCP agent.

    Args:
        server_config: MCP server configurations
        model: LLM model name
        temperature: Sampling temperature

    Returns:
        Initialized LangGraphMcpAgentV2
    """
    agent = LangGraphMcpAgentV2(
        servers=server_config,
        model=model,
        temperature=temperature,
    )
    return agent


# =============================================================================
# Comparison with v1 (manual implementation)
# =============================================================================

"""
V1 (手动工具转换) vs V2 (官方适配器):

V1 (~350 lines core):
- 手动 MCPToolManager 类
- 手动工具转换 (MCP → LangChain)
- 手动 session 管理
- 需要处理 stdio_client, ClientSession 等

V2 (~150 lines core):
- 使用 MultiServerMCPClient (一行配置)
- 自动工具转换
- 自动连接管理
- 支持多服务器
- 内置重试、错误处理

代码节省: ~60% (350 → 150 lines)
功能增强:
+ 支持多服务器
+ 更好的错误处理
+ 官方维护
+ 传输方式自动处理
+ 拦截器支持 (认证、日志等)
"""


# =============================================================================
# Example Usage
# =============================================================================


async def example():
    """Example: Using LangGraph MCP Agent v2."""
    import tempfile
    import os

    # Create temp MCP server
    server_code = '''
from fastmcp import FastMCP

mcp = FastMCP("Demo")

@mcp.tool
def calculator(operation: str, a: float, b: float) -> str:
    """Perform math operations."""
    ops = {
        "add": lambda: a + b,
        "subtract": lambda: a - b,
        "multiply": lambda: a * b,
        "divide": lambda: a / b if b != 0 else "Error: Division by zero",
    }
    result = ops.get(operation, lambda: "Unknown operation")()
    return f"Result: {result}"

@mcp.tool
def weather(city: str) -> str:
    """Get weather."""
    return f"Weather in {city}: Sunny, 25°C"

if __name__ == "__main__":
    mcp.run()
'''

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(server_code)
        server_path = f.name

    try:
        print("=" * 60)
        print("🚀 LangGraph MCP Agent v2 (Official Adapters)")
        print("=" * 60)

        # Create agent with server config
        print("\n1️⃣  Creating agent with MCP config...")
        agent = LangGraphMcpAgentV2(
            {
                "demo": {
                    "transport": "stdio",
                    "command": "python",
                    "args": [server_path],
                }
            }
        )
        print("   ✓ Agent created")

        # Run query
        print("\n2️⃣  Running query:")
        print(f"   Q: 'What is 125 multiplied by 301?'")
        print(f"   A: ", end="", flush=True)

        result = await agent.ainvoke("What is 125 multiplied by 301?")
        print(result)

        print("\n" + "=" * 60)
        print("✅ Example completed!")
        print("=" * 60)

    finally:
        os.unlink(server_path)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    asyncio.run(example())
