"""LangChainMcpAgent - An MCP coder implemented with LangChain.

This module provides a simplified MCP coder using LangChain's coder framework.
Compared to the raw FastMCP implementation, this version leverages:
- AgentExecutor for automatic tool call loops
- LangChain-MCP adapter for tool conversion
- LangChain ChatModels for unified LLM interface

Install: uv add langchain langchain-openai langchain-mcp
"""

import asyncio
from typing import Optional, List, Dict, Any, AsyncIterator
from dataclasses import dataclass, field

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_mcp import MCPToolkit
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@dataclass
class StreamChunk:
    """Represents a chunk of streamed response."""

    content: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    is_complete: bool = False


class LangChainMcpAgent:
    """An MCP coder implemented with LangChain (highly simplified).

    Example:
        # Stdio transport
        coder = LangChainMcpAgent(
            name="MCP Agent",
            command="python",
            args=["mcp_server.py"],
        )

        # Synchronous usage
        result = coder.invoke("What can you help me with?")

        # Async usage
        result = await coder.ainvoke("Tell me a story")

        # Streaming usage
        async for chunk in coder.astream("Hello"):
            print(chunk.content, end="")
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_steps: int = 10,
        # Stdio transport
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
    ):
        """Initialize the LangChain MCP coder.

        Args:
            name: Agent name
            description: Agent description
            model: Model name (e.g., "gpt-4o", defaults to env MODEL_NAME or gpt-4o)
            temperature: Sampling temperature
            max_steps: Maximum reasoning steps
            command: Command for stdio transport (e.g., "python")
            args: Arguments for the command
            env: Environment variables
        """
        self.name = name
        self.description = description
        self.max_steps = max_steps

        # Initialize LangChain ChatOpenAI model
        self.llm = ChatOpenAI(
            model=model or "gpt-4o",
            temperature=temperature,
        )

        # MCP server parameters
        self.command = command
        self.args = args or []
        self.env = env

        # Will be initialized on first use
        self._agent_executor: Optional[AgentExecutor] = None
        self._tools: Optional[List] = None
        self._session: Optional[ClientSession] = None

    def _build_prompt(self) -> ChatPromptTemplate:
        """Build the coder prompt with system message and chat history."""
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a helpful assistant with access to tools. "
                        "Use tools when needed and provide clear final answers."
                    ),
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

    async def _initialize(self) -> None:
        """Initialize MCP connection and coder executor."""
        if self._agent_executor is not None:
            return

        # Set up MCP stdio server parameters
        server_params = StdioServerParameters(
            command=self.command or "python",
            args=self.args,
            env=self.env,
        )

        # Connect to MCP server and get tools
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                toolkit = MCPToolkit(session)
                self._tools = await toolkit.get_tools()

                # Create OpenAI Tools coder
                prompt = self._build_prompt()
                agent = create_openai_tools_agent(self.llm, self._tools, prompt)

                # Create coder executor
                self._agent_executor = AgentExecutor(
                    agent=agent,
                    tools=self._tools,
                    max_iterations=self.max_steps,
                    handle_parsing_errors=True,
                )

    def invoke(self, input: str) -> str:
        """Process user input synchronously.

        Args:
            input: User query string

        Returns:
            Final response from the coder
        """
        return asyncio.run(self.ainvoke(input))

    async def ainvoke(self, input: str) -> str:
        """Process user input asynchronously.

        Args:
            input: User query string

        Returns:
            Final response from the coder
        """
        await self._initialize()

        result = await self._agent_executor.ainvoke(
            {
                "input": input,
                "chat_history": [],
            }
        )

        return result["output"]

    async def astream(self, input: str) -> AsyncIterator[StreamChunk]:
        """Stream the response asynchronously.

        Args:
            input: User input message

        Yields:
            StreamChunk objects containing content and tool calls
        """
        await self._initialize()

        async for chunk in self._agent_executor.astream(
            {
                "input": input,
                "chat_history": [],
            }
        ):
            if "output" in chunk:
                yield StreamChunk(content=chunk["output"])
            elif "actions" in chunk:
                actions = chunk["actions"]
                tool_calls = [{"name": a.tool, "input": a.tool_input} for a in actions]
                yield StreamChunk(tool_calls=tool_calls)

        yield StreamChunk(is_complete=True)

    def stream(self, input: str) -> str:
        """Stream the response synchronously.

        Args:
            input: User input message

        Returns:
            Final accumulated response
        """
        accumulated = ""

        async def _stream():
            nonlocal accumulated
            async for chunk in self.astream(input):
                if chunk.content:
                    accumulated += chunk.content
                    print(chunk.content, end="", flush=True)
                if chunk.is_complete:
                    print()

        asyncio.run(_stream())
        return accumulated



# =============================================================================
# Example Usage
# =============================================================================


async def example():
    """Example: Connect LangChainMcpAgent to a FastMCP server."""
    import tempfile
    import os

    # Create a temporary MCP server script
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

    # Write server script to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(server_code)
        server_path = f.name

    try:
        print("=" * 50)
        print("🚀 LangChain MCP Example")
        print("=" * 50)

        # Create LangChainMcpAgent
        print("\n1️⃣  Creating LangChainMcpAgent...")
        agent = LangChainMcpAgent(
            name="DemoAgent",
            args=[server_path],
        )
        print("   ✓ Agent created")

        # Test with natural language query
        print("\n2️⃣  Natural language query:")
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
