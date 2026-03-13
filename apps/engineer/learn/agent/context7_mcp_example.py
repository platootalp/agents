"""Context7 MCP Remote Service Example.

This example shows how to connect to the Context7 MCP service
using the standard McpAgent with custom headers for API key authentication.

Usage:
    export CONTEXT7_API_KEY="your-api-key"
    uv run python learn/agent/context7_mcp_example.py
"""

import asyncio
import os

from dotenv import load_dotenv

from apps.engineer.learn.agent.core.model import Model
from apps.engineer.learn.agent.mcp_agent import McpAgent


async def context7_example():
    """Example: Connect to Context7 MCP service using McpAgent with headers."""
    print("=" * 60)
    print("🚀 Context7 MCP Remote Service Example")
    print("=" * 60)

    # Load environment variables
    load_dotenv()
    api_key = os.getenv("CONTEXT7_API_KEY")

    if not api_key:
        print("\n⚠️  CONTEXT7_API_KEY not set!")
        print("Please set the environment variable:")
        print("  export CONTEXT7_API_KEY='your-api-key'")
        return

    try:
        # Create agent with custom headers for Context7
        print("\n1️⃣  Creating McpAgent with Context7 configuration...")
        agent = McpAgent(
            name="Context7Agent",
            model=Model(),
            server_url="https://mcp.context7.com/mcp",
            headers={
                "CONTEXT7_API_KEY": api_key,
            }
        )
        print("   ✓ Agent created with custom headers")

        # Connect and use
        print("\n2️⃣  Connecting to Context7 MCP service...")
        await agent.connect()
        print("   ✓ Connected to https://mcp.context7.com/mcp")

        # List available tools
        print("\n3️⃣  Available tools:")
        tools = await agent._list_tools()
        for i, tool in enumerate(tools[:5], 1):
            desc = tool.get("description", "No description")
            print(f"   {i}. {tool['name']}: {desc}...")
        if len(tools) > 5:
            print(f"   ... and {len(tools) - 5} more tools")

        # Test streaming
        print("\n4️⃣  Testing streaming query:")
        print("   Query: Langgraph包括哪些核心组件?'")
        print("   Response: ", end="", flush=True)

        # async for chunk in agent.astream("Langgraph包括哪些核心组件"):
        #     if chunk.reasoning:
        #         print(f"{chunk.reasoning}", end="", flush=True)
        #     if chunk.content:
        #         print(chunk.content, end="", flush=True)
        #     if chunk.tool_calls:
        #         for tc in chunk.tool_calls:
        #             print(f"\n🔧 Tool: {tc['name']}")

        result = await agent.ainvoke("Langgraph包括哪些核心组件")
        print(result)

        print("\n" + "=" * 60)
        print("✅ Example completed!")
        print("=" * 60)

    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("Make sure FastMCP is installed: uv add fastmcp")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


async def generic_remote_mcp_example():
    """Generic example for any remote MCP service with headers."""
    print("=" * 60)
    print("🌐 Generic Remote MCP Example")
    print("=" * 60)

    # This pattern works for any MCP service requiring custom headers
    agent = McpAgent(
        name="RemoteAgent",
        model=Model(),
        server_url="https://your-mcp-service.com/mcp",
        headers={
            "Authorization": "Bearer your-token",
            "X-Custom-Header": "value",
        },
    )

    print("\n✓ Agent configured for remote MCP service")
    print("  URL: https://your-mcp-service.com/mcp")
    print("  Headers: Authorization, X-Custom-Header")


if __name__ == "__main__":
    # Run Context7 example
    asyncio.run(context7_example())
