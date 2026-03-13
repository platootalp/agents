"""Debug script to test McpAgent conversation loop."""

import asyncio
import sys

sys.path.insert(0, "/Users/lijunyi/road/agents")

from apps.engineer.learn.agent.mcp_agent import McpAgent
from apps.engineer.learn.agent.core.model import Model

# Create a simple test server code
SERVER_CODE = '''
from fastmcp import FastMCP

mcp = FastMCP("Test Server")

@mcp.tool
def search(query: str) -> str:
    """Search for information."""
    return f"Search results for '{query}': Found 10 items about AI and machine learning."

if __name__ == "__main__":
    mcp.run()
'''


async def test_conversation():
    import tempfile
    import os

    # Write server script to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(SERVER_CODE)
        server_path = f.name

    try:
        print("=" * 60)
        print("Testing McpAgent Conversation Loop")
        print("=" * 60)

        agent = McpAgent(
            name="TestAgent",
            model=Model(),
            command="python",
            args=[server_path],
        )
        print("\n✓ Agent created")

        # Test invoke (non-streaming)
        print("\n--- Testing invoke() ---")
        query = "Search for information about AI"
        print(f"Query: {query}")

        # Patch to debug
        original_run_step = agent._run_non_streaming_step

        async def debug_run_step(all_tools):
            print(f"  [DEBUG] Running step with {len(all_tools)} tools")
            result = await original_run_step(all_tools)
            print(f"  [DEBUG] Step result: {result[:50] if result else 'None (continue)'}...")
            return result

        agent._run_non_streaming_step = debug_run_step

        response = await agent.ainvoke(query)
        print(f"\nFinal response: {response}")

        print("\n--- Message History ---")
        for i, msg in enumerate(agent.message_history):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:100]
            tool_calls = msg.get("tool_calls", [])
            print(f"  [{i}] {role}: {content}{'...' if len(content) >= 100 else ''}")
            if tool_calls:
                print(f"      tool_calls: {len(tool_calls)} calls")

    finally:
        os.unlink(server_path)


if __name__ == "__main__":
    asyncio.run(test_conversation())
