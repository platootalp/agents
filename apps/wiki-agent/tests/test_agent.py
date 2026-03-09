#!/usr/bin/env python3
"""
测试 wiki_agent.py 的基本功能和流式输出
"""

import asyncio
import sys
import os

from dotenv import load_dotenv

load_dotenv()

# 添加 src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wiki_agent import WikiAgent

from loguru import logger

# 配置日志
logger.remove()
logger.add(sys.stderr, level="INFO")


async def test_basic():
    """测试基本功能"""
    print("=== 测试 Wiki Agent 基本功能 ===\n")

    try:
        agent = WikiAgent()

        # 初始化
        print("1. 正在初始化 Agent...")
        await agent.initialize()
        print("✅ Agent 初始化完成\n")

        # 测试搜索
        print("2. 测试搜索功能...")
        result = await agent.run("获取pageId为662190775的文档内容")

        if result.get("success"):
            print(f"✅ 搜索成功")
            print(f"   工具调用次数: {result.get('tool_calls_count', 0)}")
            print(f"   输出: {result.get('output', 'N/A')[:200]}...")
        else:
            print(f"❌ 搜索失败: {result.get('error')}")

        await agent.close()
        print("\n✅ 测试完成")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()


async def test_streaming():
    """测试流式输出功能"""
    print("\n=== 测试 Wiki Agent 流式输出 ===\n")

    try:
        agent = WikiAgent()

        print("1. 正在初始化 Agent...")
        await agent.initialize()
        print("✅ Agent 初始化完成\n")

        print("2. 测试流式输出...")
        print("   流式响应: ", end="", flush=True)

        chunk_count = 0
        async for chunk in agent.run_stream("获取pageId为662190775的文档内容"):
            if chunk.get("type") == "content":
                print(chunk.get("content", ""), end="", flush=True)
                chunk_count += 1
            elif chunk.get("type") == "tool_start":
                print(f"\n   [工具调用: {chunk.get('tool_name')}]")
            elif chunk.get("type") == "tool_end":
                print(f"   [工具完成]")
            elif chunk.get("type") == "complete":
                print(f"\n   [完成 - 共 {chunk.get('tool_calls_count', 0)} 次工具调用]")

        await agent.close()
        print(f"\n✅ 流式测试完成 - 收到 {chunk_count} 个内容块")

    except Exception as e:
        print(f"❌ 流式测试失败: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # 运行基本测试
    asyncio.run(test_basic())

    # 运行流式测试
    print("\n" + "=" * 50 + "\n")
    asyncio.run(test_streaming())
