"""
Wiki Agent 使用示例
"""

import asyncio
import json
from pathlib import Path
import sys

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wiki_agent import WikiAgent
from wiki_mcp_client import WikiMCPClient


async def example_direct_client():
    """示例：直接使用客户端"""
    print("=" * 50)
    print("示例1: 直接使用 WikiMCPClient")
    print("=" * 50)

    client = WikiMCPClient()

    try:
        # 搜索页面
        print("\n1. 搜索 'API 文档':")
        result = await client.search("API 文档", limit=5)
        data = json.loads(result)
        print(json.dumps(data, indent=2, ensure_ascii=False))

        # 读取页面（需要实际的 page_id）
        # print("\n2. 读取页面:")
        # result = await client.read("12345678")
        # print(result)

    finally:
        await client.close()


async def example_agent():
    """示例：使用 Agent 模式"""
    print("\n" + "=" * 50)
    print("示例2: 使用 WikiAgent")
    print("=" * 50)

    agent = WikiAgent()

    try:
        # 示例查询
        queries = [
            "搜索关于 '部署' 的文档",
            "在父页面 ID 12345678 下创建标题为 '测试文档' 的页面",
        ]

        for query in queries:
            print(f"\n查询: {query}")
            result = await agent.run(query)
            print(f"结果: {json.dumps(result, indent=2, ensure_ascii=False)}")

    finally:
        await agent.close()


async def example_create_page():
    """示例：创建页面"""
    print("\n" + "=" * 50)
    print("示例3: 创建 Wiki 页面")
    print("=" * 50)

    client = WikiMCPClient()

    try:
        # 先搜索父页面
        print("\n1. 搜索父页面 'FileUpload 接入':")
        search_result = await client.search("FileUpload 接入", limit=1)
        search_data = json.loads(search_result)

        if search_data.get("success") and search_data.get("results"):
            parent_id = search_data["results"][0].get("page_id")
            print(f"找到父页面 ID: {parent_id}")

            # 创建子页面
            print(f"\n2. 在父页面下创建新页面:")
            html_content = """
            <h1>新页面标题</h1>
            <p>这是由 Wiki Agent 自动创建的内容。</p>
            <ul>
                <li>功能1</li>
                <li>功能2</li>
            </ul>
            """
            create_result = await client.create(
                parent_id=parent_id,
                title="测试文档 - Auto Created",
                content=html_content,
            )
            print(create_result)
        else:
            print("未找到父页面")

    finally:
        await client.close()


async def example_list_structure():
    """示例：列出页面结构"""
    print("\n" + "=" * 50)
    print("示例4: 列出页面结构")
    print("=" * 50)

    client = WikiMCPClient()

    try:
        # 搜索页面
        print("\n1. 搜索页面:")
        search_result = await client.search("架构文档", limit=1)
        search_data = json.loads(search_result)

        if search_data.get("success") and search_data.get("results"):
            page_id = search_data["results"][0].get("page_id")

            # 列出子页面
            print(f"\n2. 列出页面 {page_id} 的子页面:")
            children_result = await client.list_children(page_id, recursive=True)
            children_data = json.loads(children_result)
            print(json.dumps(children_data, indent=2, ensure_ascii=False))

    finally:
        await client.close()


async def main():
    """主函数"""
    print("Wiki Agent 使用示例")
    print("注意：运行前请确保:")
    print("1. Chrome 已启动并开启远程调试 (chrome --remote-debugging-port=9222)")
    print("2. 已登录 Wiki")
    print("3. 已设置 OPENAI_API_KEY 环境变量 (Agent 模式需要)")
    print()

    # 选择要运行的示例
    examples = {
        "1": ("直接客户端", example_direct_client),
        "2": ("Agent 模式", example_agent),
        "3": ("创建页面", example_create_page),
        "4": ("列出结构", example_list_structure),
    }

    print("可用示例:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")

    choice = input("\n选择要运行的示例 (1-4, 或按 Enter 运行所有): ").strip()

    if choice in examples:
        await examples[choice][1]()
    else:
        # 运行所有示例
        for key, (name, func) in examples.items():
            try:
                await func()
            except Exception as e:
                print(f"示例 {name} 失败: {e}")


if __name__ == "__main__":
    asyncio.run(main())
