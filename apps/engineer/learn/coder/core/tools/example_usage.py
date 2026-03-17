"""
Agent工具体系使用示例

展示如何使用新实现的工具体系
"""

import asyncio

# 导入工具体系
from apps.engineer.learn.agent.core.tools import (
    # 基础组件
    BaseTool,
    Tool,
    ToolResult,
    ToolManager,
    ToolExecutor,
    tool,
    # 内置工具
    ShellTool,
    ReadFileTool,
    WriteFileTool,
    EditFileTool,
    GlobTool,
    GrepTool,
    WebSearchTool,
    WebFetchTool,
)

# 便捷导入所有内置工具
from apps.engineer.learn.agent.core.tools.builtin import (
    create_all_builtin_tools,
    register_all_tools_with_manager,
)


def example_basic_tool_usage():
    """示例1: 基本工具使用"""
    print("=" * 50)
    print("示例1: 基本工具使用")
    print("=" * 50)

    # 创建Shell工具实例
    shell = ShellTool()

    # 同步执行命令
    result = shell.run(command="echo 'Hello, Agent Tools!'")
    print(f"命令结果: {result.output}")
    print(f"执行成功: {result.success}")
    print(f"耗时: {result.elapsed_ms:.2f}ms")
    print()


def example_file_operations():
    """示例2: 文件操作"""
    print("=" * 50)
    print("示例2: 文件操作")
    print("=" * 50)

    # 写入文件
    write_tool = WriteFileTool()
    result = write_tool.run(
        file_path="/tmp/test_agent_tools.txt",
        content="Hello from Agent Tools!\nLine 2\nLine 3",
    )
    print(f"写入结果: {result.output}")

    # 读取文件
    read_tool = ReadFileTool()
    result = read_tool.run(
        file_path="/tmp/test_agent_tools.txt",
        limit=2,
    )
    print(f"读取结果:\n{result.output}")

    # 编辑文件
    edit_tool = EditFileTool()
    result = edit_tool.run(
        file_path="/tmp/test_agent_tools.txt",
        old_string="Line 2",
        new_string="Modified Line 2",
    )
    print(f"编辑结果: {result.output}")

    # 读取修改后的文件
    result = read_tool.run(file_path="/tmp/test_agent_tools.txt")
    print(f"修改后内容:\n{result.output}")
    print()


def example_glob_and_grep():
    """示例3: 文件搜索和内容搜索"""
    print("=" * 50)
    print("示例3: 文件搜索和内容搜索")
    print("=" * 50)

    # 搜索Python文件
    glob_tool = GlobTool()
    result = glob_tool.run(
        pattern="*.py",
        path="/apps/engineer/learn/coder/core/tools",
    )
    print(f"Glob搜索结果:\n{result.output}")

    # 在文件中搜索内容
    grep_tool = GrepTool()
    result = grep_tool.run(
        pattern="class.*Tool",
        path="/apps/engineer/learn/coder/core/tools",
        output_mode="content",
    )
    print(f"Grep搜索结果:\n{result.output}")
    print()


def example_tool_manager():
    """示例4: 工具管理器"""
    print("=" * 50)
    print("示例4: 工具管理器")
    print("=" * 50)

    # 创建工具管理器
    manager = ToolManager()

    # 注册工具
    manager.register_tool(ShellTool(), category="shell")
    manager.register_tool(ReadFileTool(), category="file_system")
    manager.register_tool(WriteFileTool(), category="file_system")

    # 列出所有工具
    print(f"已注册工具: {manager.list_tools()}")
    print(f"分类列表: {manager.list_categories()}")
    print(f"file_system分类工具: {[t.name for t in manager.get_tools_by_category('file_system')]}")

    # 通过管理器执行工具
    result = manager.run_tool("shell", command="pwd")
    print(f"通过管理器执行shell: {result.output[:100]}")
    print()


def example_tool_executor():
    """示例5: 工具执行器 - 工具链"""
    print("=" * 50)
    print("示例5: 工具执行器 - 工具链")
    print("=" * 50)

    # 创建管理器和执行器
    manager = ToolManager()
    register_all_tools_with_manager(manager)

    executor = ToolExecutor(manager)

    # 定义工具链
    tool_calls = [
        {
            "name": "shell",
            "args": {"command": "echo 'Step 1'"},
        },
        {
            "name": "shell",
            "args": {"command": "echo 'Step 2'"},
        },
        {
            "name": "shell",
            "args": {"command": "echo 'Step 3'"},
        },
    ]

    # 执行工具链
    results = executor.execute_tool_chain(tool_calls)

    for i, result in enumerate(results):
        print(f"步骤 {i + 1}: 成功={result.success}")
    print()


def example_decorator_tools():
    """示例6: 装饰器创建工具"""
    print("=" * 50)
    print("示例6: 装饰器创建工具")
    print("=" * 50)

    # 使用装饰器创建工具
    @tool(name="greet", description="打招呼工具")
    def greet(name: str) -> str:
        return f"Hello, {name}!"

    @tool(name="calculate", description="简单计算器")
    def calculate(expression: str) -> str:
        try:
            result = eval(expression)
            return f"结果: {result}"
        except Exception as e:
            return f"计算错误: {e}"

    # 执行工具
    result = greet.run(name="Agent")
    print(f"问候结果: {result.output}")

    result = calculate.run(expression="2 + 3 * 4")
    print(f"计算结果: {result.output}")
    print()


def example_openai_format():
    """示例7: OpenAI格式转换"""
    print("=" * 50)
    print("示例7: OpenAI格式转换")
    print("=" * 50)

    shell = ShellTool()

    # 转换为OpenAI工具格式
    openai_format = shell.to_openai_tool()
    print("OpenAI格式:")
    import json

    print(json.dumps(openai_format, indent=2, ensure_ascii=False))
    print()


async def example_async_tools():
    """示例8: 异步工具使用"""
    print("=" * 50)
    print("示例8: 异步工具使用")
    print("=" * 50)

    shell = ShellTool()

    # 异步执行
    result = await shell.arun(command="echo 'Async Hello!'")
    print(f"异步执行结果: {result.output}")
    print()


def example_builtin_factory():
    """示例9: 使用内置工具工厂"""
    print("=" * 50)
    print("示例9: 使用内置工具工厂")
    print("=" * 50)

    # 创建所有内置工具
    tools = create_all_builtin_tools()
    print(f"创建了 {len(tools)} 个内置工具:")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description[:50]}...")
    print()


def example_callback_system():
    """示例10: 回调系统"""
    print("=" * 50)
    print("示例10: 回调系统")
    print("=" * 50)

    shell = ShellTool()

    # 注册回调
    def on_start(data):
        print(f"  [回调] 工具开始执行: {data.get('command', 'unknown')}")

    def on_end(result):
        print(f"  [回调] 工具执行完成，耗时: {result.elapsed_ms:.2f}ms")

    shell.register_callback(ToolCallbackType.ON_TOOL_START, on_start)
    shell.register_callback(ToolCallbackType.ON_TOOL_END, on_end)

    # 执行工具
    result = shell.run(command="echo 'With callbacks!'")
    print(f"结果: {result.success}")
    print()


# 修正导入
from apps.engineer.learn.agent.core.tools.base import ToolCallbackType


def main():
    """运行所有示例"""
    print("\n" + "=" * 70)
    print("Agent工具体系使用示例")
    print("=" * 70 + "\n")

    try:
        example_basic_tool_usage()
    except Exception as e:
        print(f"示例1错误: {e}")

    try:
        example_file_operations()
    except Exception as e:
        print(f"示例2错误: {e}")

    try:
        example_glob_and_grep()
    except Exception as e:
        print(f"示例3错误: {e}")

    try:
        example_tool_manager()
    except Exception as e:
        print(f"示例4错误: {e}")

    try:
        example_tool_executor()
    except Exception as e:
        print(f"示例5错误: {e}")

    try:
        example_decorator_tools()
    except Exception as e:
        print(f"示例6错误: {e}")

    try:
        example_openai_format()
    except Exception as e:
        print(f"示例7错误: {e}")

    # 异步示例
    try:
        asyncio.run(example_async_tools())
    except Exception as e:
        print(f"示例8错误: {e}")

    try:
        example_builtin_factory()
    except Exception as e:
        print(f"示例9错误: {e}")

    try:
        example_callback_system()
    except Exception as e:
        print(f"示例10错误: {e}")

    print("\n" + "=" * 70)
    print("所有示例执行完成!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
