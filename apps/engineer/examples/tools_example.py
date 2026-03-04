"""
工具系统使用示例
演示如何使用工具集成功能
"""

# genAI_main_start
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from engineer.core.tools import (
    tool, structured_tool, async_tool,
    Tool, StructuredTool,
    ToolManager, ToolExecutor,
    calculator, get_current_time,
    get_all_builtin_tools, list_builtin_tools
)
from pydantic import BaseModel, Field


def example_1_basic_tool():
    """示例1: 使用装饰器创建简单工具"""
    print("\n" + "="*60)
    print("示例1: 使用装饰器创建简单工具")
    print("="*60)
    
    @tool(description="向用户问候")
    def greet(name: str) -> str:
        return f"你好, {name}! 欢迎使用工具系统。"
    
    # 调用工具
    result = greet.run(name="小明")
    print(f"成功: {result.success}")
    print(f"输出: {result.output}")


def example_2_structured_tool():
    """示例2: 使用结构化工具（带参数验证）"""
    print("\n" + "="*60)
    print("示例2: 使用结构化工具（带参数验证）")
    print("="*60)
    
    # 定义参数模式
    class SearchInput(BaseModel):
        query: str = Field(description="搜索查询")
        max_results: int = Field(default=10, description="最大结果数", ge=1, le=100)
    
    @structured_tool
    def search(input: SearchInput) -> str:
        """在互联网上搜索信息"""
        return f"搜索'{input.query}'，返回{input.max_results}条结果"
    
    # 调用工具
    result = search.run(query="Python教程", max_results=5)
    print(f"输出: {result.output}")
    
    # 参数验证测试
    try:
        result = search.run(query="测试", max_results=200)  # 超出范围
    except ValueError as e:
        print(f"参数验证失败（预期）: {e}")


def example_3_builtin_tools():
    """示例3: 使用内置工具"""
    print("\n" + "="*60)
    print("示例3: 使用内置工具")
    print("="*60)
    
    # 使用计算器
    result = calculator.run(expression="2 + 2 * 10")
    print(f"计算结果: {result.output}")
    
    # 使用时间工具
    result = get_current_time.run()
    print(f"当前时间: {result.output}")
    
    # 列出所有内置工具
    print("\n" + list_builtin_tools())


def example_4_tool_manager():
    """示例4: 使用工具管理器"""
    print("\n" + "="*60)
    print("示例4: 使用工具管理器")
    print("="*60)
    
    # 创建自定义工具
    @tool(description="将文本转换为大写")
    def to_upper(text: str) -> str:
        return text.upper()
    
    @tool(description="将文本转换为小写")
    def to_lower(text: str) -> str:
        return text.lower()
    
    # 创建工具管理器
    manager = ToolManager(verbose=True)
    manager.register_tool(to_upper)
    manager.register_tool(to_lower)
    
    # 列出所有工具
    print(f"注册的工具: {manager.list_tools()}")
    
    # 运行工具
    result = manager.run_tool("to_upper", text="hello world")
    print(f"大写结果: {result.output}")
    
    result = manager.run_tool("to_lower", text="HELLO WORLD")
    print(f"小写结果: {result.output}")


def example_5_tool_executor():
    """示例5: 使用工具执行器（工具链）"""
    print("\n" + "="*60)
    print("示例5: 使用工具执行器（工具链）")
    print("="*60)
    
    # 创建工具
    @tool(description="步骤1")
    def step1(data: str) -> str:
        return f"[步骤1完成] {data}"
    
    @tool(description="步骤2")
    def step2(data: str) -> str:
        return f"[步骤2完成] {data}"
    
    @tool(description="步骤3")
    def step3(data: str) -> str:
        return f"[步骤3完成] {data}"
    
    # 创建管理器和执行器
    manager = ToolManager([step1, step2, step3])
    executor = ToolExecutor(manager, verbose=True)
    
    # 执行工具链
    tool_calls = [
        {"name": "step1", "args": {"data": "初始数据"}},
        {"name": "step2", "args": {"data": "处理中的数据"}},
        {"name": "step3", "args": {"data": "最终数据"}},
    ]
    
    results = executor.execute_tool_chain(tool_calls)
    
    print("\n工具链执行结果:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result.output}")


def example_6_openai_format():
    """示例6: 转换为OpenAI工具格式"""
    print("\n" + "="*60)
    print("示例6: 转换为OpenAI工具格式")
    print("="*60)
    
    # 创建带参数模式的工具
    class WeatherInput(BaseModel):
        location: str = Field(description="城市名称")
        unit: str = Field(default="celsius", description="温度单位")
    
    @structured_tool
    def get_weather(input: WeatherInput) -> str:
        """获取指定城市的天气信息"""
        return f"{input.location}的天气: 晴朗, 温度: 25{input.unit}"
    
    # 转换为OpenAI格式
    openai_tool = get_weather.to_openai_tool()
    
    import json
    print("OpenAI工具格式:")
    print(json.dumps(openai_tool, ensure_ascii=False, indent=2))
    
    # 转换为Anthropic格式
    anthropic_tool = get_weather.to_anthropic_tool()
    print("\nAnthropic工具格式:")
    print(json.dumps(anthropic_tool, ensure_ascii=False, indent=2))


def example_7_tool_callbacks():
    """示例7: 工具回调"""
    print("\n" + "="*60)
    print("示例7: 工具回调")
    print("="*60)
    
    from engineer.core.tools.base_tool import ToolCallbackType
    
    @tool(description="执行任务")
    def perform_task(task: str) -> str:
        return f"任务'{task}'已完成"
    
    # 注册回调
    def on_start(data):
        print(f"🚀 工具开始执行，参数: {data}")
    
    def on_end(result):
        print(f"✅ 工具执行完成，结果: {result.output}")
    
    def on_error(error):
        print(f"❌ 工具执行失败，错误: {error}")
    
    perform_task.register_callback(ToolCallbackType.ON_TOOL_START, on_start)
    perform_task.register_callback(ToolCallbackType.ON_TOOL_END, on_end)
    perform_task.register_callback(ToolCallbackType.ON_TOOL_ERROR, on_error)
    
    # 运行工具
    result = perform_task.run(task="数据分析")


def example_8_async_tool():
    """示例8: 异步工具"""
    print("\n" + "="*60)
    print("示例8: 异步工具")
    print("="*60)
    
    import asyncio
    
    @async_tool(description="异步获取数据")
    async def async_fetch(url: str) -> str:
        """模拟异步网络请求"""
        await asyncio.sleep(1)  # 模拟延迟
        return f"从 {url} 获取的数据"
    
    # 异步运行
    async def run_async():
        result = await async_fetch.arun(url="https://api.example.com/data")
        print(f"异步结果: {result.output}")
    
    asyncio.run(run_async())


def main():
    """运行所有示例"""
    print("\n" + "="*70)
    print("工具系统使用示例")
    print("="*70)
    
    example_1_basic_tool()
    example_2_structured_tool()
    example_3_builtin_tools()
    example_4_tool_manager()
    example_5_tool_executor()
    example_6_openai_format()
    example_7_tool_callbacks()
    example_8_async_tool()
    
    print("\n" + "="*70)
    print("所有示例运行完成！")
    print("="*70)


if __name__ == "__main__":
    main()
# genAI_main_end

