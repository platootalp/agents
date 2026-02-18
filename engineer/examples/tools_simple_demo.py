"""
工具系统简单演示
不依赖外部模块，演示核心工具功能
"""

# genAI_main_start
import sys
import os

# 添加项目路径到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# 直接导入工具模块（避免触发 src.core.__init__.py）
from core.tools import (
    tool, structured_tool,
    Tool, BaseTool, ToolResult,
    ToolManager, ToolExecutor
)
from pydantic import BaseModel, Field


def demo_1_basic_tool():
    """演示1: 基础工具"""
    print("\n" + "="*60)
    print("演示1: 使用装饰器创建简单工具")
    print("="*60)
    
    @tool(description="向用户问候")
    def greet(name: str) -> str:
        return f"你好, {name}! 欢迎使用工具系统。"
    
    result = greet.run(name="小明")
    print(f"✓ 成功: {result.success}")
    print(f"✓ 输出: {result.output}")


def demo_2_structured_tool():
    """演示2: 结构化工具"""
    print("\n" + "="*60)
    print("演示2: 使用结构化工具（带参数验证）")
    print("="*60)
    
    class CalculatorInput(BaseModel):
        x: int = Field(description="第一个数")
        y: int = Field(description="第二个数")
        operation: str = Field(default="add", description="操作：add, subtract, multiply, divide")
    
    @structured_tool
    def my_calculator(input: CalculatorInput) -> str:
        """简单的计算器"""
        if input.operation == "add":
            result = input.x + input.y
        elif input.operation == "subtract":
            result = input.x - input.y
        elif input.operation == "multiply":
            result = input.x * input.y
        elif input.operation == "divide":
            result = input.x / input.y if input.y != 0 else "错误：除数为0"
        else:
            return f"不支持的操作: {input.operation}"
        
        return f"{input.x} {input.operation} {input.y} = {result}"
    
    result = my_calculator.run(x=10, y=5, operation="add")
    print(f"✓ 加法: {result.output}")
    
    result = my_calculator.run(x=10, y=5, operation="multiply")
    print(f"✓ 乘法: {result.output}")


def demo_3_tool_manager():
    """演示3: 工具管理器"""
    print("\n" + "="*60)
    print("演示3: 使用工具管理器")
    print("="*60)
    
    @tool(description="将文本转换为大写")
    def to_upper(text: str) -> str:
        return text.upper()
    
    @tool(description="将文本转换为小写")
    def to_lower(text: str) -> str:
        return text.lower()
    
    @tool(description="反转文本")
    def reverse_text(text: str) -> str:
        return text[::-1]
    
    # 创建管理器
    manager = ToolManager([to_upper, to_lower, reverse_text], verbose=True)
    
    print(f"\n✓ 注册的工具: {manager.list_tools()}")
    
    # 运行工具
    result = manager.run_tool("to_upper", text="hello world")
    print(f"✓ 大写: {result.output}")
    
    result = manager.run_tool("reverse_text", text="Hello")
    print(f"✓ 反转: {result.output}")


def demo_4_tool_chain():
    """演示4: 工具链执行"""
    print("\n" + "="*60)
    print("演示4: 工具链执行")
    print("="*60)
    
    @tool(description="步骤1: 添加前缀")
    def add_prefix(text: str) -> str:
        return f"[前缀] {text}"
    
    @tool(description="步骤2: 转换为大写")
    def to_upper(text: str) -> str:
        return text.upper()
    
    @tool(description="步骤3: 添加后缀")
    def add_suffix(text: str) -> str:
        return f"{text} [后缀]"
    
    manager = ToolManager([add_prefix, to_upper, add_suffix])
    executor = ToolExecutor(manager, verbose=True)
    
    # 定义工具链
    pipeline = [
        {"name": "add_prefix", "args": {"text": "hello"}},
        {"name": "to_upper", "args": {"text": "[前缀] hello"}},
        {"name": "add_suffix", "args": {"text": "[前缀] HELLO"}},
    ]
    
    print("\n执行工具链...")
    results = executor.execute_tool_chain(pipeline)
    
    print("\n工具链结果:")
    for i, result in enumerate(results, 1):
        print(f"  步骤{i}: {result.output}")


def demo_5_tool_format():
    """演示5: 工具格式转换"""
    print("\n" + "="*60)
    print("演示5: 工具格式转换（OpenAI/Anthropic）")
    print("="*60)
    
    class SearchInput(BaseModel):
        query: str = Field(description="搜索查询")
        max_results: int = Field(default=10, description="最大结果数")
    
    @structured_tool
    def search(input: SearchInput) -> str:
        """搜索互联网信息"""
        return f"搜索'{input.query}'，返回{input.max_results}条结果"
    
    import json
    
    print("\n✓ OpenAI 工具格式:")
    openai_format = search.to_openai_tool()
    print(f"  type: {openai_format['type']}")
    print(f"  name: {openai_format['function']['name']}")
    print(f"  description: {openai_format['function']['description']}")
    
    print("\n✓ Anthropic 工具格式:")
    anthropic_format = search.to_anthropic_tool()
    print(f"  name: {anthropic_format['name']}")
    print(f"  description: {anthropic_format['description']}")


def demo_6_error_handling():
    """演示6: 错误处理"""
    print("\n" + "="*60)
    print("演示6: 错误处理")
    print("="*60)
    
    @tool(description="可能出错的工具")
    def risky_operation(should_fail: bool) -> str:
        if should_fail:
            raise ValueError("故意触发的错误")
        return "操作成功"
    
    # 成功情况
    result = risky_operation.run(should_fail=False)
    print(f"✓ 成功情况 - 成功: {result.success}, 输出: {result.output}")
    
    # 失败情况
    result = risky_operation.run(should_fail=True)
    print(f"✓ 失败情况 - 成功: {result.success}, 错误: {result.error}")


def main():
    """运行所有演示"""
    print("\n" + "="*70)
    print("工具系统核心功能演示")
    print("="*70)
    
    demo_1_basic_tool()
    demo_2_structured_tool()
    demo_3_tool_manager()
    demo_4_tool_chain()
    demo_5_tool_format()
    demo_6_error_handling()
    
    print("\n" + "="*70)
    print("✅ 所有演示运行完成！")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
# genAI_main_end

