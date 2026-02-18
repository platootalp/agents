"""
工具系统单元测试
"""

# genAI_main_start
import sys
import os
import unittest
import asyncio

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.tools import (
    tool, structured_tool, async_tool,
    Tool, StructuredTool, BaseTool,
    ToolManager, ToolExecutor, ToolResult,
    calculator, get_current_time,
    create_tool_from_function
)
from pydantic import BaseModel, Field


class TestBasicTools(unittest.TestCase):
    """测试基础工具功能"""
    
    def test_tool_decorator(self):
        """测试 @tool 装饰器"""
        @tool(description="测试工具")
        def test_func(x: int) -> int:
            return x * 2
        
        result = test_func.run(x=5)
        self.assertTrue(result.success)
        self.assertEqual(result.output, 10)
    
    def test_tool_with_custom_name(self):
        """测试自定义工具名称"""
        @tool(name="custom_name", description="自定义名称的工具")
        def my_func() -> str:
            return "test"
        
        self.assertEqual(my_func.name, "custom_name")
    
    def test_tool_error_handling(self):
        """测试工具错误处理"""
        @tool(description="会出错的工具")
        def error_func() -> str:
            raise ValueError("测试错误")
        
        result = error_func.run()
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)


class TestStructuredTools(unittest.TestCase):
    """测试结构化工具功能"""
    
    def test_structured_tool_decorator(self):
        """测试 @structured_tool 装饰器"""
        class TestInput(BaseModel):
            x: int = Field(description="输入值")
            y: int = Field(description="输入值")
        
        @structured_tool
        def add(input: TestInput) -> int:
            return input.x + input.y
        
        result = add.run(x=3, y=5)
        self.assertTrue(result.success)
        self.assertEqual(result.output, 8)
    
    def test_parameter_validation(self):
        """测试参数验证"""
        class StrictInput(BaseModel):
            value: int = Field(description="必须是正数", gt=0)
        
        @structured_tool
        def strict_func(input: StrictInput) -> int:
            return input.value
        
        # 有效参数
        result = strict_func.run(value=10)
        self.assertTrue(result.success)
        
        # 无效参数
        result = strict_func.run(value=-5)
        self.assertFalse(result.success)


class TestToolClass(unittest.TestCase):
    """测试工具类"""
    
    def test_tool_class(self):
        """测试 Tool 类"""
        def my_func(text: str) -> str:
            return text.upper()
        
        tool_instance = Tool(
            name="upper",
            description="转换为大写",
            func=my_func
        )
        
        result = tool_instance.run(text="hello")
        self.assertEqual(result.output, "HELLO")
    
    def test_structured_tool_class(self):
        """测试 StructuredTool 类"""
        class MyInput(BaseModel):
            count: int = Field(description="计数")
        
        def my_func(count: int) -> str:
            return "x" * count
        
        tool_instance = StructuredTool(
            name="repeat",
            description="重复字符",
            func=my_func,
            args_schema=MyInput
        )
        
        result = tool_instance.run(count=3)
        self.assertEqual(result.output, "xxx")


class TestToolManager(unittest.TestCase):
    """测试工具管理器"""
    
    def setUp(self):
        """设置测试环境"""
        @tool(description="工具1")
        def tool1(x: int) -> int:
            return x * 2
        
        @tool(description="工具2")
        def tool2(x: int) -> int:
            return x + 10
        
        self.tool1 = tool1
        self.tool2 = tool2
    
    def test_register_tool(self):
        """测试注册工具"""
        manager = ToolManager()
        manager.register_tool(self.tool1)
        
        self.assertTrue(manager.has_tool("tool1"))
        self.assertIn("tool1", manager.list_tools())
    
    def test_run_tool(self):
        """测试运行工具"""
        manager = ToolManager([self.tool1, self.tool2])
        
        result = manager.run_tool("tool1", x=5)
        self.assertEqual(result.output, 10)
        
        result = manager.run_tool("tool2", x=5)
        self.assertEqual(result.output, 15)
    
    def test_get_tool(self):
        """测试获取工具"""
        manager = ToolManager([self.tool1])
        
        tool = manager.get_tool("tool1")
        self.assertIsNotNone(tool)
        self.assertEqual(tool.name, "tool1")
    
    def test_unregister_tool(self):
        """测试注销工具"""
        manager = ToolManager([self.tool1])
        manager.unregister_tool("tool1")
        
        self.assertFalse(manager.has_tool("tool1"))


class TestToolExecutor(unittest.TestCase):
    """测试工具执行器"""
    
    def test_tool_chain(self):
        """测试工具链执行"""
        @tool(description="步骤1")
        def step1(data: str) -> str:
            return f"[1]{data}"
        
        @tool(description="步骤2")
        def step2(data: str) -> str:
            return f"[2]{data}"
        
        manager = ToolManager([step1, step2])
        executor = ToolExecutor(manager)
        
        tool_calls = [
            {"name": "step1", "args": {"data": "start"}},
            {"name": "step2", "args": {"data": "middle"}},
        ]
        
        results = executor.execute_tool_chain(tool_calls)
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].output, "[1]start")
        self.assertEqual(results[1].output, "[2]middle")


class TestBuiltinTools(unittest.TestCase):
    """测试内置工具"""
    
    def test_calculator(self):
        """测试计算器工具"""
        result = calculator.run(expression="2 + 2")
        self.assertTrue(result.success)
        self.assertIn("4", result.output)
    
    def test_calculator_with_math(self):
        """测试计算器的 math 函数"""
        result = calculator.run(expression="math.sqrt(16)")
        self.assertTrue(result.success)
        self.assertIn("4", result.output)
    
    def test_get_current_time(self):
        """测试获取当前时间"""
        result = get_current_time.run()
        self.assertTrue(result.success)
        self.assertIsInstance(result.output, str)


class TestAsyncTools(unittest.TestCase):
    """测试异步工具"""
    
    def test_async_tool_decorator(self):
        """测试 @async_tool 装饰器"""
        @async_tool(description="异步工具")
        async def async_func(x: int) -> int:
            await asyncio.sleep(0.1)
            return x * 2
        
        async def run_test():
            result = await async_func.arun(x=5)
            self.assertTrue(result.success)
            self.assertEqual(result.output, 10)
        
        asyncio.run(run_test())


class TestToolConversion(unittest.TestCase):
    """测试工具格式转换"""
    
    def test_to_openai_tool(self):
        """测试转换为 OpenAI 格式"""
        class MyInput(BaseModel):
            query: str = Field(description="查询")
        
        @structured_tool
        def search(input: MyInput) -> str:
            return f"搜索: {input.query}"
        
        openai_format = search.to_openai_tool()
        
        self.assertEqual(openai_format["type"], "function")
        self.assertIn("function", openai_format)
        self.assertEqual(openai_format["function"]["name"], "search")
    
    def test_to_anthropic_tool(self):
        """测试转换为 Anthropic 格式"""
        class MyInput(BaseModel):
            text: str = Field(description="文本")
        
        @structured_tool
        def process(input: MyInput) -> str:
            return input.text
        
        anthropic_format = process.to_anthropic_tool()
        
        self.assertIn("name", anthropic_format)
        self.assertIn("description", anthropic_format)
        self.assertEqual(anthropic_format["name"], "process")


class TestToolCallbacks(unittest.TestCase):
    """测试工具回调"""
    
    def test_callback_execution(self):
        """测试回调执行"""
        from src.core.tools.base_tool import ToolCallbackType
        
        callback_data = {"started": False, "ended": False}
        
        @tool(description="测试回调")
        def test_func(x: int) -> int:
            return x * 2
        
        def on_start(data):
            callback_data["started"] = True
        
        def on_end(result):
            callback_data["ended"] = True
        
        test_func.register_callback(ToolCallbackType.ON_TOOL_START, on_start)
        test_func.register_callback(ToolCallbackType.ON_TOOL_END, on_end)
        
        test_func.run(x=5)
        
        self.assertTrue(callback_data["started"])
        self.assertTrue(callback_data["ended"])


class TestToolResult(unittest.TestCase):
    """测试工具结果"""
    
    def test_tool_result_to_string(self):
        """测试结果转字符串"""
        result = ToolResult(output="test output")
        self.assertEqual(result.to_string(), "test output")
    
    def test_tool_result_to_dict(self):
        """测试结果转字典"""
        result = ToolResult(
            output="test",
            success=True,
            metadata={"key": "value"}
        )
        
        result_dict = result.to_dict()
        self.assertEqual(result_dict["output"], "test")
        self.assertTrue(result_dict["success"])
        self.assertEqual(result_dict["metadata"]["key"], "value")


def run_tests():
    """运行所有测试"""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == "__main__":
    run_tests()
# genAI_main_end

