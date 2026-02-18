# 工具集成系统

一个功能强大的工具集成框架，参考 LangChain 的工具设计理念，为 LLM 应用提供灵活的工具调用能力。

## 📋 目录

- [功能特性](#功能特性)
- [快速开始](#快速开始)
- [核心组件](#核心组件)
- [使用指南](#使用指南)
- [内置工具](#内置工具)
- [最佳实践](#最佳实践)

## ✨ 功能特性

- **🎯 多种工具创建方式**：支持装饰器、函数式、类式等多种工具定义方式
- **✅ 参数验证**：基于 Pydantic 的强类型参数验证
- **🔄 同步/异步支持**：同时支持同步和异步工具执行
- **🔧 工具管理器**：统一管理和调用多个工具
- **⚡ 工具链执行**：支持顺序执行和并行执行
- **🎣 回调机制**：支持工具执行前后的回调函数
- **🤝 LLM集成**：轻松转换为 OpenAI、Anthropic 等格式
- **📦 内置工具集**：提供计算器、文件操作、时间日期等常用工具

## 🚀 快速开始

### 1. 使用装饰器创建工具

```python
from src.core.tools import tool

@tool(description="向用户问候")
def greet(name: str) -> str:
    return f"你好, {name}!"

# 调用工具
result = greet.run(name="小明")
print(result.output)  # 输出: 你好, 小明!
```

### 2. 使用结构化工具（带参数验证）

```python
from src.core.tools import structured_tool
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(description="搜索查询")
    max_results: int = Field(default=10, description="最大结果数", ge=1, le=100)

@structured_tool
def search(input: SearchInput) -> str:
    """在互联网上搜索信息"""
    return f"搜索'{input.query}'，返回{input.max_results}条结果"

result = search.run(query="Python教程", max_results=5)
```

### 3. 使用内置工具

```python
from src.core.tools import calculator, get_current_time

# 使用计算器
result = calculator.run(expression="2 + 2 * 10")
print(result.output)  # 输出: 计算结果: 22

# 获取当前时间
result = get_current_time.run()
print(result.output)
```

## 🧩 核心组件

### BaseTool - 工具基类

所有工具的抽象基类，定义了工具的标准接口。

```python
from src.core.tools import BaseTool

class MyTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="my_tool",
            description="我的自定义工具"
        )
    
    def _run(self, *args, **kwargs):
        # 实现工具逻辑
        return "执行结果"
```

### Tool - 通用工具

通过传入函数快速创建工具。

```python
from src.core.tools import Tool

def my_function(arg1: str, arg2: int) -> str:
    return f"{arg1} - {arg2}"

my_tool = Tool(
    name="my_tool",
    description="执行某项任务",
    func=my_function
)
```

### StructuredTool - 结构化工具

强制使用 Pydantic 模型进行参数验证。

```python
from src.core.tools import StructuredTool
from pydantic import BaseModel, Field

class MyInput(BaseModel):
    text: str = Field(description="输入文本")
    count: int = Field(description="计数")

def my_function(text: str, count: int) -> str:
    return text * count

my_tool = StructuredTool(
    name="repeat_text",
    description="重复文本",
    func=my_function,
    args_schema=MyInput
)
```

### ToolManager - 工具管理器

管理多个工具的注册、查找和调用。

```python
from src.core.tools import ToolManager, tool

@tool(description="工具1")
def tool1(x: int) -> int:
    return x * 2

@tool(description="工具2")
def tool2(x: int) -> int:
    return x + 10

# 创建管理器
manager = ToolManager([tool1, tool2])

# 运行工具
result = manager.run_tool("tool1", x=5)
print(result.output)  # 输出: 10
```

### ToolExecutor - 工具执行器

提供高级执行功能，如工具链、并行执行。

```python
from src.core.tools import ToolManager, ToolExecutor

manager = ToolManager([tool1, tool2, tool3])
executor = ToolExecutor(manager)

# 执行工具链
tool_calls = [
    {"name": "tool1", "args": {"x": 5}},
    {"name": "tool2", "args": {"x": 10}},
    {"name": "tool3", "args": {"x": 15}},
]

results = executor.execute_tool_chain(tool_calls)
```

## 📖 使用指南

### 装饰器方式

#### @tool - 简单工具装饰器

```python
from src.core.tools import tool

@tool(name="custom_name", description="自定义描述", verbose=True)
def my_function(param: str) -> str:
    """函数文档字符串会作为默认描述"""
    return f"处理: {param}"
```

#### @async_tool - 异步工具装饰器

```python
from src.core.tools import async_tool
import asyncio

@async_tool(description="异步获取数据")
async def fetch_data(url: str) -> str:
    await asyncio.sleep(1)
    return f"从{url}获取的数据"

# 异步调用
result = await fetch_data.arun(url="https://api.example.com")
```

#### @structured_tool - 结构化工具装饰器

```python
from src.core.tools import structured_tool
from pydantic import BaseModel, Field

class CalculatorInput(BaseModel):
    expression: str = Field(description="数学表达式")

@structured_tool
def calculate(input: CalculatorInput) -> str:
    """计算数学表达式"""
    return str(eval(input.expression))
```

### 函数式创建

```python
from src.core.tools import create_tool_from_function

def greet(name: str) -> str:
    """向用户问候"""
    return f"你好, {name}!"

greet_tool = create_tool_from_function(greet)
```

### 工具回调

```python
from src.core.tools import tool, ToolCallbackType

@tool(description="执行任务")
def my_task(data: str) -> str:
    return f"处理: {data}"

# 注册回调
def on_start(data):
    print(f"开始执行，参数: {data}")

def on_end(result):
    print(f"执行完成，结果: {result.output}")

my_task.register_callback(ToolCallbackType.ON_TOOL_START, on_start)
my_task.register_callback(ToolCallbackType.ON_TOOL_END, on_end)

result = my_task.run(data="测试数据")
```

### LLM 集成

#### 转换为 OpenAI 工具格式

```python
from src.core.tools import tool

@tool(description="搜索信息")
def search(query: str) -> str:
    return f"搜索结果: {query}"

# 转换为 OpenAI 格式
openai_tool = search.to_openai_tool()

# 用于 OpenAI API
import openai
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "帮我搜索Python教程"}],
    tools=[openai_tool]
)
```

#### 转换为 Anthropic 工具格式

```python
# 转换为 Anthropic 格式
anthropic_tool = search.to_anthropic_tool()

# 用于 Anthropic API
import anthropic
client = anthropic.Anthropic()
response = client.messages.create(
    model="claude-3-opus-20240229",
    messages=[{"role": "user", "content": "帮我搜索Python教程"}],
    tools=[anthropic_tool]
)
```

## 📦 内置工具

### 计算器工具

```python
from src.core.tools import calculator

result = calculator.run(expression="2 + 2 * 10")
result = calculator.run(expression="math.sqrt(16)")
```

### 文件操作工具

```python
from src.core.tools import read_file, write_file, list_directory

# 读取文件
result = read_file.run(file_path="test.txt")

# 写入文件
result = write_file.run(file_path="output.txt", content="Hello, World!")

# 列出目录
result = list_directory.run(directory=".", pattern="*.py")
```

### 时间日期工具

```python
from src.core.tools import get_current_time, get_current_date, date_calculator

# 获取当前时间
result = get_current_time.run()

# 获取当前日期
result = get_current_date.run()

# 日期计算
result = date_calculator.run(days=7)  # 7天后
result = date_calculator.run(days=-7, from_date="2024-01-01")  # 7天前
```

### 文本处理工具

```python
from src.core.tools import text_length, text_replace

# 计算文本长度
result = text_length.run(text="Hello, World!")

# 文本替换
result = text_replace.run(text="Hello World", old="World", new="Python")
```

### JSON 工具

```python
from src.core.tools import json_parse, json_extract

# 解析 JSON
result = json_parse.run(json_string='{"name": "张三", "age": 30}')

# 提取 JSON 值
result = json_extract.run(
    json_string='{"user": {"name": "张三"}}',
    key_path="user.name"
)
```

### 列出所有内置工具

```python
from src.core.tools import list_builtin_tools, get_all_builtin_tools

# 打印工具列表
print(list_builtin_tools())

# 获取所有工具实例
tools = get_all_builtin_tools()
```

## 💡 最佳实践

### 1. 使用 Pydantic 进行参数验证

```python
from pydantic import BaseModel, Field, validator

class EmailInput(BaseModel):
    email: str = Field(description="邮箱地址")
    
    @validator('email')
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('无效的邮箱地址')
        return v

@structured_tool
def send_email(input: EmailInput) -> str:
    return f"发送邮件到: {input.email}"
```

### 2. 错误处理

```python
@tool(description="可能失败的操作")
def risky_operation(data: str) -> str:
    try:
        # 执行可能失败的操作
        result = process_data(data)
        return f"成功: {result}"
    except Exception as e:
        return f"失败: {str(e)}"

result = risky_operation.run(data="test")
if not result.success:
    print(f"操作失败: {result.error}")
```

### 3. 使用工具管理器组织工具

```python
from src.core.tools import ToolManager

# 按功能分组
file_tools = ToolManager([read_file, write_file, list_directory])
text_tools = ToolManager([text_length, text_replace])
math_tools = ToolManager([calculator])

# 统一管理
all_tools = ToolManager()
for tool in file_tools.get_all_tools():
    all_tools.register_tool(tool)
for tool in text_tools.get_all_tools():
    all_tools.register_tool(tool)
```

### 4. 工具链设计

```python
# 设计可组合的工具
@tool(description="步骤1: 获取数据")
def fetch_data(source: str) -> str:
    return f"data_from_{source}"

@tool(description="步骤2: 处理数据")
def process_data(data: str) -> str:
    return f"processed_{data}"

@tool(description="步骤3: 保存数据")
def save_data(data: str) -> str:
    return f"saved_{data}"

# 执行工具链
manager = ToolManager([fetch_data, process_data, save_data])
executor = ToolExecutor(manager)

pipeline = [
    {"name": "fetch_data", "args": {"source": "api"}},
    {"name": "process_data", "args": {"data": "raw"}},
    {"name": "save_data", "args": {"data": "final"}},
]

results = executor.execute_tool_chain(pipeline)
```

### 5. 详细日志

```python
# 开启详细日志
@tool(description="调试工具", verbose=True)
def debug_tool(x: int) -> int:
    return x * 2

manager = ToolManager(verbose=True)
manager.register_tool(debug_tool)

result = manager.run_tool("debug_tool", x=10)
```

## 🔗 集成示例

### 与 LLM 配合使用

```python
from src.core.tools import ToolManager, get_all_builtin_tools
from src.core import create_model

# 创建工具管理器
tools = get_all_builtin_tools()
manager = ToolManager(tools)

# 创建 LLM
llm = create_model("gpt-4")

# 将工具转换为 LLM 可用格式
openai_tools = manager.to_openai_tools()

# 让 LLM 使用工具
messages = [
    {"role": "user", "content": "帮我计算 2 + 2 * 10"}
]

# LLM 会返回需要调用的工具
# 然后使用管理器执行工具
tool_call = {"name": "calculator", "args": {"expression": "2 + 2 * 10"}}
result = manager.run_tool(**tool_call)
```

## 📝 注意事项

1. **参数验证**：推荐使用 `StructuredTool` 或 `@structured_tool` 进行严格的参数验证
2. **异步支持**：对于 I/O 密集型操作，使用异步工具可以提高性能
3. **错误处理**：工具内部应妥善处理异常，避免中断整个流程
4. **工具命名**：使用清晰、描述性的工具名称，方便 LLM 理解和选择
5. **描述清晰**：提供详细的工具描述，帮助 LLM 正确使用工具

## 🔧 扩展开发

### 创建自定义工具类

```python
from src.core.tools import BaseTool, ToolResult

class CustomTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="custom_tool",
            description="自定义工具"
        )
        # 初始化资源
        self.resource = None
    
    def _run(self, param: str) -> str:
        # 实现工具逻辑
        return f"处理: {param}"
    
    async def _arun(self, param: str) -> str:
        # 异步实现
        return await async_process(param)
```

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

