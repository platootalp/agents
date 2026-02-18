# 工具系统快速上手指南

## 🚀 5分钟快速开始

### 1. 创建你的第一个工具

```python
from src.core.tools import tool

@tool(description="计算两个数的和")
def add(x: int, y: int) -> int:
    return x + y

# 使用工具
result = add.run(x=5, y=3)
print(result.output)  # 输出: 8
```

### 2. 使用内置工具

```python
from src.core.tools import calculator, get_current_time

# 计算器
result = calculator.run(expression="2 + 2 * 10")
print(result.output)  # 输出: 计算结果: 22

# 获取时间
result = get_current_time.run()
print(result.output)  # 输出: 2026年01月13日 15:30:45
```

### 3. 创建结构化工具（带参数验证）

```python
from src.core.tools import structured_tool
from pydantic import BaseModel, Field

class UserInput(BaseModel):
    name: str = Field(description="用户名")
    age: int = Field(description="年龄", ge=0, le=150)

@structured_tool
def greet_user(input: UserInput) -> str:
    return f"你好, {input.name}! 你今年{input.age}岁。"

# 使用工具
result = greet_user.run(name="小明", age=25)
print(result.output)  # 输出: 你好, 小明! 你今年25岁。
```

### 4. 管理多个工具

```python
from src.core.tools import ToolManager

# 创建管理器
manager = ToolManager([add, greet_user, calculator])

# 列出所有工具
print(manager.list_tools())  # ['add', 'greet_user', 'calculator']

# 运行工具
result = manager.run_tool("add", x=10, y=20)
print(result.output)  # 30
```

### 5. 执行工具链

```python
from src.core.tools import ToolExecutor

executor = ToolExecutor(manager)

# 定义工具链
chain = [
    {"name": "add", "args": {"x": 5, "y": 3}},
    {"name": "calculator", "args": {"expression": "8 * 2"}},
]

# 执行工具链
results = executor.execute_tool_chain(chain)

for i, result in enumerate(results):
    print(f"步骤{i+1}: {result.output}")
```

## 📦 内置工具列表

| 工具名 | 功能 | 示例 |
|--------|------|------|
| `calculator` | 计算数学表达式 | `calculator.run(expression="2+2")` |
| `read_file` | 读取文件 | `read_file.run(file_path="test.txt")` |
| `write_file` | 写入文件 | `write_file.run(file_path="out.txt", content="hello")` |
| `list_directory` | 列出目录 | `list_directory.run(directory=".")` |
| `get_current_time` | 获取当前时间 | `get_current_time.run()` |
| `get_current_date` | 获取当前日期 | `get_current_date.run()` |
| `date_calculator` | 日期计算 | `date_calculator.run(days=7)` |
| `text_length` | 计算文本长度 | `text_length.run(text="hello")` |
| `text_replace` | 文本替换 | `text_replace.run(text="hi", old="hi", new="hello")` |
| `json_parse` | 解析JSON | `json_parse.run(json_string='{"a":1}')` |
| `json_extract` | 提取JSON值 | `json_extract.run(json_string='...', key_path="user.name")` |

## 🎯 常见使用场景

### 场景1: 为 LLM 提供工具

```python
from src.core.tools import ToolManager, get_all_builtin_tools

# 获取所有内置工具
tools = get_all_builtin_tools()
manager = ToolManager(tools)

# 转换为 OpenAI 格式
openai_tools = manager.to_openai_tools()

# 在 OpenAI API 中使用
# response = openai.ChatCompletion.create(
#     model="gpt-4",
#     messages=[...],
#     tools=openai_tools
# )
```

### 场景2: 自动化工作流

```python
# 定义文件处理工作流
workflow = [
    {"name": "read_file", "args": {"file_path": "input.txt"}},
    {"name": "text_replace", "args": {"old": "错误", "new": "正确"}},
    {"name": "write_file", "args": {"file_path": "output.txt"}},
]

# 执行工作流
results = executor.execute_tool_chain(workflow)
```

### 场景3: 添加自定义工具

```python
@tool(description="发送邮件")
def send_email(to: str, subject: str, body: str) -> str:
    # 实现邮件发送逻辑
    return f"邮件已发送到 {to}"

# 注册到管理器
manager.register_tool(send_email)
```

## 🔧 高级功能

### 异步工具

```python
from src.core.tools import async_tool
import asyncio

@async_tool(description="异步获取数据")
async def fetch_data(url: str) -> str:
    await asyncio.sleep(1)  # 模拟网络请求
    return f"从 {url} 获取的数据"

# 异步使用
result = await fetch_data.arun(url="https://api.example.com")
```

### 工具回调

```python
from src.core.tools import tool, ToolCallbackType

@tool(description="带回调的工具")
def my_tool(data: str) -> str:
    return f"处理: {data}"

# 注册回调
def on_start(args):
    print(f"工具开始执行: {args}")

def on_end(result):
    print(f"工具执行完成: {result.output}")

my_tool.register_callback(ToolCallbackType.ON_TOOL_START, on_start)
my_tool.register_callback(ToolCallbackType.ON_TOOL_END, on_end)

# 运行工具（会触发回调）
result = my_tool.run(data="测试")
```

### 并行执行工具

```python
# 定义多个独立的工具调用
parallel_calls = [
    {"name": "calculator", "args": {"expression": "2+2"}},
    {"name": "get_current_time", "args": {}},
    {"name": "text_length", "args": {"text": "hello"}},
]

# 并行执行
results = executor.execute_parallel(parallel_calls)
```

## 📚 更多资源

- **完整文档**: 查看 `README.md`
- **实现报告**: 查看 `IMPLEMENTATION.md`
- **示例代码**: 查看 `examples/tools_example.py`
- **简化演示**: 查看 `examples/tools_simple_demo.py`
- **单元测试**: 查看 `tests/test_tools.py`

## ⚙️ 依赖安装

```bash
# 安装 pydantic（必需）
pip install pydantic

# 或安装所有项目依赖
pip install -e .
```

## 💡 提示

1. **工具命名**: 使用清晰、描述性的名称，方便 LLM 理解
2. **参数验证**: 对于重要参数，使用 `StructuredTool` 进行严格验证
3. **错误处理**: 在工具内部处理异常，返回友好的错误信息
4. **文档描述**: 提供详细的工具描述，帮助 LLM 正确使用工具
5. **工具粒度**: 保持工具功能单一，便于组合和复用

## 🐛 故障排除

### 问题1: ModuleNotFoundError: No module named 'pydantic'

**解决方案**:
```bash
pip install pydantic
```

### 问题2: 参数验证失败

**解决方案**: 检查参数类型和约束条件是否正确

```python
class Input(BaseModel):
    age: int = Field(ge=0, le=150)  # 年龄必须在 0-150 之间
```

### 问题3: 工具执行失败

**解决方案**: 检查工具返回的 `ToolResult` 对象

```python
result = my_tool.run(arg="value")

if not result.success:
    print(f"工具执行失败: {result.error}")
else:
    print(f"工具执行成功: {result.output}")
```

## 🎉 开始构建你的工具！

现在你已经掌握了基础知识，可以开始创建自己的工具了！

```python
from src.core.tools import tool

@tool(description="你的第一个工具")
def my_first_tool(message: str) -> str:
    return f"你说: {message}"

# 运行它！
result = my_first_tool.run(message="Hello, Tools!")
print(result.output)
```

祝你使用愉快！ 🚀

