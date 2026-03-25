# Agent工具体系设计文档

## 1. 概述

### 1.1 设计目标

Agent工具体系旨在为AI Agent提供一套完整的工具支持，使Agent能够安全、高效地与外部环境交互（执行命令、操作文件、访问网络等）。

**核心目标：**
- **标准化接口**：统一所有工具的接口定义，便于管理和扩展
- **类型安全**：使用Pydantic进行严格的参数验证
- **异步优先**：支持高并发工具执行
- **错误隔离**：工具错误不影响Agent主流程
- **可观测性**：完善的日志、监控和调试支持

### 1.2 设计原则

1. **单一职责**：每个工具只做一件事，做好一件事
2. **开闭原则**：对扩展开放，对修改封闭
3. **最小权限**：工具执行遵循最小权限原则
4. **安全优先**：默认安全，危险操作需显式授权
5. **向后兼容**：支持渐进式升级，不破坏现有代码

## 2. 架构设计

### 2.1 分层架构

```
┌─────────────────────────────────────────────────────────────┐
│                    集成层 (Integration)                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ LLM Adapters│  │Agent Tools  │  │    API Gateway      │  │
│  │OpenAI/Claude│  │   Mixin     │  │   (REST/gRPC)       │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                    管理层 (Management)                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ToolManager  │  │ToolExecutor │  │   ToolRegistry      │  │
│  │(注册/查找)  │  │(执行引擎)   │  │   (工具发现)        │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                    实现层 (Implementation)                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────┐  │
│  │   Shell  │ │ File     │ │   Web    │ │  Custom Tools  │  │
│  │   Tools  │ │ System   │ │  Tools   │ │                │  │
│  └──────────┘ └──────────┘ └──────────┘ └────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                    抽象层 (Abstraction)                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  BaseTool   │  │ ToolResult  │  │   ToolCallbackType  │  │
│  │  (抽象基类)  │  │ (结果封装)  │  │   (回调类型)        │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 核心组件

| 组件 | 职责 | 关键类 |
|------|------|--------|
| **抽象层** | 定义工具标准接口 | `BaseTool`, `ToolResult` |
| **实现层** | 提供具体工具实现 | `ShellTool`, `ReadFileTool`... |
| **管理层** | 工具注册、查找、执行 | `ToolManager`, `ToolExecutor` |
| **集成层** | LLM格式转换、API暴露 | `to_openai_tool()`... |

## 3. 核心概念

### 3.1 工具（Tool）

工具是Agent与外部环境交互的基本单元。每个工具封装一个具体功能（如执行shell命令、读取文件）。

**工具生命周期：**

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│   创建    │ -> │  验证    │ -> │  执行    │ -> │  回调    │
│ (Create) │    │(Validate)│    │ (Execute)│    │(Callback)│
└──────────┘    └──────────┘    └──────────┘    └──────────┘
       │              │              │              │
       ▼              ▼              ▼              ▼
  实例化工具      Pydantic      同步/异步      生命周期
                参数验证       执行逻辑        事件通知
```

### 3.2 工具结果（ToolResult）

标准化的工具执行结果，统一了成功和失败的返回格式。

```python
@dataclass
class ToolResult:
    output: Union[str, Dict]     # 执行输出
    success: bool                # 是否成功
    error: Optional[str]         # 错误信息
    metadata: Dict               # 元数据
    elapsed_ms: float           # 执行耗时
```

### 3.3 工具管理器（ToolManager）

工具的统一注册中心和查找服务。

**功能：**
- 工具注册与注销
- 按名称查找工具
- 按分类组织工具
- 格式转换（OpenAI/Anthropic）

### 3.4 工具执行器（ToolExecutor）

提供高级执行功能：
- **顺序执行**：工具链依次执行
- **并行执行**：多个工具并发执行
- **条件执行**：根据结果执行不同工具

## 4. 接口定义

### 4.1 BaseTool 抽象接口

```python
class BaseTool(ABC):
    """所有工具的抽象基类"""
    
    # 属性
    name: str                    # 工具名称（唯一标识）
    description: str             # 工具描述
    args_schema: Type[BaseModel] # 参数Schema
    return_direct: bool          # 是否直接返回
    
    # 核心方法
    def run(self, **kwargs) -> ToolResult
    async def arun(self, **kwargs) -> ToolResult
    
    # 格式转换
    def to_openai_tool() -> Dict[str, Any]
    def to_anthropic_tool() -> Dict[str, Any]
    
    # 回调注册
    def register_callback(type: ToolCallbackType, callback: Callable)
```

### 4.2 ToolManager 接口

```python
class ToolManager:
    """工具管理器"""
    
    # 注册管理
    def register_tool(tool: BaseTool, category: Optional[str])
    def register_tools(tools: List[BaseTool], category: Optional[str])
    def remove_tool(name: str) -> bool
    
    # 查找工具
    def get_tool(name: str) -> Optional[BaseTool]
    def has_tool(name: str) -> bool
    def list_tools() -> List[str]
    def get_tools_by_category(category: str) -> List[BaseTool]
    
    # 执行工具
    def run_tool(name: str, **kwargs) -> ToolResult
    async def arun_tool(name: str, **kwargs) -> ToolResult
    
    # 格式转换
    def get_openai_tools() -> List[Dict[str, Any]]
    def get_anthropic_tools() -> List[Dict[str, Any]]
```

### 4.3 ToolExecutor 接口

```python
class ToolExecutor:
    """工具执行器"""
    
    # 顺序执行
    def execute_tool_chain(
        tool_calls: List[Dict],
        stop_on_error: bool = True
    ) -> List[ToolResult]
    
    # 并行执行
    def execute_parallel(
        tool_calls: List[Dict],
        max_workers: int = 5
    ) -> List[ToolResult]
    
    # 条件执行
    def execute_conditional(
        tool_call: Dict,
        condition: Callable[[ToolResult], bool],
        success_tool: Optional[Dict],
        failure_tool: Optional[Dict]
    ) -> Dict[str, Any]
```

## 5. 内置工具

### 5.1 Shell工具

| 属性 | 说明 |
|------|------|
| **名称** | `shell` / `bash` |
| **功能** | 执行shell/bash命令 |
| **安全特性** | 危险命令黑名单、超时控制 |
| **参数** | `command`, `timeout`, `working_dir`, `env_vars` |

**示例：**
```python
shell = ShellTool()
result = shell.run(
    command="ls -la",
    timeout=30,
    working_dir="/tmp"
)
```

### 5.2 文件系统工具

| 工具 | 名称 | 功能 |
|------|------|------|
| 读取 | `read_file` | 读取文件内容，支持limit/offset |
| 写入 | `write_file` | 写入文件，支持追加模式 |
| 编辑 | `edit_file` | 字符串替换编辑 |
| 搜索 | `glob` | glob模式文件搜索 |
| 内容搜索 | `grep` | 正则表达式内容搜索 |

**示例：**
```python
# 读取文件
read_tool = ReadFileTool()
result = read_tool.run(
    file_path="/path/to/file.txt",
    limit=50,
    offset=1
)

# 搜索文件
glob_tool = GlobTool()
result = glob_tool.run(
    pattern="**/*.py",
    path="./src"
)
```

### 5.3 Web工具

| 工具 | 名称 | 功能 |
|------|------|------|
| 搜索 | `web_search` | 网络搜索（DuckDuckGo） |
| 获取 | `web_fetch` | 获取网页内容 |

**示例：**
```python
# 网络搜索
search_tool = WebSearchTool()
result = search_tool.run(
    query="Python async",
    num_results=5
)

# 获取网页
fetch_tool = WebFetchTool()
result = fetch_tool.run(
    url="https://example.com",
    timeout=30
)
```

## 6. 使用指南

### 6.1 直接使用工具

```python
from agent.core.tools import ShellTool, ReadFileTool

# 创建工具实例
shell = ShellTool()

# 同步执行
result = shell.run(command="echo Hello")
if result.success:
    print(result.output)

# 异步执行
result = await shell.arun(command="echo Hello")
```

### 6.2 使用工具管理器

```python
from agent.core.tools import ToolManager, ShellTool, ReadFileTool

# 创建管理器
manager = ToolManager()

# 注册工具
manager.register_tool(ShellTool(), category="shell")
manager.register_tool(ReadFileTool(), category="file_system")

# 列出工具
print(manager.list_tools())  # ['shell', 'read_file']
print(manager.list_categories())  # ['shell', 'file_system']

# 执行工具
result = manager.run_tool("shell", command="pwd")
```

### 6.3 使用工具执行器

```python
from agent.core.tools import ToolManager, ToolExecutor

manager = ToolManager()
# ... 注册工具 ...

executor = ToolExecutor(manager)

# 顺序执行工具链
tool_calls = [
    {"name": "shell", "args": {"command": "echo Step 1"}},
    {"name": "shell", "args": {"command": "echo Step 2"}},
]
results = executor.execute_tool_chain(tool_calls)

# 并行执行
tool_calls = [
    {"name": "web_search", "args": {"query": "Python"}},
    {"name": "web_search", "args": {"query": "JavaScript"}},
]
results = executor.execute_parallel(tool_calls, max_workers=2)
```

### 6.4 使用装饰器创建工具

```python
from agent.core.tools import tool, structured_tool
from pydantic import BaseModel

# 简单工具
@tool(name="greet", description="打招呼")
def greet(name: str) -> str:
    return f"Hello, {name}!"

# 结构化工具
class SearchInput(BaseModel):
    query: str
    max_results: int = 10

@structured_tool(args_schema=SearchInput)
def search(input: SearchInput) -> str:
    return f"Search: {input.query}"

# 使用工具
result = greet.run(name="World")
```

## 7. 扩展指南

### 7.1 创建自定义工具

```python
from agent.core.tools import BaseTool
from pydantic import BaseModel, Field

class MyInput(BaseModel):
    """输入参数定义"""
    param: str = Field(description="参数说明")

class MyTool(BaseTool):
    """自定义工具"""
    
    def __init__(self):
        super().__init__(
            name="my_tool",
            description="工具描述",
            args_schema=MyInput,
        )
    
    def _run(self, param: str) -> str:
        """同步执行逻辑"""
        return f"Result: {param}"
    
    async def _arun(self, param: str) -> str:
        """异步执行逻辑"""
        # 默认实现会在线程池中运行 _run
        # 可以重写以提供原生异步支持
        return await super()._arun(param=param)
```

### 7.2 工具最佳实践

1. **参数验证**
   - 使用Pydantic定义输入参数
   - 提供清晰的字段描述
   - 设置合理的默认值和约束

2. **错误处理**
   - 捕获所有异常并返回ToolResult
   - 提供有意义的错误信息
   - 不要暴露敏感信息

3. **安全性**
   - 验证所有输入参数
   - 实现权限控制
   - 设置超时和限制

4. **性能**
   - 支持异步执行
   - 避免阻塞操作
   - 考虑缓存策略

## 8. 安全设计

### 8.1 输入验证

所有工具参数通过Pydantic进行严格验证：

```python
class ShellInput(BaseModel):
    command: str = Field(description="命令")
    timeout: int = Field(default=60, ge=1, le=300)  # 范围限制
```

### 8.2 危险操作保护

Shell工具实现危险命令黑名单：

```python
DANGEROUS_COMMANDS = [
    "rm -rf /",
    "> /dev/sda",
    ":(){ :|:& };:",  # Fork bomb
]
```

### 8.3 执行隔离

- 工具错误不影响Agent主流程
- 每个工具在独立的上下文中执行
- 支持超时强制终止

## 9. 性能优化

### 9.1 并发执行

```python
# 并行执行多个工具
results = executor.execute_parallel(tool_calls, max_workers=5)

# 异步执行
results = await executor.aexecute_parallel(tool_calls)
```

### 9.2 缓存策略

工具可内置缓存机制：

```python
from functools import lru_cache

class CachedTool(BaseTool):
    @lru_cache(maxsize=100)
    def _run(self, query: str) -> str:
        # 缓存结果
        return expensive_operation(query)
```

## 10. 调试与监控

### 10.1 回调系统

```python
# 注册回调监听工具执行
tool.register_callback(ToolCallbackType.ON_TOOL_START, on_start)
tool.register_callback(ToolCallbackType.ON_TOOL_END, on_end)
tool.register_callback(ToolCallbackType.ON_TOOL_ERROR, on_error)
```

### 10.2 执行日志

每个ToolResult包含执行元数据：

```python
result = tool.run(param="value")
print(f"执行耗时: {result.elapsed_ms}ms")
print(f"元数据: {result.metadata}")
```

## 11. 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| 1.0.0 | 2026-03-16 | 初始版本，实现完整工具体系 |

## 12. 参考资料

- [LangChain Tools](https://python.langchain.com/docs/modules/agents/tools/)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [Anthropic Tool Use](https://docs.anthropic.com/claude/docs/tool-use)
- [Pydantic Documentation](https://docs.pydantic.dev/)

---

**文档版本**: 1.0.0  
**最后更新**: 2026-03-16  
**维护者**: AI Agent Team
