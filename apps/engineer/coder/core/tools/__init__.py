"""
Agent工具体系

提供完整的工具支持，包括：
- 工具基类和装饰器
- 工具管理器和执行器
- 内置工具：shell、file_system、web
"""

# 基础模块
from .base import (
    BaseTool,
    Tool,
    StructuredTool,
    ToolResult,
    ToolCallbackType,
    tool,
    structured_tool,
    async_tool,
)

# 管理器
from .manager import (
    ToolManager,
    ToolExecutor,
)

# Shell工具
from .builtin.shell import (
    ShellTool,
    ShellInput,
    ShellOutput,
    bash,
)

# 文件系统工具
from .builtin.file_system import (
    ReadFileTool,
    WriteFileTool,
    EditFileTool,
    GlobTool,
    GrepTool,
    ReadFileInput,
    WriteFileInput,
    EditFileInput,
    GlobInput,
    GrepInput,
    read,
    write,
    edit,
)

# Web工具
from .builtin.web import (
    WebSearchTool,
    WebFetchTool,
    WebSearchInput,
    WebFetchInput,
    WebSearchResult,
    web_search,
    web_fetch,
)

__all__ = [
    # 基础
    "BaseTool",
    "Tool",
    "StructuredTool",
    "ToolResult",
    "ToolCallbackType",
    "tool",
    "structured_tool",
    "async_tool",
    # 管理器
    "ToolManager",
    "ToolExecutor",
    # Shell
    "ShellTool",
    "ShellInput",
    "ShellOutput",
    "bash",
    # 文件系统
    "ReadFileTool",
    "WriteFileTool",
    "EditFileTool",
    "GlobTool",
    "GrepTool",
    "ReadFileInput",
    "WriteFileInput",
    "EditFileInput",
    "GlobInput",
    "GrepInput",
    "read",
    "write",
    "edit",
    # Web
    "WebSearchTool",
    "WebFetchTool",
    "WebSearchInput",
    "WebFetchInput",
    "WebSearchResult",
    "web_search",
    "web_fetch",
]
