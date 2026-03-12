"""
工具集成模块
提供工具定义、管理和执行的完整功能，参考LangChain工具设计
"""

# genAI_main_start
from .base_tool import (
    BaseTool,
    ToolResult,
    ToolCallbackType
)

from .tool import (
    Tool,
    StructuredTool,
    create_tool_from_function
)

from .decorators import (
    tool,
    async_tool,
    structured_tool
)

from .tool_manager import (
    ToolManager,
    ToolExecutor
)

from .builtin_tools import (
    calculator,
    read_file,
    write_file,
    list_directory,
    get_current_time,
    get_current_date,
    date_calculator,
    text_length,
    text_replace,
    json_parse,
    json_extract,
    get_all_builtin_tools,
    get_builtin_tool_by_name,
    list_builtin_tools
)

__all__ = [
    # 基础类
    "BaseTool",
    "ToolResult",
    "ToolCallbackType",
    
    # 工具类
    "Tool",
    "StructuredTool",
    "create_tool_from_function",
    
    # 装饰器
    "tool",
    "async_tool",
    "structured_tool",
    
    # 管理器
    "ToolManager",
    "ToolExecutor",
    
    # 内置工具
    "calculator",
    "read_file",
    "write_file",
    "list_directory",
    "get_current_time",
    "get_current_date",
    "date_calculator",
    "text_length",
    "text_replace",
    "json_parse",
    "json_extract",
    "get_all_builtin_tools",
    "get_builtin_tool_by_name",
    "list_builtin_tools",
]

__version__ = "1.0.0"
# genAI_main_end

