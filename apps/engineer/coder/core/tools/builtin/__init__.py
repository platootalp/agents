"""
创建所有内置工具的便捷函数

Usage:
    from apps.engineer.coder.core.tools.builtin import create_all_builtin_tools

    tools = create_all_builtin_tools()
    manager = ToolManager()
    manager.register_tools(tools)
"""

from typing import List

from ..base import BaseTool

# Shell工具
from .shell import ShellTool, bash

# 文件系统工具
from .file_system import (
    ReadFileTool,
    WriteFileTool,
    EditFileTool,
    GlobTool,
    GrepTool,
    read,
    write,
    edit,
)

# 目录列表工具
from .list_dir import ListDirectoryTool, list_dir

# Web工具
from .web import (
    WebSearchTool,
    WebFetchTool,
    web_search,
    web_fetch,
)


def create_all_builtin_tools() -> List[BaseTool]:
    """
    创建所有内置工具的实例

    Returns:
        List[BaseTool]: 所有内置工具实例列表
    """
    return [
        ShellTool(),
        ReadFileTool(),
        WriteFileTool(),
        EditFileTool(),
        GlobTool(),
        GrepTool(),
        ListDirectoryTool(),
        WebSearchTool(),
        WebFetchTool(),
    ]


def create_builtin_tools_by_category(category: str) -> List[BaseTool]:
    """
    按分类创建内置工具

    Args:
        category: 分类名称 ("shell", "file_system", "list", "web", "all")

    Returns:
        List[BaseTool]: 工具实例列表
    """
    tools_map = {
        "shell": [ShellTool()],
        "file_system": [
            ReadFileTool(),
            WriteFileTool(),
            EditFileTool(),
            GlobTool(),
            GrepTool(),
        ],
        "list": [ListDirectoryTool()],
        "web": [
            WebSearchTool(),
            WebFetchTool(),
        ],
        "all": create_all_builtin_tools(),
    }

    return tools_map.get(category, [])


def register_all_tools_with_manager(manager):
    """
    将所有内置工具注册到管理器

    Args:
        manager: ToolManager实例
    """
    manager.register_tool(ShellTool(), category="shell")
    manager.register_tool(ReadFileTool(), category="file_system")
    manager.register_tool(WriteFileTool(), category="file_system")
    manager.register_tool(EditFileTool(), category="file_system")
    manager.register_tool(GlobTool(), category="file_system")
    manager.register_tool(GrepTool(), category="file_system")
    manager.register_tool(ListDirectoryTool(), category="list")
    manager.register_tool(WebSearchTool(), category="web")
    manager.register_tool(WebFetchTool(), category="web")


__all__ = [
    "create_all_builtin_tools",
    "create_builtin_tools_by_category",
    "register_all_tools_with_manager",
    "ListDirectoryTool",
    "list_dir",
]
