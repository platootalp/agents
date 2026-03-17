"""
List 工具 - 目录列表和文件浏览

提供安全的目录浏览功能
"""

import json
import os
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

from ..base import BaseTool, Tool, tool


class ListDirectoryInput(BaseModel):
    """列出目录输入参数"""

    directory: str = Field(default=".", description="要列出的目录路径")
    pattern: Optional[str] = Field(default=None, description="文件匹配模式，如 '*.py'")
    show_hidden: bool = Field(default=False, description="是否显示隐藏文件")
    recursive: bool = Field(default=False, description="是否递归列出子目录")
    max_depth: int = Field(default=2, description="递归最大深度", ge=1, le=10)


class ListDirectoryOutput(BaseModel):
    """列出目录输出结果"""

    directory: str = Field(description="目录路径")
    total_items: int = Field(description="总项目数")
    subdirectories: List[dict] = Field(description="子目录列表")
    files: List[dict] = Field(description="文件列表")
    errors: List[str] = Field(default_factory=list, description="错误信息")


class ListDirectoryTool(BaseTool):
    """
    列出目录工具

    安全地列出目录内容，支持过滤和递归
    """

    def __init__(self):
        super().__init__(
            name="list_directory",
            description="列出目录中的文件和子目录，支持过滤和递归",
            args_schema=ListDirectoryInput,
        )

    def _run(
        self,
        directory: str = ".",
        pattern: Optional[str] = None,
        show_hidden: bool = False,
        recursive: bool = False,
        max_depth: int = 2,
    ) -> str:
        """
        列出目录内容

        Args:
            directory: 目录路径
            pattern: 文件匹配模式
            show_hidden: 是否显示隐藏文件
            recursive: 是否递归
            max_depth: 最大递归深度

        Returns:
            str: JSON格式的目录列表
        """
        try:
            base_path = Path(directory).expanduser().resolve()

            if not base_path.exists():
                return json.dumps(
                    {"error": f"目录不存在: {directory}", "success": False},
                    ensure_ascii=False,
                )

            if not base_path.is_dir():
                return json.dumps(
                    {"error": f"路径不是目录: {directory}", "success": False},
                    ensure_ascii=False,
                )

            result = self._list_directory_recursive(
                base_path, pattern, show_hidden, recursive, max_depth, current_depth=1
            )

            return json.dumps(result, ensure_ascii=False, indent=2)

        except Exception as e:
            return json.dumps({"error": str(e), "success": False}, ensure_ascii=False)

    def _list_directory_recursive(
        self,
        path: Path,
        pattern: Optional[str],
        show_hidden: bool,
        recursive: bool,
        max_depth: int,
        current_depth: int,
    ) -> dict:
        """递归列出目录内容"""
        import fnmatch

        subdirectories = []
        files = []
        errors = []

        try:
            items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))

            for item in items:
                # 跳过隐藏文件
                if not show_hidden and item.name.startswith("."):
                    continue

                # 应用模式过滤
                if pattern and not fnmatch.fnmatch(item.name, pattern):
                    continue

                try:
                    stat = item.stat()
                    info = {
                        "name": item.name,
                        "path": str(item.relative_to(path)),
                        "size": stat.st_size if item.is_file() else None,
                        "modified": stat.st_mtime,
                    }

                    if item.is_dir():
                        info["type"] = "directory"
                        # 递归处理子目录
                        if recursive and current_depth < max_depth:
                            sub_result = self._list_directory_recursive(
                                item, pattern, show_hidden, recursive, max_depth, current_depth + 1
                            )
                            info["children"] = {
                                "subdirectories": sub_result.get("subdirectories", []),
                                "files": sub_result.get("files", []),
                            }
                        subdirectories.append(info)
                    else:
                        info["type"] = "file"
                        files.append(info)

                except (OSError, PermissionError) as e:
                    errors.append(f"无法访问 {item.name}: {str(e)}")

        except PermissionError as e:
            errors.append(f"权限错误: {str(e)}")

        return {
            "directory": str(path),
            "total_items": len(subdirectories) + len(files),
            "subdirectories": subdirectories,
            "files": files,
            "errors": errors,
            "success": True,
        }


@tool(name="list_dir", description="列出目录内容，支持过滤和递归")
def list_dir(
    directory: str = ".",
    pattern: Optional[str] = None,
    show_hidden: bool = False,
    recursive: bool = False,
) -> str:
    """
    便捷的目录列表函数

    Args:
        directory: 目录路径
        pattern: 文件匹配模式
        show_hidden: 是否显示隐藏文件
        recursive: 是否递归

    Returns:
        目录列表JSON字符串
    """
    tool = ListDirectoryTool()
    result = tool.run(
        directory=directory,
        pattern=pattern,
        show_hidden=show_hidden,
        recursive=recursive,
    )
    return str(result.output) if result.success else f"Error: {result.error}"
