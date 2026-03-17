"""
文件系统工具 - 文件读写、编辑、搜索

提供安全的文件操作功能
"""

import fnmatch
import json
import os
import re
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from ..base import BaseTool, Tool, tool


class ReadFileInput(BaseModel):
    """读取文件输入参数"""

    file_path: str = Field(description="文件路径（绝对或相对路径）")
    encoding: str = Field(default="utf-8", description="文件编码")
    limit: Optional[int] = Field(default=None, description="读取的最大行数", ge=1)
    offset: int = Field(default=1, description="起始行号（1-based）", ge=1)


class WriteFileInput(BaseModel):
    """写入文件输入参数"""

    file_path: str = Field(description="文件路径")
    content: str = Field(description="文件内容")
    encoding: str = Field(default="utf-8", description="文件编码")
    append: bool = Field(default=False, description="是否追加模式")


class EditFileInput(BaseModel):
    """编辑文件输入参数"""

    file_path: str = Field(description="文件路径")
    old_string: str = Field(description="要被替换的字符串")
    new_string: str = Field(description="替换后的字符串")


class GlobInput(BaseModel):
    """文件搜索输入参数"""

    pattern: str = Field(description="glob模式，如 '*.py' 或 '**/*.txt'")
    path: Optional[str] = Field(default=".", description="搜索目录")
    recursive: bool = Field(default=True, description="是否递归搜索")


class GrepInput(BaseModel):
    """内容搜索输入参数"""

    pattern: str = Field(description="搜索模式（正则表达式）")
    path: str = Field(default=".", description="搜索路径（文件或目录）")
    include: Optional[str] = Field(default="*", description="文件匹配模式")
    output_mode: str = Field(default="content", description="输出模式: content/files/count")


class FileInfo(BaseModel):
    """文件信息"""

    path: str = Field(description="文件路径")
    size: int = Field(description="文件大小（字节）")
    modified: float = Field(description="修改时间戳")
    is_file: bool = Field(description="是否是文件")
    is_dir: bool = Field(description="是否是目录")


class ReadFileTool(BaseTool):
    """读取文件工具"""

    def __init__(self):
        super().__init__(
            name="read_file",
            description="读取文件内容，支持限制行数和指定起始行",
            args_schema=ReadFileInput,
        )

    def _run(
        self,
        file_path: str,
        encoding: str = "utf-8",
        limit: Optional[int] = None,
        offset: int = 1,
    ) -> str:
        """读取文件内容"""
        try:
            path = Path(file_path)

            if not path.exists():
                return json.dumps({"error": f"文件不存在: {file_path}"}, ensure_ascii=False)

            if not path.is_file():
                return json.dumps({"error": f"路径不是文件: {file_path}"}, ensure_ascii=False)

            with open(path, "r", encoding=encoding, errors="replace") as f:
                lines = f.readlines()

            # 处理行范围
            start_idx = max(0, offset - 1)
            end_idx = len(lines)
            if limit:
                end_idx = min(start_idx + limit, len(lines))

            selected_lines = lines[start_idx:end_idx]
            content = "".join(selected_lines)

            result = {
                "file_path": str(path.absolute()),
                "total_lines": len(lines),
                "start_line": start_idx + 1,
                "end_line": end_idx,
                "content": content,
            }

            return json.dumps(result, ensure_ascii=False, indent=2)

        except Exception as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)


class WriteFileTool(BaseTool):
    """写入文件工具"""

    def __init__(self):
        super().__init__(
            name="write_file",
            description="写入或追加文件内容",
            args_schema=WriteFileInput,
        )

    def _run(
        self,
        file_path: str,
        content: str,
        encoding: str = "utf-8",
        append: bool = False,
    ) -> str:
        """写入文件内容"""
        try:
            path = Path(file_path)

            # 确保父目录存在
            path.parent.mkdir(parents=True, exist_ok=True)

            mode = "a" if append else "w"
            with open(path, mode, encoding=encoding) as f:
                f.write(content)

            result = {
                "file_path": str(path.absolute()),
                "mode": "append" if append else "write",
                "bytes_written": len(content.encode(encoding)),
                "success": True,
            }

            return json.dumps(result, ensure_ascii=False, indent=2)

        except Exception as e:
            return json.dumps({"error": str(e), "success": False}, ensure_ascii=False)


class EditFileTool(BaseTool):
    """编辑文件工具"""

    def __init__(self):
        super().__init__(
            name="edit_file",
            description="编辑文件内容，替换指定的字符串",
            args_schema=EditFileInput,
        )

    def _run(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
    ) -> str:
        """编辑文件内容"""
        try:
            path = Path(file_path)

            if not path.exists():
                return json.dumps(
                    {"error": f"文件不存在: {file_path}", "success": False}, ensure_ascii=False
                )

            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            # 检查旧字符串是否存在
            if old_string not in content:
                return json.dumps(
                    {
                        "error": f"未找到要替换的字符串",
                        "file_path": str(path.absolute()),
                        "success": False,
                    },
                    ensure_ascii=False,
                )

            # 执行替换
            new_content = content.replace(old_string, new_string, 1)

            with open(path, "w", encoding="utf-8") as f:
                f.write(new_content)

            result = {
                "file_path": str(path.absolute()),
                "replacements": content.count(old_string),
                "success": True,
            }

            return json.dumps(result, ensure_ascii=False, indent=2)

        except Exception as e:
            return json.dumps({"error": str(e), "success": False}, ensure_ascii=False)


class GlobTool(BaseTool):
    """文件搜索工具"""

    def __init__(self):
        super().__init__(
            name="glob",
            description="使用glob模式搜索文件，如 '*.py' 或 'src/**/*.js'",
            args_schema=GlobInput,
        )

    def _run(
        self,
        pattern: str,
        path: str = ".",
        recursive: bool = True,
    ) -> str:
        """搜索文件"""
        try:
            base_path = Path(path)

            if not base_path.exists():
                return json.dumps({"error": f"路径不存在: {path}", "files": []}, ensure_ascii=False)

            matches = []

            if recursive:
                # 递归搜索
                for file_path in base_path.rglob(pattern):
                    matches.append(str(file_path))
            else:
                # 仅当前目录
                for file_path in base_path.glob(pattern):
                    matches.append(str(file_path))

            # 按字母顺序排序
            matches.sort()

            result = {
                "pattern": pattern,
                "path": str(base_path.absolute()),
                "recursive": recursive,
                "count": len(matches),
                "files": matches,
            }

            return json.dumps(result, ensure_ascii=False, indent=2)

        except Exception as e:
            return json.dumps({"error": str(e), "files": []}, ensure_ascii=False)


class GrepTool(BaseTool):
    """内容搜索工具"""

    def __init__(self):
        super().__init__(
            name="grep",
            description="在文件中搜索内容，支持正则表达式",
            args_schema=GrepInput,
        )

    def _run(
        self,
        pattern: str,
        path: str = ".",
        include: str = "*",
        output_mode: str = "content",
    ) -> str:
        """搜索文件内容"""
        try:
            import re as re_module

            search_path = Path(path)
            regex = re_module.compile(pattern, re_module.MULTILINE)

            matches = []

            if search_path.is_file():
                # 搜索单个文件
                files_to_search = [search_path]
            else:
                # 搜索目录
                files_to_search = [
                    f
                    for f in search_path.rglob("*")
                    if f.is_file() and fnmatch.fnmatch(f.name, include)
                ][:100]  # 限制文件数量

            for file_path in files_to_search:
                try:
                    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()

                    if regex.search(content):
                        if output_mode == "files_with_matches":
                            matches.append(str(file_path))
                        elif output_mode == "count":
                            count = len(regex.findall(content))
                            matches.append({"file": str(file_path), "count": count})
                        else:  # content
                            lines = content.split("\n")
                            for i, line in enumerate(lines, 1):
                                if regex.search(line):
                                    matches.append(
                                        {
                                            "file": str(file_path),
                                            "line": i,
                                            "content": line,
                                        }
                                    )
                except Exception:
                    continue

            result = {
                "pattern": pattern,
                "path": str(search_path.absolute()),
                "matches": matches,
                "total_matches": len(matches),
            }

            return json.dumps(result, ensure_ascii=False, indent=2)

        except Exception as e:
            return json.dumps({"error": str(e), "matches": []}, ensure_ascii=False)


# 便捷的函数版本
@tool(name="read", description="读取文件内容")
def read(file_path: str, limit: Optional[int] = None, offset: int = 1) -> str:
    """便捷的文件读取"""
    tool = ReadFileTool()
    result = tool.run(file_path=file_path, limit=limit, offset=offset)
    return str(result.output) if result.success else f"Error: {result.error}"


@tool(name="write", description="写入文件内容")
def write(file_path: str, content: str, append: bool = False) -> str:
    """便捷的文件写入"""
    tool = WriteFileTool()
    result = tool.run(file_path=file_path, content=content, append=append)
    return str(result.output) if result.success else f"Error: {result.error}"


@tool(name="edit", description="编辑文件，替换指定内容")
def edit(file_path: str, old_string: str, new_string: str) -> str:
    """便捷的文件编辑"""
    tool = EditFileTool()
    result = tool.run(file_path=file_path, old_string=old_string, new_string=new_string)
    return str(result.output) if result.success else f"Error: {result.error}"
