"""
内置工具集
提供常用的预定义工具，如计算器、搜索、文件操作等
"""

# genAI_main_start
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import json
import os
import datetime
import math
from .decorators import tool, structured_tool


# ==================== 计算器工具 ====================

class CalculatorInput(BaseModel):
    """计算器输入参数"""
    expression: str = Field(description="要计算的数学表达式，如 '2 + 2' 或 'math.sqrt(16)'")


@structured_tool(description="计算数学表达式，支持基本运算和math库函数")
def calculator(input: CalculatorInput) -> str:
    """计算数学表达式
    
    支持的操作：
    - 基本运算: +, -, *, /, //, %, **
    - math库函数: sqrt, sin, cos, tan, log, exp等
    
    Args:
        input: 包含expression的输入对象
    
    Returns:
        计算结果的字符串表示
    
    Examples:
        >>> calculator.run(expression="2 + 2")
        >>> calculator.run(expression="math.sqrt(16)")
    """
    try:
        # 安全的计算环境，只允许math模块和基本运算
        safe_dict = {
            "__builtins__": {},
            "math": math,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "pow": pow
        }
        result = eval(input.expression, safe_dict)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


# ==================== 文件操作工具 ====================

class FileReadInput(BaseModel):
    """文件读取输入参数"""
    file_path: str = Field(description="要读取的文件路径")
    encoding: str = Field(default="utf-8", description="文件编码")


@structured_tool(description="读取文件内容")
def read_file(input: FileReadInput) -> str:
    """读取文件内容
    
    Args:
        input: 包含file_path和encoding的输入对象
    
    Returns:
        文件内容或错误信息
    """
    try:
        with open(input.file_path, 'r', encoding=input.encoding) as f:
            content = f.read()
        return f"文件内容:\n{content}"
    except Exception as e:
        return f"读取文件失败: {str(e)}"


class FileWriteInput(BaseModel):
    """文件写入输入参数"""
    file_path: str = Field(description="要写入的文件路径")
    content: str = Field(description="要写入的内容")
    encoding: str = Field(default="utf-8", description="文件编码")
    mode: str = Field(default="w", description="写入模式: 'w'覆盖, 'a'追加")


@structured_tool(description="写入内容到文件")
def write_file(input: FileWriteInput) -> str:
    """写入内容到文件
    
    Args:
        input: 包含file_path、content等的输入对象
    
    Returns:
        操作结果信息
    """
    try:
        os.makedirs(os.path.dirname(input.file_path) or '.', exist_ok=True)
        with open(input.file_path, input.mode, encoding=input.encoding) as f:
            f.write(input.content)
        return f"成功写入文件: {input.file_path}"
    except Exception as e:
        return f"写入文件失败: {str(e)}"


class ListDirectoryInput(BaseModel):
    """列出目录输入参数"""
    directory: str = Field(default=".", description="要列出的目录路径")
    pattern: Optional[str] = Field(default=None, description="文件名过滤模式（可选）")


@structured_tool(description="列出目录中的文件和子目录")
def list_directory(input: ListDirectoryInput) -> str:
    """列出目录内容
    
    Args:
        input: 包含directory和pattern的输入对象
    
    Returns:
        目录内容列表或错误信息
    """
    try:
        import fnmatch
        items = os.listdir(input.directory)
        
        if input.pattern:
            items = [item for item in items if fnmatch.fnmatch(item, input.pattern)]
        
        # 分类显示
        dirs = [item for item in items if os.path.isdir(os.path.join(input.directory, item))]
        files = [item for item in items if os.path.isfile(os.path.join(input.directory, item))]
        
        result = f"目录: {input.directory}\n\n"
        result += f"子目录 ({len(dirs)}):\n"
        result += "\n".join(f"  📁 {d}" for d in sorted(dirs))
        result += f"\n\n文件 ({len(files)}):\n"
        result += "\n".join(f"  📄 {f}" for f in sorted(files))
        
        return result
    except Exception as e:
        return f"列出目录失败: {str(e)}"


# ==================== 时间日期工具 ====================

@tool(description="获取当前日期和时间")
def get_current_time() -> str:
    """获取当前日期和时间
    
    Returns:
        格式化的当前日期时间字符串
    """
    now = datetime.datetime.now()
    return now.strftime("%Y年%m月%d日 %H:%M:%S")


@tool(description="获取当前日期")
def get_current_date() -> str:
    """获取当前日期
    
    Returns:
        格式化的当前日期字符串
    """
    today = datetime.date.today()
    return today.strftime("%Y年%m月%d日")


class DateCalculatorInput(BaseModel):
    """日期计算输入参数"""
    days: int = Field(description="要加减的天数（正数为加，负数为减）")
    from_date: Optional[str] = Field(default=None, description="起始日期（格式: YYYY-MM-DD），默认为今天")


@structured_tool(description="计算日期加减")
def date_calculator(input: DateCalculatorInput) -> str:
    """计算日期加减
    
    Args:
        input: 包含days和from_date的输入对象
    
    Returns:
        计算后的日期字符串
    """
    try:
        if input.from_date:
            base_date = datetime.datetime.strptime(input.from_date, "%Y-%m-%d").date()
        else:
            base_date = datetime.date.today()
        
        result_date = base_date + datetime.timedelta(days=input.days)
        return f"{base_date} + {input.days}天 = {result_date}"
    except Exception as e:
        return f"日期计算失败: {str(e)}"


# ==================== 文本处理工具 ====================

class TextLengthInput(BaseModel):
    """文本长度输入参数"""
    text: str = Field(description="要计算长度的文本")


@structured_tool(description="计算文本长度（字符数和单词数）")
def text_length(input: TextLengthInput) -> str:
    """计算文本长度
    
    Args:
        input: 包含text的输入对象
    
    Returns:
        文本长度统计信息
    """
    char_count = len(input.text)
    word_count = len(input.text.split())
    line_count = len(input.text.splitlines())
    
    return f"字符数: {char_count}\n单词数: {word_count}\n行数: {line_count}"


class TextReplaceInput(BaseModel):
    """文本替换输入参数"""
    text: str = Field(description="原始文本")
    old: str = Field(description="要替换的文本")
    new: str = Field(description="替换后的文本")


@structured_tool(description="文本替换")
def text_replace(input: TextReplaceInput) -> str:
    """文本替换
    
    Args:
        input: 包含text、old、new的输入对象
    
    Returns:
        替换后的文本
    """
    result = input.text.replace(input.old, input.new)
    count = input.text.count(input.old)
    return f"已替换{count}处\n\n结果:\n{result}"


# ==================== JSON工具 ====================

class JsonParseInput(BaseModel):
    """JSON解析输入参数"""
    json_string: str = Field(description="要解析的JSON字符串")


@structured_tool(description="解析JSON字符串")
def json_parse(input: JsonParseInput) -> str:
    """解析JSON字符串
    
    Args:
        input: 包含json_string的输入对象
    
    Returns:
        格式化的JSON内容或错误信息
    """
    try:
        data = json.loads(input.json_string)
        formatted = json.dumps(data, ensure_ascii=False, indent=2)
        return f"JSON解析成功:\n{formatted}"
    except Exception as e:
        return f"JSON解析失败: {str(e)}"


class JsonExtractInput(BaseModel):
    """JSON提取输入参数"""
    json_string: str = Field(description="JSON字符串")
    key_path: str = Field(description="键路径，用点分隔，如 'user.name'")


@structured_tool(description="从JSON中提取指定键的值")
def json_extract(input: JsonExtractInput) -> str:
    """从JSON中提取值
    
    Args:
        input: 包含json_string和key_path的输入对象
    
    Returns:
        提取的值或错误信息
    """
    try:
        data = json.loads(input.json_string)
        keys = input.key_path.split('.')
        
        value = data
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            elif isinstance(value, list) and key.isdigit():
                value = value[int(key)]
            else:
                return f"无法访问路径: {input.key_path}"
        
        return f"提取的值: {json.dumps(value, ensure_ascii=False, indent=2)}"
    except Exception as e:
        return f"提取失败: {str(e)}"


# ==================== 工具集合 ====================

def get_all_builtin_tools() -> List[Any]:
    """获取所有内置工具
    
    Returns:
        内置工具列表
    """
    return [
        # 计算器
        calculator,
        # 文件操作
        read_file,
        write_file,
        list_directory,
        # 时间日期
        get_current_time,
        get_current_date,
        date_calculator,
        # 文本处理
        text_length,
        text_replace,
        # JSON工具
        json_parse,
        json_extract,
    ]


def get_builtin_tool_by_name(name: str) -> Optional[Any]:
    """根据名称获取内置工具
    
    Args:
        name: 工具名称
    
    Returns:
        工具实例或None
    """
    tools = get_all_builtin_tools()
    for tool in tools:
        if tool.name == name:
            return tool
    return None


def list_builtin_tools() -> str:
    """列出所有内置工具
    
    Returns:
        格式化的工具列表字符串
    """
    tools = get_all_builtin_tools()
    result = "=== 内置工具列表 ===\n\n"
    
    categories = {
        "计算器": ["calculator"],
        "文件操作": ["read_file", "write_file", "list_directory"],
        "时间日期": ["get_current_time", "get_current_date", "date_calculator"],
        "文本处理": ["text_length", "text_replace"],
        "JSON工具": ["json_parse", "json_extract"],
    }

    for category, tool_names in categories.items():
        result += f"\n📦 {category}:\n"
        for tool in tools:
            if tool.name in tool_names:
                result += f"  • {tool.name}: {tool.description}\n"
    
    return result
# genAI_main_end

