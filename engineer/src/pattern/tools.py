# tools.py
import datetime
import json
import os
from typing import Callable, Dict, Any, Union
from tavily import TavilyClient

# =========================
# 【关键】在本文件定义注册器（无需外部模块）
# =========================
_TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {}


def register_tool(name: str, description: str, signature: str = ""):
    """
    新增 signature 参数，例如 "search(query: str)"
    """

    def decorator(func):
        _TOOL_REGISTRY[name] = {
            "func": func,
            "description": description,
            "signature": signature or name + "()"  # 默认为 name()
        }
        return func

    return decorator


def get_all_tools() -> Dict[str, Dict[str, Any]]:
    """获取所有已注册工具"""
    return _TOOL_REGISTRY.copy()


# =========================
# 工具函数（使用本文件的 @register_tool）
# =========================

@register_tool(
    name="search",
    signature="search[input_data: Union[str, dict]] -> str",
    description="搜索信息"
)
def search(input_data: Union[str, dict]) -> str:
    if isinstance(input_data, dict):
        query = input_data.get("query", "")
        max_results = input_data.get("max_results", 3)
    else:
        query = str(input_data)
        max_results = 3

    if not query:
        return "Error: empty query"

    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    resp = client.search(query=query, search_depth="basic", max_results=max_results)
    results = resp.get("results", [])
    if not results:
        return "No relevant information found."
    return "\n".join(r["content"] for r in results)

@register_tool(
    name="get_current_date",
    signature="get_current_date[input_data: str] -> str",
    description="获取当前日期，格式 'YYYY-MM-DD'"
)
def get_current_date(input_data: str = "") -> str:
    return datetime.date.today().strftime("%Y-%m-%d")


# 在 tools.py 中注册
@register_tool(
    name="get_holiday_date",
    signature="get_holiday_date[holiday_name: str]",
    description="获取中国法定节假日的公历日期。支持春节、国庆等，输入如 '2026年春节'，返回 YYYY-MM-DD"
)
def get_holiday_date(holiday_name: str) -> str:
    """
    专为节假日设计的结构化工具。
    优先使用硬编码规则， fallback 到搜索 + LLM 提取（可选）。
    """
    # === 1. 硬编码常见节日（高可靠）===
    # 格式: "年份 + 节日名" -> "YYYY-MM-DD"
    KNOWN_HOLIDAYS = {
        "2024年春节": "2024-02-10",
        "2025年春节": "2025-01-29",
        "2026年春节": "2026-02-17",
        "2027年春节": "2027-02-06",
        "2024年国庆": "2024-10-01",
        "2025年国庆": "2025-10-01",
        "2026年国庆": "2026-10-01",
    }
    if holiday_name in KNOWN_HOLIDAYS:
        return KNOWN_HOLIDAYS[holiday_name]

    # === 3. 未找到
    return f"Error: 未找到 {holiday_name} 的日期"


# =========================
# ToolExecutor（使用本文件的注册表）
# ========================
class ToolExecutor:
    def __init__(self):
        self.tools: Dict[str, Callable] = {}  # func
        self.tool_info: Dict[str, Dict[str, str]] = {}  # full info: signature, description

        self._auto_register_tools()

    def _auto_register_tools(self):
        all_tools = get_all_tools()
        for name, info in all_tools.items():
            self.tools[name] = info["func"]
            self.tool_info[name] = {
                "signature": info["signature"],
                "description": info["description"]
            }

    def get_tool_descriptions(self) -> str:
        lines = []
        for name, info in self.tool_info.items():
            sig = info["signature"]
            desc = info["description"]
            lines.append(f"- {sig}")
            if desc:
                lines.append(f"  用途：{desc}")
        return "\n".join(lines)

    def execute_tool(self, name: str, input_data: str) -> str:
        if name not in self.tools:
            return f"Error: Tool '{name}' not found."

        func = self.tools[name]
        try:
            # 尝试解析 JSON
            if input_data.strip().startswith("{") and input_data.strip().endswith("}"):
                try:
                    kwargs = json.loads(input_data)
                    if isinstance(kwargs, dict):
                        return str(func(kwargs))  # 传 dict
                except (json.JSONDecodeError, TypeError):
                    pass

            # 默认：纯字符串
            return str(func(input_data))

        except Exception as e:
            return f"Tool execution error: {str(e)}"
