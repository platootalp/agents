"""
Pattern 包 - 各种 Agent 范式实现

提供统一的 BaseAgent 基类，以及多种具体的 Agent 实现：
- ReflectionAgent: 反思式 Agent
- LLMCompilerAgent: LLM 编译器 Agent（DAG 并行编排）
- ReActAgent: ReAct 范式 Agent
- ReWOOAgent: ReWOO 范式 Agent
- PlanSolveAgent: 规划-求解 Agent
- SelfDiscoverAgent: 自我发现 Agent
"""
# genAI_main_start
from pattern.base_agent import BaseAgent
from pattern.reflection import ReflectionAgent
from pattern.llm_compiler import LLMCompilerAgent
from pattern.react import ReActAgent
from pattern.rewoo import ReWOOAgent
from pattern.plan_and_solve import PlanSolveAgent
from pattern.self_discover import SelfDiscoverAgent
from pattern.tools import ToolExecutor, register_tool, get_all_tools

__all__ = [
    "BaseAgent",
    "ReflectionAgent",
    "LLMCompilerAgent",
    "ReActAgent",
    "ReWOOAgent",
    "PlanSolveAgent",
    "SelfDiscoverAgent",
    "ToolExecutor",
    "register_tool",
    "get_all_tools",
]
# genAI_main_end
