"""
Agent 模块 - 使用新工具系统的各种 Agent 实现

本模块提供多种 Agent 实现，所有实现都基于 coder.core.tools 中的新工具系统：
- ToolUseAgent: 基础工具使用 Agent
- McpAgent: MCP (Model Context Protocol) Agent
- SkillsUseAgent: Skill 系统 Agent
- SubAgent: 子任务 Agent
- TaskAgent: 任务规划与执行 Agent
"""

# 新工具系统导入
try:
    from apps.engineer.coder.core.tools.base import BaseTool, Tool, ToolResult, tool
    from apps.engineer.coder.core.tools.manager import ToolManager, ToolExecutor
except ImportError:
    from apps.engineer.coder.core.tools.base import BaseTool, Tool, ToolResult, tool
    from apps.engineer.coder.core.tools.manager import ToolManager, ToolExecutor

# Agent 导入
try:
    from apps.engineer.coder.agents.tool_use_agent import ToolUseAgent
    from apps.engineer.coder.agents.mcp_agent import McpAgent, StreamChunk
    from apps.engineer.coder.agents.skills_agent import (
        SkillsUseAgent,
        Skills,
        SkillsRepository,
        SkillsToolSet,
    )
    from apps.engineer.coder.agents.subagent_agent import (
        SubAgent,
        Task,
        TaskResult,
        TaskStatus,
    )
    from apps.engineer.coder.agents.task_agent import (
        TaskAgent,
        Task as TaskAgentTask,
        TaskGraph,
        TaskStatus as TaskAgentStatus,
        TaskPriority,
    )
except ImportError:
    from apps.engineer.coder.agents.tool_use_agent import ToolUseAgent
    from apps.engineer.coder.agents.mcp_agent import McpAgent, StreamChunk
    from apps.engineer.coder.agents.skills_agent import (
        SkillsUseAgent,
        Skills,
        SkillsRepository,
        SkillsToolSet,
    )
    from apps.engineer.coder.agents.subagent_agent import (
        SubAgent,
        Task,
        TaskResult,
        TaskStatus,
    )
    from apps.engineer.coder.agents.task_agent import (
        TaskAgent,
        Task as TaskAgentTask,
        TaskGraph,
        TaskStatus as TaskAgentStatus,
        TaskPriority,
    )

__all__ = [
    # 工具系统 (来自新工具模块)
    "BaseTool",
    "Tool",
    "ToolResult",
    "tool",
    "ToolManager",
    "ToolExecutor",
    # Agents
    "ToolUseAgent",
    "McpAgent",
    "SkillsUseAgent",
    "SubAgent",
    "TaskAgent",
    # 数据模型
    "StreamChunk",
    "Skills",
    "SkillsRepository",
    "SkillsToolSet",
    "Task",
    "TaskResult",
    "TaskStatus",
    "TaskAgentTask",
    "TaskGraph",
    "TaskAgentStatus",
    "TaskPriority",
]
