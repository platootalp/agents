"""
SubAgent - 子代理，用于处理主代理分配的子任务

架构设计:
=========

角色定位:
    SubAgent 是一个专门的代理，负责执行主代理（Main Agent）分解后的子任务。
    它专注于单一任务的完整执行，并返回结构化的执行摘要。

继承关系:
    ToolUseAgent (tool_use_agent.py)
        ↓ 继承
    SubAgent (当前文件)

核心职责:
    1. 任务执行: 接收并执行主代理分配的具体任务
    2. 工具使用: 使用可用工具完成复杂操作
    3. 进度报告: 向主代理报告执行进度和中间结果
    4. 结构化结果: 生成结构化的任务结果总结，供主Agent提取关键信息

数据模型:
    Task: 任务定义，包含描述、上下文、期望输出
    TaskResult: 任务执行结果，包含状态、输出、摘要、元数据

执行模式:
    1. 同步模式 (run()): 阻塞执行，完成后返回 TaskResult
    2. 简单流式 (run_stream()): 流式输出，返回完整字符串
    3. 进度流式 (run_stream(yield_progress=True)): 生成器模式，yield事件

共享组件:
    - ToolUseAgent: 提供对话循环和工具执行
    - MessageBuilder: 构建任务相关的系统提示

设计特点:
    - 单一职责: 每个SubAgent只处理一个任务
    - 可隔离: 独立的工具集和上下文
    - 可复用: SubAgent可以被池化复用
    - 可观察: 提供进度回调接口
"""

import json
from typing import Optional, List, Dict, Any, Callable, Iterator, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# 新工具系统导入
try:
    from apps.engineer.learn.coder.core.tools.base import BaseTool, Tool, ToolResult
except ImportError:
    from learn.coder.core.tools.base import BaseTool, Tool, ToolResult

# Agent基础导入
try:
    from apps.engineer.learn.coder.core.model import Model
    from apps.engineer.learn.coder.agents.tool_use_agent import ToolUseAgent
    from apps.engineer.learn.coder.core.utils import MessageBuilder
except ImportError:
    from learn.coder.core.model import Model
    from learn.coder.agents.tool_use_agent import ToolUseAgent
    from learn.coder.core.utils import MessageBuilder


class TaskStatus(str, Enum):
    """任务执行状态"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """任务定义 - 描述主代理分配给SubAgent的任务"""

    task_id: str
    description: str
    expected_output: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    max_steps: int = 10
    timeout: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "expected_output": self.expected_output,
            "context": self.context,
            "max_steps": self.max_steps,
            "timeout": self.timeout,
            "metadata": self.metadata,
        }


@dataclass
class TaskResult:
    """任务执行结果 - SubAgent完成任务后返回给主代理的结构化结果"""

    task_id: str
    status: TaskStatus
    output: str = ""
    result_summary: Dict[str, Any] = field(default_factory=dict)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    execution_time: float = 0.0
    steps_taken: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "output": self.output,
            "result_summary": self.result_summary,
            "tool_calls": self.tool_calls,
            "execution_time": self.execution_time,
            "steps_taken": self.steps_taken,
            "metadata": self.metadata,
            "error": self.error,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


class SubAgent(ToolUseAgent):
    """
    子代理 - 执行主代理分配的特定任务

    继承 ToolUseAgent，使用 Task 作为输入，TaskResult 作为输出。
    支持 BaseTool 工具类型。
    """

    DEFAULT_SYSTEM_PROMPT_TEMPLATE = """你是一个专门执行特定任务的代理。

你的任务是:
{task_description}

期望输出:
{expected_output}

任务上下文:
{context}

执行指南:
1. 仔细分析任务需求
2. 使用可用工具高效完成任务
3. 确保输出符合期望格式
4. 如有问题，明确报告失败原因

当前时间: {current_time}
"""

    def __init__(
        self,
        name: str,
        description: str = "",
        model: Optional[Model] = None,
        max_steps: int = 10,
        temperature: float = 0.3,
        task: Optional[Task] = None,
        tools: Optional[List[BaseTool]] = None,
        progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ):
        super().__init__(name, description, model, tools, max_steps)
        self.temperature = temperature
        self.task = task
        self.progress_callback = progress_callback
        self._execution_history: List[Dict[str, Any]] = []
        self._start_time: Optional[datetime] = None

    def _build_system_prompt(self) -> str:
        if self.task:
            context_str = json.dumps(self.task.context, ensure_ascii=False, indent=2)
            return self.DEFAULT_SYSTEM_PROMPT_TEMPLATE.format(
                task_description=self.task.description,
                expected_output=self.task.expected_output or "完成指定任务",
                context=context_str if context_str != "{}" else "无",
                current_time=datetime.now().isoformat(),
            )
        return super()._build_system_prompt()

    def _report_progress(self, message: str, data: Optional[Dict[str, Any]] = None):
        if self.progress_callback:
            self.progress_callback(message, data or {})

    def _generate_result_summary(
        self, output: str, tool_calls: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        tool_names = list(set(tc.get("name", "unknown") for tc in tool_calls))
        return {
            "status": "completed",
            "completion_time": datetime.now().isoformat(),
            "output_length": len(output),
            "output_preview": output[:500] if len(output) > 500 else output,
            "tools_used": tool_names,
            "tool_calls_count": len(tool_calls),
            "tool_calls_details": [
                {"name": tc.get("name"), "args": tc.get("arguments")} for tc in tool_calls[:5]
            ],
            "execution_metrics": {
                "steps": len(tool_calls),
                "tools_count": len(tool_names),
            },
        }

    def run(self) -> TaskResult:
        if not self.task:
            return TaskResult(
                task_id="unknown",
                status=TaskStatus.FAILED,
                error="No task assigned",
            )

        self._start_time = datetime.now()
        self._report_progress(
            f"SubAgent '{self.name}' started task '{self.task.task_id}'",
            {"task": self.task.to_dict()},
        )

        try:
            output = self.invoke(self.task.description)
            execution_time = (datetime.now() - self._start_time).total_seconds()
            tool_calls = self._extract_tool_calls()
            result_summary = self._generate_result_summary(output, tool_calls)

            self._report_progress(
                f"SubAgent '{self.name}' completed task '{self.task.task_id}'",
                {"execution_time": execution_time, "steps": len(tool_calls)},
            )

            return TaskResult(
                task_id=self.task.task_id,
                status=TaskStatus.COMPLETED,
                output=output,
                result_summary=result_summary,
                tool_calls=tool_calls,
                execution_time=execution_time,
                steps_taken=len(tool_calls),
                metadata={"agent_name": self.name, "temperature": self.temperature},
            )
        except Exception as e:
            execution_time = (
                (datetime.now() - self._start_time).total_seconds() if self._start_time else 0
            )
            self._report_progress(
                f"SubAgent '{self.name}' failed task '{self.task.task_id}'",
                {"error": str(e)},
            )
            return TaskResult(
                task_id=self.task.task_id,
                status=TaskStatus.FAILED,
                error=str(e),
                execution_time=execution_time,
                metadata={"agent_name": self.name},
            )

    def _extract_tool_calls(self) -> List[Dict[str, Any]]:
        tool_calls = []
        for msg in getattr(self, "message_history", []):
            if msg.get("role") == "assistant":
                tool_calls_data = msg.get("tool_calls", [])
                for tc in tool_calls_data:
                    tool_calls.append(
                        {
                            "id": tc.get("id"),
                            "name": tc.get("function", {}).get("name"),
                            "arguments": tc.get("function", {}).get("arguments"),
                        }
                    )
        return tool_calls

    def run_stream(self, yield_progress: bool = False) -> Union[str, Iterator[Dict[str, Any]]]:
        if not self.task:
            raise ValueError("No task assigned")
        if yield_progress:
            return self._run_stream_generator()
        else:
            return self._run_stream_simple()

    def _run_stream_simple(self) -> str:
        self._start_time = datetime.now()
        self._report_progress(f"SubAgent '{self.name}' started stream task '{self.task.task_id}'")
        output = self.stream(self.task.description, reset=True)
        self._report_progress(f"SubAgent '{self.name}' completed stream task '{self.task.task_id}'")
        return output

    def _run_stream_generator(self) -> Iterator[Dict[str, Any]]:
        if not self.task:
            yield {"type": "error", "data": {"error": "No task assigned"}}
            return

        self._start_time = datetime.now()
        yield {
            "type": "start",
            "data": {
                "task_id": self.task.task_id,
                "agent_name": self.name,
                "timestamp": datetime.now().isoformat(),
            },
        }

        self._report_progress(f"SubAgent '{self.name}' started stream task '{self.task.task_id}'")

        try:
            output_chunks = []
            for chunk in self._stream_conversation():
                output_chunks.append(chunk)
                yield {"type": "chunk", "data": {"content": chunk}}

            output = "".join(output_chunks)
            execution_time = (datetime.now() - self._start_time).total_seconds()
            tool_calls = self._extract_tool_calls()
            result_summary = self._generate_result_summary(output, tool_calls)

            result = TaskResult(
                task_id=self.task.task_id,
                status=TaskStatus.COMPLETED,
                output=output,
                result_summary=result_summary,
                tool_calls=tool_calls,
                execution_time=execution_time,
                steps_taken=len(tool_calls),
                metadata={"agent_name": self.name, "temperature": self.temperature},
            )

            self._report_progress(
                f"SubAgent '{self.name}' completed stream task '{self.task.task_id}'",
                {"execution_time": execution_time, "steps": len(tool_calls)},
            )

            yield {"type": "complete", "data": result.to_dict()}

        except Exception as e:
            execution_time = (
                (datetime.now() - self._start_time).total_seconds() if self._start_time else 0
            )
            self._report_progress(
                f"SubAgent '{self.name}' failed task '{self.task.task_id}'",
                {"error": str(e)},
            )
            yield {"type": "error", "data": {"error": str(e), "execution_time": execution_time}}

    def _stream_conversation(self) -> Iterator[str]:
        if not self.task:
            return
        full_output = self.stream(self.task.description, reset=True)
        for char in full_output:
            yield char
