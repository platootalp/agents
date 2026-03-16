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

执行流程:
    1. 接收任务: 从主代理接收 Task 对象
    2. 分析规划: 分析任务需求，规划执行步骤
    3. 执行循环: 使用工具循环执行（invoke/stream）
    4. 生成结果总结: 基于执行历史和结果生成结构化的结果数据
    5. 返回结果: 返回 TaskResult 给主代理（或yield进度事件）

与主代理的交互:
    主代理 → 创建SubAgent → 分配Task → SubAgent.run() → 返回TaskResult
                                              ↓
                                        [可选] 进度回调 / 流式事件
                                              ↓
                                        主代理更新状态 / 实时显示

流式事件类型:
    - start: 任务开始
    - chunk: 输出片段（可实时显示）
    - tool_call: 工具调用开始
    - tool_result: 工具调用完成
    - complete: 任务完成（包含完整TaskResult）
    - error: 执行错误

共享组件:
    - ToolUseAgent: 提供对话循环和工具执行
    - MessageBuilder: 构建任务相关的系统提示
    - ToolParser: 解析工具调用

设计特点:
    - 单一职责: 每个SubAgent只处理一个任务
    - 可隔离: 独立的工具集和上下文
    - 可复用: SubAgent可以被池化复用
    - 可观察: 提供进度回调接口

使用场景:
    - 并行任务处理: 主代理同时创建多个SubAgent处理不同子任务
    - 专业化处理: 不同SubAgent配置不同工具集处理特定类型任务
    - 沙箱执行: SubAgent在受限环境中执行可能有风险的操作
"""

import json
from typing import Optional, List, Dict, Any, Callable, Iterator, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Handle imports for both direct execution and module import
try:
    # Try absolute imports first (when running from project root)
    from apps.engineer.learn.agent.core.model import Model
    from apps.engineer.learn.agent.core.tool import Tool
    from apps.engineer.learn.agent.tool_use_agent import ToolUseAgent
    from apps.engineer.learn.agent.core.utils import MessageBuilder
except ImportError:
    # Fall back to relative imports with path adjustment
    import sys
    from pathlib import Path

    # Add the engineer directory to path when running directly
    current_file = Path(__file__).resolve()
    engineer_dir = current_file.parent.parent.parent  # apps/engineer/
    if str(engineer_dir) not in sys.path:
        sys.path.insert(0, str(engineer_dir))

    from learn.agent.core.model import Model
    from learn.agent.core.tool import Tool
    from learn.agent.tool_use_agent import ToolUseAgent
    from learn.agent.core.utils import MessageBuilder


class TaskStatus(str, Enum):
    """任务执行状态"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """
    任务定义

    描述主代理分配给SubAgent的任务。
    包含任务描述、期望输出、上下文信息。
    """

    task_id: str
    description: str
    expected_output: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    max_steps: int = 10
    timeout: Optional[int] = None  # 秒
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
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
    """
    任务执行结果

    SubAgent完成任务后返回给主代理的结构化结果。
    包含执行状态、输出内容、结构化结果总结和元数据。
    """

    task_id: str
    status: TaskStatus
    output: str = ""  # 完整输出（原始文本输出）
    result_summary: Dict[str, Any] = field(default_factory=dict)  # 结构化结果总结
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    execution_time: float = 0.0  # 执行耗时（秒）
    steps_taken: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
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
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def get_result_value(self, key: str, default: Any = None) -> Any:
        """获取结果总结中的特定字段值"""
        return self.result_summary.get(key, default)


class SubAgent(ToolUseAgent):
    """
    子代理 - 执行主代理分配的特定任务

    架构位置:
        继承 ToolUseAgent → 复用对话循环和工具执行
        使用 Task 作为输入 → TaskResult 作为输出

    核心职责:
        1. 任务执行: 专注于完成单一任务
        2. 工具使用: 使用配置的工具集完成任务
        3. 进度报告: 向主代理报告执行进度
        4. 结果总结: 生成结构化结果总结供主代理决策

    共享组件:
        - ToolUseAgent: 提供 invoke/stream 接口
        - MessageBuilder: 构建任务相关的系统提示
        - Task/TaskResult: 标准化的任务输入输出格式

    关键方法:
        __init__(): 初始化SubAgent，配置任务和工具
        run(): 执行任务并返回结果
        _build_system_prompt(): 构建任务相关的系统提示
        _generate_result_summary(): 生成结构化结果总结

    工作流程:
        1. 初始化: 接收Task对象和工具集
        2. 构建提示: _build_system_prompt() 包含任务描述和期望输出
        3. 执行: run_conversation() 使用工具循环
        4. 收集结果: 记录工具调用和输出
        5. 生成结果总结: _generate_result_summary() 提炼关键信息
        6. 返回: TaskResult 包含完整结果和结构化结果总结

    扩展方式:
        自定义结果总结:
            重写 _generate_result_summary() 方法，提供特定领域的结构化结果

        添加进度回调:
            设置 progress_callback，在关键步骤调用主代理的回调函数

        自定义工具集:
            在初始化时传入特定的工具集，使SubAgent专业化

    示例:
        ```python
        # 创建任务
        task = Task(
            task_id="task_001",
            description="搜索Python最佳实践",
            expected_output="列出5个Python编程最佳实践"
        )

        # 创建SubAgent
        sub_agent = SubAgent(
            name="ResearchAgent",
            description="专门用于搜索和研究的代理",
            model=Model(),
            task=task,
            tools=[search_tool, summarize_tool]
        )

        # 执行任务
        result = sub_agent.run()

        # 使用结果
        print(result.result_summary)  # 获取结构化结果总结
        print(result.output)          # 获取完整输出
        ```
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
        tools: Optional[List[Tool]] = None,
        progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ):
        """
        初始化SubAgent

        Args:
            name: 代理名称
            description: 代理描述
            model: LLM模型
            max_steps: 最大执行步数
            temperature: 生成温度
            task: 要执行的任务
            tools: 可用工具列表
            progress_callback: 进度回调函数 (message, data)
        """
        super().__init__(name, description, model, tools, max_steps)
        self.temperature = temperature
        self.task = task
        self.progress_callback = progress_callback
        self._execution_history: List[Dict[str, Any]] = []
        self._start_time: Optional[datetime] = None

    def _build_system_prompt(self) -> str:
        """
        构建任务相关的系统提示

        基于Task对象构建包含任务描述、期望输出和上下文的系统提示。

        Returns:
            系统提示词
        """
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
        """
        报告执行进度

        如果设置了progress_callback，调用它通知主代理进度更新。

        Args:
            message: 进度消息
            data: 附加数据
        """
        if self.progress_callback:
            self.progress_callback(message, data or {})

    def _generate_result_summary(
        self, output: str, tool_calls: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        生成结构化结果总结

        基于执行结果和工具调用历史生成结构化的结果数据，
        供主代理快速理解任务完成情况和提取关键信息。

        子类可以重写此方法以提供特定领域的结构化结果。

        Args:
            output: 完整输出
            tool_calls: 工具调用历史

        Returns:
            结构化的结果字典，包含关键信息、统计数据等
        """
        # 基础结构：包含关键字段和统计信息
        tool_names = list(set(tc.get("name", "unknown") for tc in tool_calls))

        return {
            # 执行概况
            "status": "completed",
            "completion_time": datetime.now().isoformat(),
            # 输出摘要
            "output_length": len(output),
            "output_preview": output[:500] if len(output) > 500 else output,
            # 工具使用统计
            "tools_used": tool_names,
            "tool_calls_count": len(tool_calls),
            "tool_calls_details": [
                {"name": tc.get("name"), "args": tc.get("arguments")}
                for tc in tool_calls[:5]  # 最多记录前5个工具调用
            ],
            # 执行指标
            "execution_metrics": {
                "steps": len(tool_calls),
                "tools_count": len(tool_names),
            },
        }

    def run(self) -> TaskResult:
        """
        执行任务并返回结果

        SubAgent的核心方法，执行分配的任务并返回结构化的结果。

        执行流程:
            1. 记录开始时间
            2. 报告开始执行
            3. 使用ToolUseAgent的invoke执行任务
            4. 收集工具调用历史
            5. 生成执行摘要
            6. 构建TaskResult

        Returns:
            TaskResult 包含执行状态、输出、摘要等信息

        Raises:
            如果执行过程中发生异常，返回FAILED状态的TaskResult
        """
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
            # 执行任务
            output = self.invoke(self.task.description)

            # 计算执行时间
            execution_time = (datetime.now() - self._start_time).total_seconds()

            # 收集工具调用历史（从对话历史中提取）
            tool_calls = self._extract_tool_calls()

            # 生成结构化结果总结
            result_summary = self._generate_result_summary(output, tool_calls)

            # 报告完成
            self._report_progress(
                f"SubAgent '{self.name}' completed task '{self.task.task_id}'",
                {
                    "execution_time": execution_time,
                    "steps": len(tool_calls),
                },
            )

            return TaskResult(
                task_id=self.task.task_id,
                status=TaskStatus.COMPLETED,
                output=output,
                result_summary=result_summary,
                tool_calls=tool_calls,
                execution_time=execution_time,
                steps_taken=len(tool_calls),
                metadata={
                    "agent_name": self.name,
                    "temperature": self.temperature,
                },
            )

        except Exception as e:
            execution_time = (
                (datetime.now() - self._start_time).total_seconds() if self._start_time else 0
            )

            self._report_progress(
                f"SubAgent '{self.name}' failed task '{self.task.task_id}'", {"error": str(e)}
            )

            return TaskResult(
                task_id=self.task.task_id,
                status=TaskStatus.FAILED,
                error=str(e),
                execution_time=execution_time,
                metadata={"agent_name": self.name},
            )

    def _extract_tool_calls(self) -> List[Dict[str, Any]]:
        """
        从执行历史中提取工具调用

        遍历对话历史，提取所有工具调用的信息。

        Returns:
            工具调用列表
        """
        tool_calls = []
        # 从对话历史中提取工具调用信息
        # 注意: message_history 来自 BaseAgent/ConversationMixin
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
        """
        流式执行任务

        支持两种模式:
        1. 简单模式 (yield_progress=False): 直接返回完整输出字符串
        2. 流式模式 (yield_progress=True): 返回生成器，yield进度和输出片段

        Args:
            yield_progress: 是否yield中间进度，默认为False

        Returns:
            - yield_progress=False: 完整输出字符串
            - yield_progress=True: 生成器，yield Dict包含type和data

        流式模式输出格式:
            {"type": "start", "data": {"task_id": "..."}}
            {"type": "chunk", "data": {"content": "..."}}
            {"type": "tool_call", "data": {"name": "...", "args": "..."}}
            {"type": "tool_result", "data": {"name": "...", "result": "..."}}
            {"type": "complete", "data": TaskResult.to_dict()}
            {"type": "error", "data": {"error": "..."}}

        示例:
            ```python
            # 简单模式
            output = sub_agent.run_stream()
            print(output)

            # 流式模式
            for event in sub_agent.run_stream(yield_progress=True):
                if event["type"] == "chunk":
                    print(event["data"]["content"], end="")
                elif event["type"] == "complete":
                    result = TaskResult(**event["data"])
            ```
        """
        if not self.task:
            raise ValueError("No task assigned")

        if yield_progress:
            return self._run_stream_generator()
        else:
            return self._run_stream_simple()

    def _run_stream_simple(self) -> str:
        """简单流式执行，返回完整输出"""
        if not self.task:
            raise ValueError("No task assigned")

        self._start_time = datetime.now()
        self._report_progress(f"SubAgent '{self.name}' started stream task '{self.task.task_id}'")

        output = self.stream(self.task.description, reset=True)

        self._report_progress(f"SubAgent '{self.name}' completed stream task '{self.task.task_id}'")

        return output

    def _run_stream_generator(self) -> Iterator[Dict[str, Any]]:
        """
        流式执行生成器

        Yields进度事件，让主代理可以实时跟踪执行过程。
        """
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
                metadata={
                    "agent_name": self.name,
                    "temperature": self.temperature,
                },
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

            yield {
                "type": "error",
                "data": {"error": str(e), "execution_time": execution_time},
            }

    def _stream_conversation(self) -> Iterator[str]:
        """
        流式对话生成器

        从父类的stream方法中提取输出片段。
        注意：这需要ToolUseAgent支持真正的流式输出。
        """
        if not self.task:
            return

        full_output = self.stream(self.task.description, reset=True)

        for char in full_output:
            yield char


# ============================================================================
# 示例用法
# ============================================================================


def example_progress_callback(message: str, data: Dict[str, Any]):
    """示例进度回调函数"""
    print(f"[Progress] {message}")
    if data:
        print(f"  Data: {json.dumps(data, ensure_ascii=False)}")


def example_search_tool(args: str) -> str:
    """示例搜索工具"""
    try:
        parsed = json.loads(args)
        query = parsed.get("query", args)
    except:
        query = args

    # 模拟搜索结果
    return (
        f"搜索结果 for '{query}':\n1. Python最佳实践指南\n2. 高效Python编程技巧\n3. Python代码规范"
    )


def example_summarize_tool(args: str) -> str:
    """示例摘要工具"""
    try:
        parsed = json.loads(args)
        text = parsed.get("text", args)
    except:
        text = args

    # 模拟摘要
    lines = text.split("\n")
    return f"摘要: 找到{len(lines)}条结果，主要关注Python编程最佳实践"


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    # 创建工具
    tools = [
        Tool(name="search", description="搜索信息", func=example_search_tool),
        Tool(name="summarize", description="总结文本", func=example_summarize_tool),
    ]

    # 创建任务
    task = Task(
        task_id="research_001",
        description="搜索Python最佳实践并总结",
        expected_output="提供Python编程的最佳实践总结",
        context={"topic": "Python programming", "max_results": 5},
    )

    # 创建SubAgent
    sub_agent = SubAgent(
        name="ResearchSubAgent",
        description="专门用于研究和总结的子代理",
        model=Model(),
        task=task,
        tools=tools,
        max_steps=5,
        progress_callback=example_progress_callback,
    )

    # 执行任务 - 同步模式
    print("=" * 60)
    print("SubAgent 同步执行示例")
    print("=" * 60)

    result = sub_agent.run()

    print("\n" + "=" * 60)
    print("执行结果")
    print("=" * 60)
    print(f"\n任务ID: {result.task_id}")
    print(f"状态: {result.status.value}")
    print(f"执行时间: {result.execution_time:.2f}秒")
    print(f"工具调用次数: {result.steps_taken}")
    print(f"\n结构化结果总结:")
    print(json.dumps(result.result_summary, ensure_ascii=False, indent=2))

    if result.error:
        print(f"\n错误: {result.error}")

    # 流式执行示例
    print("\n" + "=" * 60)
    print("SubAgent 流式执行示例")
    print("=" * 60)

    # 简单流式模式
    print("\n1. 简单流式模式 (直接返回输出):")
    stream_output = sub_agent.run_stream(yield_progress=False)
    if isinstance(stream_output, str):
        print(f"流式输出长度: {len(stream_output)} 字符")

    # 进度流式模式
    print("\n2. 进度流式模式 (yield事件):")
    print("开始流式执行...")

    stream_result = sub_agent.run_stream(yield_progress=True)
    if hasattr(stream_result, "__iter__"):
        for event in stream_result:
            if not isinstance(event, dict):
                continue

            event_type = event.get("type", "")
            data = event.get("data", {})

            if event_type == "start":
                print(f"[开始] 任务ID: {data.get('task_id', 'unknown')}")

            elif event_type == "chunk":
                # 输出片段 (可以实时显示)
                content = data.get("content", "")
                print(content, end="", flush=True)

            elif event_type == "complete":
                print(f"\n[完成] 状态: {data.get('status', 'unknown')}")
                exec_time = data.get("execution_time", 0)
                print(f"执行时间: {exec_time:.2f}秒")

            elif event_type == "error":
                print(f"\n[错误] {data.get('error', 'Unknown error')}")
