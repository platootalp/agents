"""
TaskAgent - 任务规划与执行Agent

架构设计:
=========

核心概念:
    TaskAgent 使用 Plan-and-Solve 模式，通过工具循环（非ReAct）实现任务管理。
    LLM 通过调用工具来规划、执行和跟踪任务，而不是在提示词中进行推理。

继承关系:
    ToolUseAgent (01.tool_use_agent.py)
        ↓ 继承
    TaskAgent (当前文件)

数据模型:
    Task (Pydantic BaseModel)
        - 任务实体，包含完整任务信息
        - 字段: task_id, name, description, status, priority
        - 关系: parent_id, subtasks, depends_on, blocks
        - 执行: requirements, context, result, error, tool_calls
        - 时间戳: create_time, start_time, complete_time, update_time

    TaskGraph
        - 任务图管理，维护任务依赖关系
        - 提供拓扑排序确定执行顺序
        - 管理任务状态传播

任务状态机:
    PENDING → RUNNING → COMPLETED
        ↓           ↓
    BLOCKED ← FAILED

    - PENDING: 等待执行
    - RUNNING: 正在执行
    - COMPLETED: 成功完成
    - FAILED: 执行失败
    - BLOCKED: 被依赖任务阻塞

工具集 (TaskToolkit):
    LLM可调用的工具，实现任务管理CRUD:
    - create_task: 创建新任务
    - decompose_task: 分解任务为子任务
    - update_task: 更新任务信息
    - complete_task: 标记任务完成
    - fail_task: 标记任务失败
    - list_tasks: 列出所有任务
    - get_task_details: 获取任务详情
    - get_next_task: 获取下一个可执行任务

共享组件使用:
    - ToolUseAgent: 提供对话循环和工具执行基础
    - MessageBuilder: 构建包含任务上下文的系统提示
    - ToolUseAgent.tools: 注册TaskToolkit工具供LLM使用

设计模式:
    1. Plan-and-Solve: LLM规划 → 工具执行 → 结果反馈
    2. 状态机: TaskStatus枚举管理任务生命周期
    3. 图结构: TaskGraph管理任务依赖关系
    4. 工具即API: 任务管理通过工具暴露给LLM

工作流程:
    1. 规划阶段:
       User Goal → LLM → create_task/decompose_task → TaskGraph

    2. 执行阶段:
       TaskGraph.get_next_task() → LLM → 工具调用/代码生成 → complete_task

    3. 跟踪阶段:
       更新任务状态 → 检查依赖 → 解锁阻塞任务 → 继续执行

    完整流程:
    User Goal
        ↓
    _build_system_prompt() (包含当前任务状态)
        ↓
    LLM 规划 → create_task/decompose_task
        ↓
    循环:
        get_next_task() → LLM 执行 → 工具调用 → complete_task/update_task
        ↓
    所有任务完成 → 生成最终报告

关键方法:
    _build_system_prompt(): 构建包含任务图和上下文的系统提示
    run(): 主执行循环，协调规划和执行
    _generate_report(): 生成执行报告

扩展指南:
    自定义任务类型:
        1. 继承 Task 添加自定义字段
        2. 在TaskToolkit中添加对应工具
        3. 更新 _build_system_prompt() 描述新类型

    集成其他系统:
        1. 在工具中调用外部API
        2. 将结果保存到Task.context
        3. LLM自动在后续步骤中使用上下文
"""

import json
import re
import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

try:
    from apps.engineer.learn.agent.core.model import Model
    from apps.engineer.learn.agent.core.tool import Tool
    from apps.engineer.learn.agent.tool_use_agent import ToolUseAgent
    from apps.engineer.learn.agent.core.utils import MessageBuilder
except ImportError:
    from learn.agent.core.model import Model
    from learn.agent.core.tool import Tool
    from learn.agent.tool_use_agent import ToolUseAgent
    from learn.agent.core.utils import MessageBuilder


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class TaskPriority(int, Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class Task(BaseModel):
    """任务实体 - 包含执行历史和上下文"""

    task_id: str = Field(description="唯一任务ID")
    name: str = Field(description="任务名称")
    description: str = Field(description="任务描述")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="当前状态")
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM, description="优先级")

    # 层级关系
    parent_id: str | None = Field(default=None, description="父任务ID")
    subtasks: list[str] = Field(default_factory=list, description="子任务ID列表")

    # 依赖关系
    depends_on: list[str] = Field(default_factory=list, description="依赖的任务ID")
    blocks: list[str] = Field(default_factory=list, description="阻塞的任务ID")

    # 执行相关
    requirements: list[str] = Field(default_factory=list, description="完成标准")
    context: dict[str, Any] = Field(default_factory=dict, description="任务上下文")
    result: str | None = Field(default=None, description="执行结果")
    error: str | None = Field(default=None, description="错误信息")
    tool_calls: list[dict[str, Any]] = Field(
        default_factory=list, description="执行时的工具调用历史"
    )

    # 元数据
    create_time: datetime = Field(default_factory=datetime.now)
    start_time: datetime | None = Field(default=None)
    complete_time: datetime | None = Field(default=None)
    update_time: datetime = Field(default_factory=datetime.now)

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """自定义序列化"""
        data = super().model_dump(**kwargs)
        for field in ["create_time", "start_time", "complete_time", "update_time"]:
            if isinstance(data.get(field), datetime):
                data[field] = data[field].isoformat()
        return data


class TaskGraph:
    """任务图 - 管理任务依赖和执行顺序"""

    def __init__(self):
        self.tasks: dict[str, Task] = {}

    def add_task(self, task: Task) -> None:
        """添加任务"""
        self.tasks[task.task_id] = task

    def get_task(self, task_id: str) -> Task | None:
        """获取任务"""
        return self.tasks.get(task_id)

    def remove_task(self, task_id: str) -> bool:
        """删除任务"""
        if task_id in self.tasks:
            # 清理依赖关系
            task = self.tasks[task_id]
            for dep_id in task.depends_on:
                dep_task = self.tasks.get(dep_id)
                if dep_task and task_id in dep_task.blocks:
                    dep_task.blocks.remove(task_id)
            for block_id in task.blocks:
                block_task = self.tasks.get(block_id)
                if block_task and task_id in block_task.depends_on:
                    block_task.depends_on.remove(task_id)
            # 从父任务中移除
            if task.parent_id and task.parent_id in self.tasks:
                parent = self.tasks[task.parent_id]
                if task_id in parent.subtasks:
                    parent.subtasks.remove(task_id)
            del self.tasks[task_id]
            return True
        return False

    def get_ready_tasks(self) -> list[Task]:
        """获取可执行的任务（pending + 依赖已满足）"""
        ready = []
        for task in self.tasks.values():
            if task.status == TaskStatus.PENDING:
                deps_met = all(
                    self.tasks.get(dep_id) and self.tasks[dep_id].status == TaskStatus.COMPLETED
                    for dep_id in task.depends_on
                )
                if deps_met:
                    ready.append(task)
        return sorted(ready, key=lambda t: t.priority.value, reverse=True)

    def get_next_task(self) -> Task | None:
        """获取下一个最高优先级的可执行任务"""
        ready = self.get_ready_tasks()
        return ready[0] if ready else None

    def update_task_status(
        self, task_id: str, status: TaskStatus, result: str | None = None, error: str | None = None
    ) -> bool:
        """更新任务状态并传播到阻塞的任务"""
        task = self.tasks.get(task_id)
        if not task:
            return False

        task.status = status
        task.update_time = datetime.now()

        if status == TaskStatus.RUNNING and not task.start_time:
            task.start_time = datetime.now()
        elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            task.complete_time = datetime.now()

        if result:
            task.result = result
        if error:
            task.error = error

        # 更新被阻塞的任务
        if status == TaskStatus.COMPLETED:
            for blocked_id in task.blocks:
                blocked_task = self.tasks.get(blocked_id)
                if blocked_task and blocked_task.status == TaskStatus.BLOCKED:
                    all_complete = all(
                        self.tasks.get(dep_id) and self.tasks[dep_id].status == TaskStatus.COMPLETED
                        for dep_id in blocked_task.depends_on
                    )
                    if all_complete:
                        blocked_task.status = TaskStatus.PENDING

        return True

    def get_execution_summary(self) -> dict[str, Any]:
        """获取执行摘要"""
        total = len(self.tasks)
        completed = sum(1 for t in self.tasks.values() if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in self.tasks.values() if t.status == TaskStatus.FAILED)
        running = sum(1 for t in self.tasks.values() if t.status == TaskStatus.RUNNING)
        pending = sum(1 for t in self.tasks.values() if t.status == TaskStatus.PENDING)
        blocked = sum(1 for t in self.tasks.values() if t.status == TaskStatus.BLOCKED)

        return {
            "total": total,
            "completed": completed,
            "failed": failed,
            "running": running,
            "pending": pending,
            "blocked": blocked,
            "progress_pct": (completed / total * 100) if total > 0 else 0,
        }

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {task_id: task.model_dump() for task_id, task in self.tasks.items()}


class TaskAgent(ToolUseAgent):
    """
    任务管理Agent - 基于Plan-and-Solve和工具循环架构

    架构位置:
        继承 ToolUseAgent → 复用对话循环
        组合 TaskGraph → 管理任务依赖
        组合 TaskToolkit → 提供任务管理工具

    核心组件:
        1. TaskGraph (任务图)
           - 存储所有Task对象
           - 管理任务间的依赖关系
           - 提供拓扑排序确定执行顺序

        2. TaskToolkit (工具集)
           - create_task: 创建任务
           - decompose_task: 分解任务
           - update_task: 更新任务
           - complete_task: 标记完成
           - fail_task: 标记失败
           - list_tasks: 列出任务
           - get_task_details: 获取详情
           - get_next_task: 获取下一个可执行任务

        3. MessageBuilder (来自 core/utils.py)
           - 构建包含任务图状态的系统提示
           - 格式化任务上下文信息

    共享组件:
        - ToolUseAgent: 提供 invoke/stream 接口
        - ToolUseAgent.run_conversation(): 执行对话循环
        - MessageBuilder: 构建消息

    任务生命周期:
        1. 规划: LLM分析目标 → create_task/decompose_task
        2. 执行: get_next_task() → LLM执行 → 工具调用
        3. 完成: complete_task() → 更新依赖 → 解锁阻塞任务

    工作流程:
        run(goal):
            1. 初始化: 创建任务图，清空历史
            2. 规划: LLM分解目标为任务
            3. 执行循环:
               while 有未完成任务:
                   next_task = get_next_task()
                   if next_task:
                       执行(next_task)
                       complete_task(next_task)
                   else:
                       等待依赖完成
            4. 报告: 生成执行摘要

    关键方法:
        run(): 主入口，执行完整任务管理流程
        _build_system_prompt(): 构建包含任务状态的系统提示
        _generate_report(): 生成执行报告

    设计特点:
        - LLM驱动: 所有决策通过LLM，不硬编码逻辑
        - 工具透明: 任务管理通过工具暴露，可扩展
        - 状态追踪: Task对象记录完整执行历史
        - 依赖管理: TaskGraph自动处理任务依赖

    扩展方式:
        1. 添加自定义工具: 在TaskToolkit添加方法
        2. 自定义报告: 重写 _generate_report()
        3. 集成外部系统: 在工具中调用外部API
    """

    SYSTEM_PROMPT = """你是 TaskAgent，一个智能任务管理助手。

你的职责：
1. 帮助用户分解复杂目标为可管理的任务
2. 跟踪任务执行进度
3. 管理任务依赖关系
4. 确保任务按正确顺序完成

工作流程：
1. 分析用户目标
2. 使用 plan_decomposition 工具分解为任务列表（如需自动分解）
3. 或直接使用 create_task 创建具体任务
4. 使用 execute_task 执行任务
5. 使用 complete_task 标记完成
6. 使用 list_tasks 跟踪进度

你可以：
- 自动分解复杂目标
- 手动创建和管理任务
- 设置任务优先级和依赖
- 监控执行进度

任务完成后，总结执行结果。"""

    def __init__(
        self,
        name: str,
        description: str = "",
        model: Model | None = None,
        tools: list[Tool] | None = None,
        max_steps: int = 50,
    ):
        super().__init__(name, description, model, tools, max_steps)
        self.task_graph = TaskGraph()
        self.current_goal: str | None = None
        self._setup_task_tools()

    def _setup_task_tools(self) -> None:
        """设置任务管理工具"""
        task_tools = [
            Tool(
                name="plan_decomposition",
                description="自动分解目标为任务列表。输入：goal (目标描述), context (可选上下文)",
                func=self._tool_plan_decomposition,
                parameters={
                    "type": "object",
                    "properties": {
                        "goal": {"type": "string", "description": "需要分解的目标"},
                        "context": {"type": "object", "description": "可选上下文信息"},
                    },
                    "required": ["goal"],
                },
            ),
            Tool(
                name="create_task",
                description="创建新任务。输入：name, description, requirements, priority",
                func=self._tool_create_task,
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "任务名称"},
                        "description": {"type": "string", "description": "任务描述"},
                        "requirements": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "完成标准",
                        },
                        "priority": {
                            "type": "string",
                            "enum": ["low", "medium", "high", "critical"],
                            "default": "medium",
                        },
                        "depends_on": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "依赖的任务ID",
                        },
                        "parent_id": {"type": "string", "description": "父任务ID"},
                    },
                    "required": ["name", "description"],
                },
            ),
            Tool(
                name="update_task",
                description="更新任务。可更新：name, description, status, priority, requirements",
                func=self._tool_update_task,
                parameters={
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string", "description": "任务ID"},
                        "name": {"type": "string", "description": "新名称"},
                        "description": {"type": "string", "description": "新描述"},
                        "status": {
                            "type": "string",
                            "enum": ["pending", "running", "completed", "failed", "blocked"],
                        },
                        "priority": {
                            "type": "string",
                            "enum": ["low", "medium", "high", "critical"],
                        },
                        "requirements": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["task_id"],
                },
            ),
            Tool(
                name="delete_task",
                description="删除任务及其依赖关系",
                func=self._tool_delete_task,
                parameters={
                    "type": "object",
                    "properties": {"task_id": {"type": "string", "description": "要删除的任务ID"}},
                    "required": ["task_id"],
                },
            ),
            Tool(
                name="list_tasks",
                description="列出任务。可按状态筛选",
                func=self._tool_list_tasks,
                parameters={
                    "type": "object",
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": ["pending", "running", "completed", "failed", "blocked", "all"],
                            "default": "all",
                        },
                        "parent_id": {"type": "string", "description": "父任务ID（可选）"},
                    },
                },
            ),
            Tool(
                name="get_task",
                description="获取任务详情",
                func=self._tool_get_task,
                parameters={
                    "type": "object",
                    "properties": {"task_id": {"type": "string", "description": "任务ID"}},
                    "required": ["task_id"],
                },
            ),
            Tool(
                name="execute_task",
                description="执行具体任务。这会启动工具循环来完成任务",
                func=self._tool_execute_task,
                parameters={
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string", "description": "要执行的任务ID"},
                        "strategy": {
                            "type": "string",
                            "enum": ["auto", "step_by_step"],
                            "default": "auto",
                            "description": "执行策略",
                        },
                    },
                    "required": ["task_id"],
                },
            ),
            Tool(
                name="complete_task",
                description="标记任务为完成，记录结果",
                func=self._tool_complete_task,
                parameters={
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string", "description": "任务ID"},
                        "result": {"type": "string", "description": "执行结果"},
                    },
                    "required": ["task_id"],
                },
            ),
            Tool(
                name="fail_task",
                description="标记任务为失败，记录错误",
                func=self._tool_fail_task,
                parameters={
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string", "description": "任务ID"},
                        "error": {"type": "string", "description": "错误信息"},
                    },
                    "required": ["task_id", "error"],
                },
            ),
            Tool(
                name="get_task_summary",
                description="获取任务执行摘要和统计",
                func=self._tool_get_summary,
                parameters={"type": "object", "properties": {}},
            ),
            Tool(
                name="visualize_task_graph",
                description="可视化任务图结构",
                func=self._tool_visualize,
                parameters={"type": "object", "properties": {}},
            ),
        ]

        # 添加任务工具到工具列表
        if self.tools is None:
            self.tools = []
        self.tools.extend(task_tools)

    # ========== 工具实现 ==========

    def _tool_plan_decomposition(self, query: str) -> str:
        """工具：自动分解目标"""
        try:
            args = json.loads(query)
            goal = args.get("goal", "")
            context = args.get("context", {})

            if not self.model:
                return "Error: 未配置模型"

            # 构建分解提示
            prompt = f"""将以下目标分解为具体任务：

目标：{goal}

请返回 JSON 格式的任务列表：
{{
    "tasks": [
        {{
            "name": "任务名称",
            "description": "详细描述",
            "requirements": ["完成标准1", "完成标准2"],
            "priority": "medium",
            "depends_on": []
        }}
    ]
}}

分解原则：
- 每个任务应具体、可执行
- 设置合理的依赖关系
- 明确完成标准"""

            messages = [
                {"role": "system", "content": "你是任务规划专家。返回有效的 JSON。"},
                {"role": "user", "content": prompt},
            ]

            response = self.model.generate(messages, temperature=0.3)
            content = response.choices[0].message.content

            # 提取 JSON
            json_match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
            if not json_match:
                json_match = re.search(r"(\{.*\})", content, re.DOTALL)

            if json_match:
                plan = json.loads(json_match.group(1))
                tasks = plan.get("tasks", [])

                created_ids = []
                name_to_id = {}

                # 第一遍：创建所有任务
                for task_def in tasks:
                    task = Task(
                        task_id=self._generate_task_id(),
                        name=task_def["name"],
                        description=task_def.get("description", ""),
                        requirements=task_def.get("requirements", []),
                        priority=TaskPriority[task_def.get("priority", "medium").upper()],
                        context=context,
                    )
                    self.task_graph.add_task(task)
                    created_ids.append(task.task_id)
                    name_to_id[task.name] = task.task_id

                # 第二遍：设置依赖
                for task_def in tasks:
                    task_name = task_def["name"]
                    if task_name in name_to_id:
                        task = self.task_graph.get_task(name_to_id[task_name])
                        if not task:
                            continue
                        for dep_name in task_def.get("depends_on", []):
                            if dep_name in name_to_id:
                                task.depends_on.append(name_to_id[dep_name])
                                dep_task = self.task_graph.get_task(name_to_id[dep_name])
                                if dep_task:
                                    dep_task.blocks.append(task.task_id)
                                    task.status = TaskStatus.BLOCKED

                return f"✅ 已分解为 {len(created_ids)} 个任务：{', '.join(created_ids)}"
            else:
                return "Error: 无法解析分解结果"

        except Exception as e:
            return f"Error: 分解失败 - {e}"

    def _tool_create_task(self, query: str) -> str:
        """工具：创建任务"""
        try:
            args = json.loads(query)
            task = Task(
                task_id=self._generate_task_id(),
                name=args["name"],
                description=args.get("description", ""),
                requirements=args.get("requirements", []),
                priority=TaskPriority[args.get("priority", "medium").upper()],
                parent_id=args.get("parent_id"),
            )

            # 设置依赖
            for dep_id in args.get("depends_on", []):
                if dep_id in self.task_graph.tasks:
                    task.depends_on.append(dep_id)
                    dep_task = self.task_graph.get_task(dep_id)
                    if dep_task:
                        dep_task.blocks.append(task.task_id)
                    task.status = TaskStatus.BLOCKED

            # 添加到父任务
            if task.parent_id and task.parent_id in self.task_graph.tasks:
                parent = self.task_graph.get_task(task.parent_id)
                if parent:
                    parent.subtasks.append(task.task_id)

            self.task_graph.add_task(task)
            return f"✅ 创建任务 '{task.name}' (ID: {task.task_id})"

        except Exception as e:
            return f"Error: 创建失败 - {e}"

    def _tool_update_task(self, query: str) -> str:
        """工具：更新任务"""
        try:
            args = json.loads(query)
            task_id = args["task_id"]
            task = self.task_graph.get_task(task_id)

            if not task:
                return f"Error: 任务 {task_id} 不存在"

            if "name" in args:
                task.name = args["name"]
            if "description" in args:
                task.description = args["description"]
            if "status" in args:
                new_status = TaskStatus(args["status"])
                self.task_graph.update_task_status(task_id, new_status)
            if "priority" in args:
                task.priority = TaskPriority[args["priority"].upper()]
            if "requirements" in args:
                task.requirements = args["requirements"]

            task.update_time = datetime.now()
            return f"✅ 更新任务 '{task.name}'"

        except Exception as e:
            return f"Error: 更新失败 - {e}"

    def _tool_delete_task(self, query: str) -> str:
        """工具：删除任务"""
        try:
            args = json.loads(query)
            task_id = args["task_id"]
            task = self.task_graph.get_task(task_id)

            if not task:
                return f"Error: 任务 {task_id} 不存在"

            name = task.name
            self.task_graph.remove_task(task_id)
            return f"✅ 删除任务 '{name}'"

        except Exception as e:
            return f"Error: 删除失败 - {e}"

    def _tool_list_tasks(self, query: str) -> str:
        """工具：列出任务"""
        try:
            args = json.loads(query) if query else {}
            status_filter = args.get("status", "all")
            parent_filter = args.get("parent_id")

            filtered = []
            for task in self.task_graph.tasks.values():
                if status_filter != "all" and task.status != status_filter:
                    continue
                if parent_filter is not None and task.parent_id != parent_filter:
                    continue
                filtered.append(task)

            if not filtered:
                return "暂无任务"

            lines = [f"📋 共 {len(filtered)} 个任务："]
            for task in sorted(filtered, key=lambda t: t.create_time):
                icon = {
                    TaskStatus.PENDING: "⏳",
                    TaskStatus.RUNNING: "▶️",
                    TaskStatus.COMPLETED: "✅",
                    TaskStatus.FAILED: "❌",
                    TaskStatus.BLOCKED: "🚫",
                }.get(task.status, "❓")
                parent_info = f" [父:{task.parent_id}]" if task.parent_id else ""
                lines.append(f"  {icon} {task.name} [{task.task_id}]{parent_info}")

            return "\n".join(lines)

        except Exception as e:
            return f"Error: 查询失败 - {e}"

    def _tool_get_task(self, query: str) -> str:
        """工具：获取任务详情"""
        try:
            args = json.loads(query)
            task_id = args["task_id"]
            task = self.task_graph.get_task(task_id)

            if not task:
                return f"Error: 任务 {task_id} 不存在"

            lines = [
                f"📄 任务详情: {task.name}",
                f"  ID: {task.task_id}",
                f"  描述: {task.description}",
                f"  状态: {task.status}",
                f"  优先级: {task.priority.name}",
            ]

            if task.requirements:
                lines.append(f"  完成标准: {', '.join(task.requirements)}")
            if task.depends_on:
                lines.append(f"  依赖: {', '.join(task.depends_on)}")
            if task.subtasks:
                lines.append(f"  子任务: {', '.join(task.subtasks)}")
            if task.result:
                lines.append(f"  结果: {task.result}")
            if task.error:
                lines.append(f"  错误: {task.error}")

            return "\n".join(lines)

        except Exception as e:
            return f"Error: 查询失败 - {e}"

    def _tool_execute_task(self, query: str) -> str:
        """工具：执行任务 - 使用工具循环"""
        try:
            args = json.loads(query)
            task_id = args["task_id"]
            task = self.task_graph.get_task(task_id)

            if not task:
                return f"Error: 任务 {task_id} 不存在"

            if task.status == TaskStatus.COMPLETED:
                return f"任务 '{task.name}' 已完成"

            # 检查依赖
            for dep_id in task.depends_on:
                dep_task = self.task_graph.get_task(dep_id)
                if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                    return f"Error: 依赖任务 {dep_id} 未完成"

            # 更新状态为运行中
            self.task_graph.update_task_status(task_id, TaskStatus.RUNNING)

            # 构建任务执行提示
            task_prompt = f"""执行任务：{task.name}

描述：{task.description}
完成标准：{", ".join(task.requirements) if task.requirements else "无"}

请使用可用工具来完成此任务。完成后调用 complete_task 工具标记完成。
如果无法完成，调用 fail_task 工具标记失败。

直接开始执行任务，不要询问。"""

            # 保存当前对话状态
            original_history = self.message_history.copy()

            # 添加任务执行上下文（使用 MessageBuilder）
            self.message_history.append(MessageBuilder.build_user_message(task_prompt))

            # 使用工具循环执行任务
            tool_results = []
            max_tool_calls = 10

            for _ in range(max_tool_calls):
                # 获取工具定义
                openai_tools = self._get_openai_tools()

                # 生成响应
                response = self.model.generate(self.message_history, tools=openai_tools)
                message = response.choices[0].message

                # 使用 MessageBuilder 构建助手消息
                tool_calls = MessageBuilder.convert_api_tool_calls(message.tool_calls)
                assistant_msg = MessageBuilder.build_assistant_message(
                    message.content or "", tool_calls
                )
                self.message_history.append(assistant_msg)

                # 如果没有工具调用，说明任务已完成或需要确认
                if not tool_calls:
                    result = message.content or "任务执行完成"
                    task.tool_calls.append({"final_response": result})
                    self.message_history = original_history  # 恢复对话
                    return f"✅ 任务 '{task.name}' 执行完成：{result[:200]}..."

                # 执行工具调用
                for tool_call in tool_calls:
                    tool_name = tool_call["function"]["name"]
                    tool_args = tool_call["function"]["arguments"]

                    # 记录工具调用
                    task.tool_calls.append(
                        {
                            "tool": tool_name,
                            "args": tool_args,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )

                    # 如果是 complete_task 或 fail_task，特殊处理
                    if tool_name == "complete_task":
                        try:
                            complete_args = json.loads(tool_args)
                            if complete_args.get("task_id") == task_id:
                                result = complete_args.get("result", "任务完成")
                                self._tool_complete_task(tool_args)
                                self.message_history = original_history
                                return f"✅ 任务 '{task.name}' 已完成：{result}"
                        except Exception:
                            pass
                    elif tool_name == "fail_task":
                        try:
                            fail_args = json.loads(tool_args)
                            if fail_args.get("task_id") == task_id:
                                error = fail_args.get("error", "未知错误")
                                self._tool_fail_task(tool_args)
                                self.message_history = original_history
                                return f"❌ 任务 '{task.name}' 失败：{error}"
                        except Exception:
                            pass

                    # 执行普通工具
                    tool_result = self.call_tool(tool_name, tool_args)
                    tool_results.append({"tool": tool_name, "result": tool_result})

                    # 使用 MessageBuilder 构建工具响应
                    tool_msg = MessageBuilder.build_tool_response_message(
                        tool_call["id"], str(tool_result)
                    )
                    self.message_history.append(tool_msg)

            # 达到最大工具调用次数
            self.message_history = original_history
            return f"⚠️ 任务 '{task.name}' 执行中达到最大步数"

        except Exception as e:
            return f"Error: 执行失败 - {e}"

    def _tool_complete_task(self, query: str) -> str:
        """工具：标记任务完成"""
        try:
            args = json.loads(query)
            task_id = args["task_id"]
            result = args.get("result", "任务完成")

            success = self.task_graph.update_task_status(
                task_id, TaskStatus.COMPLETED, result=result
            )

            if success:
                task = self.task_graph.get_task(task_id)
                if task:
                    return f"✅ 任务 '{task.name}' 标记为完成"
                return f"✅ 任务 {task_id} 标记为完成"
            else:
                return f"Error: 任务 {task_id} 不存在"

        except Exception as e:
            return f"Error: {e}"

    def _tool_fail_task(self, query: str) -> str:
        """工具：标记任务失败"""
        try:
            args = json.loads(query)
            task_id = args["task_id"]
            error = args.get("error", "未知错误")

            success = self.task_graph.update_task_status(task_id, TaskStatus.FAILED, error=error)

            if success:
                task = self.task_graph.get_task(task_id)
                if task:
                    return f"❌ 任务 '{task.name}' 标记为失败"
                return f"❌ 任务 {task_id} 标记为失败"
            else:
                return f"Error: 任务 {task_id} 不存在"

        except Exception as e:
            return f"Error: {e}"

    def _tool_get_summary(self, query: str) -> str:
        """工具：获取执行摘要"""
        summary = self.task_graph.get_execution_summary()
        return json.dumps(summary, indent=2, ensure_ascii=False)

    def _tool_visualize(self, query: str) -> str:
        """工具：可视化任务图"""
        lines = ["📊 任务图结构:", "=" * 50]

        root_tasks = [t for t in self.task_graph.tasks.values() if t.parent_id is None]

        def print_task_tree(task: Task, indent: int = 0):
            prefix = "  " * indent
            icon = {
                TaskStatus.PENDING: "⏳",
                TaskStatus.RUNNING: "▶️",
                TaskStatus.COMPLETED: "✅",
                TaskStatus.FAILED: "❌",
                TaskStatus.BLOCKED: "🚫",
            }.get(task.status, "❓")

            lines.append(f"{prefix}{icon} {task.name} [{task.task_id}]")

            for subtask_id in task.subtasks:
                subtask = self.task_graph.get_task(subtask_id)
                if subtask:
                    print_task_tree(subtask, indent + 1)

        for task in sorted(root_tasks, key=lambda t: t.create_time):
            print_task_tree(task)

        return "\n".join(lines)

    # ========== 辅助方法 ==========

    def _generate_task_id(self) -> str:
        """生成任务ID"""
        return f"task_{uuid.uuid4().hex[:8]}"

    # ========== 重写父类方法 ==========

    def run(self, goal: str) -> dict[str, Any]:
        """
        主入口 - 执行目标

        流程：
        1. LLM 分析目标并决定如何分解
        2. 使用工具创建任务
        3. 使用工具执行任务
        4. 跟踪进度直到完成
        """
        if not self.model:
            return {"success": False, "error": "未配置模型"}

        self.current_goal = goal

        # 初始化对话（使用 MessageBuilder）
        self.message_history = [
            MessageBuilder.build_system_message(self.SYSTEM_PROMPT),
            MessageBuilder.build_user_message(
                f"目标：{goal}\n\n请分析此目标，决定如何分解和执行。"
            ),
        ]

        print(f"\n🎯 目标: {goal}")
        print("=" * 60)

        # 使用工具循环处理
        step = 0
        while step < self.max_steps:
            step += 1

            # 获取可用工具
            openai_tools = self._get_openai_tools()

            # 生成响应
            response = self.model.generate(self.message_history, tools=openai_tools)
            message = response.choices[0].message

            # 使用 MessageBuilder 构建助手消息
            tool_calls = MessageBuilder.convert_api_tool_calls(message.tool_calls)
            assistant_msg = MessageBuilder.build_assistant_message(
                message.content or "", tool_calls
            )
            self.message_history.append(assistant_msg)

            # 打印助手思考
            if message.content:
                print(f"\n🤖 {message.content}")

            # 如果没有工具调用，检查是否完成
            if not tool_calls:
                # 检查是否所有任务都完成了
                summary = self.task_graph.get_execution_summary()
                if summary["total"] > 0 and summary["pending"] == 0 and summary["running"] == 0:
                    print("\n✅ 所有任务完成！")
                    return {
                        "success": summary["failed"] == 0,
                        "summary": summary,
                        "tasks": self.task_graph.to_dict(),
                    }
                continue

            # 执行工具调用
            print("\n🔧 工具调用:")
            for tool_call in tool_calls:
                tool_name = tool_call["function"]["name"]
                tool_args = tool_call["function"]["arguments"]

                print(f"  - {tool_name}: {tool_args[:100]}...")

                # 执行工具
                tool_result = self.call_tool(tool_name, tool_args)

                # 打印结果摘要
                result_preview = (
                    tool_result[:150] + "..." if len(tool_result) > 150 else tool_result
                )
                print(f"    → {result_preview}")

                # 使用 MessageBuilder 构建工具响应
                tool_msg = MessageBuilder.build_tool_response_message(
                    tool_call["id"], str(tool_result)
                )
                self.message_history.append(tool_msg)

        # 达到最大步数
        summary = self.task_graph.get_execution_summary()
        return {
            "success": summary["failed"] == 0 and summary["pending"] == 0,
            "summary": summary,
            "tasks": self.task_graph.to_dict(),
            "error": "达到最大步数",
        }

    def stream(self, input: str, reset: bool = True) -> str:
        """流式接口 - 直接代理到 run"""
        result = self.run(input)
        if result["success"]:
            summary = result["summary"]
            return f"\n✅ 执行完成！{summary['completed']}/{summary['total']} 任务完成"
        else:
            return f"\n❌ 执行失败: {result.get('error', '未知错误')}"


# ========== 示例工具 ==========


def search_tool(query: str) -> str:
    """搜索工具"""
    try:
        args = json.loads(query)
        q = args.get("query", "")
        return f"[搜索结果] 找到关于 '{q}' 的 5 个相关文档"
    except Exception:
        return "[搜索结果] 找到相关信息"


def calculator_tool(query: str) -> str:
    """计算器工具"""
    try:
        args = json.loads(query)
        expr = args.get("expression", "")
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expr):
            return "Error: 非法字符"
        result = eval(expr, {"__builtins__": {}}, {})
        return f"结果: {result}"
    except Exception as e:
        return f"Error: {e}"


def write_file_tool(query: str) -> str:
    """写文件工具"""
    try:
        args = json.loads(query)
        return f"✅ 已写入 {args.get('path')}"
    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    from dotenv import load_dotenv

    print("TaskAgent - 基于工具循环的 Plan-and-Solve 架构")
    print("=" * 60)
    print("\n新架构特点：")
    print("1. 所有任务操作都是工具（create_task, execute_task 等）")
    print("2. LLM 通过工具调用来管理任务流")
    print("3. 任务执行使用 ToolUseAgent 的工具循环")
    print("4. 支持自动分解和手动管理")
    print("\n示例用法：")
    print("  agent = TaskAgent(name='助手', model=Model(), tools=[...])")
    print("  result = agent.run('完成数据分析项目')")
    load_dotenv()
    agent = TaskAgent(name="Task Agent", model=Model())
    result = agent.run("设计并实现一个数据分析agent")
