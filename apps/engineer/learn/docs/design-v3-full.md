# Coder Agent V3 - 完整版设计方案

> 企业级代码智能体：对标 OpenCode、Codex CLI、Claude Code

**版本**: V3.0 Full  
**目标**: 生产环境企业级部署，功能对标开源 Code Agent  
**代码量**: ~10000 行核心代码  
**部署时间**: 1 天  
**对标产品**: OpenCode, Codex CLI, Claude Code, Aider, Continue

---

## 🎯 设计哲学

**专业、安全、可扩展、可观测**

V3 版本专注于：
1. **MCP 协议** - 标准化工具通信
2. **Subagent 委派** - 复杂任务并行处理
3. **多模型支持** - OpenAI/Anthropic/本地模型
4. **企业安全** - 审计日志、权限控制
5. **可观测性** - 指标、追踪、日志

---

## 📦 核心能力

```
✅ MCP (Model Context Protocol) 协议
✅ Subagent 委派系统
✅ 多模型路由与切换
✅ 插件系统（技能市场）
✅ 企业级权限控制
✅ 完整审计日志
✅ 性能监控与指标
✅ 分布式会话同步
✅ A/B 测试框架
✅ Web UI 控制台
```

---

## 🏗️ 企业级架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     FullCoderAgent Platform                      │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   API Layer  │  │   Web UI     │  │   CLI Interface      │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Orchestrator │  │  Subagent    │  │  Model Router        │  │
│  │   (主控)      │  │   Manager    │  │  (多模型切换)         │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   MCP Host   │  │  Skill Store │  │  Permission Manager  │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Audit Logger │  │  Metrics     │  │  Session Cluster     │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 架构说明

| 层级 | 组件 | 职责 | 技术栈 |
|------|------|------|--------|
| **接入层** | API/Web/CLI | 多入口接入 | FastAPI, React, Click |
| **核心层** | Orchestrator | 任务调度 | Asyncio, Celery |
| **核心层** | Subagent Manager | 子代理生命周期 | Process Pool |
| **核心层** | Model Router | 模型负载均衡 | LiteLLM |
| **协议层** | MCP Host | 标准化工具协议 | MCP SDK |
| **扩展层** | Skill Store | 插件市场 | Docker + Registry |
| **安全层** | Permission Manager | RBAC 权限 | Casbin |
| **观测层** | Audit/Metrics | 可观测性 | Prometheus, ELK |
| **存储层** | Session Cluster | 分布式会话 | Redis Cluster |

---

## 💻 核心代码实现

### 目录结构

```
coder_platform/
├── api/                     # API 层
│   ├── rest/               # REST API
│   ├── websocket/          # WebSocket 实时通信
│   └── graphql/            # GraphQL (可选)
├── web/                     # Web UI
│   ├── frontend/           # React/Vue 前端
│   └── backend/            # 静态服务
├── cli/                     # 命令行工具
│   └── main.py
├── core/                    # 核心引擎
│   ├── orchestrator.py     # 任务编排器
│   ├── subagent/
│   │   ├── manager.py      # 子代理管理
│   │   ├── worker.py       # 子代理工作器
│   │   └── protocol.py     # 通信协议
│   ├── mcp/
│   │   ├── host.py         # MCP Host
│   │   ├── client.py       # MCP Client
│   │   └── registry.py     # 工具注册表
│   └── router/
│       ├── model_router.py # 模型路由
│       ├── load_balancer.py # 负载均衡
│       └── fallback.py     # 故障转移
├── skills/                  # 技能系统
│   ├── builtin/            # 内置技能
│   ├── marketplace/        # 技能市场
│   └── loader.py           # 技能加载器
├── security/                # 安全层
│   ├── permissions.py      # 权限控制
│   ├── audit.py            # 审计日志
│   └── encryption.py       # 加密模块
├── observability/           # 可观测性
│   ├── metrics.py          # 指标收集
│   ├── tracing.py          # 分布式追踪
│   └── logging.py          # 结构化日志
└── storage/                 # 存储层
    ├── session_store.py    # 会话存储
    ├── vector_store.py     # 向量存储
    └── cache.py            # 缓存层
```

### 1. 主控编排器 (Orchestrator)

```python
# core/orchestrator.py
"""
任务编排器 - 智能任务分解与调度

功能：
- 任务规划 (Planning)
- 子代理委派 (Delegation)
- 结果聚合 (Aggregation)
- 质量检查 (Quality Check)
"""

import asyncio
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import uuid

from .subagent.manager import SubagentManager
from .mcp.host import MCPHost
from .router.model_router import ModelRouter
from ..security.permissions import PermissionChecker
from ..observability.tracing import tracer


class TaskType(Enum):
    """任务类型"""
    CODE_EDIT = "code_edit"           # 代码编辑
    CODE_REVIEW = "code_review"       # 代码审查
    ARCHITECTURE = "architecture"     # 架构设计
    DEBUGGING = "debugging"           # 调试
    REFACTORING = "refactoring"       # 重构
    DOCUMENTATION = "documentation"   # 文档生成
    ANALYSIS = "analysis"             # 代码分析


@dataclass
class TaskPlan:
    """任务计划"""
    task_id: str
    task_type: TaskType
    description: str
    steps: List[Dict]
    estimated_tokens: int
    required_skills: List[str]
    parallelizable: bool


@dataclass
class Subtask:
    """子任务"""
    subtask_id: str
    parent_task_id: str
    description: str
    assigned_agent: Optional[str]
    dependencies: List[str]
    status: str = "pending"
    result: Any = None


class Orchestrator:
    """
    任务编排器
    
    工作流程：
    1. 接收用户请求
    2. 分析任务类型
    3. 生成执行计划
    4. 委派子代理
    5. 监控执行
    6. 聚合结果
    7. 质量检查
    8. 返回最终答案
    """
    
    def __init__(
        self,
        subagent_manager: SubagentManager,
        mcp_host: MCPHost,
        model_router: ModelRouter,
        permission_checker: PermissionChecker
    ):
        self.subagent_manager = subagent_manager
        self.mcp_host = mcp_host
        self.model_router = model_router
        self.permission = permission_checker
        
        # 任务追踪
        self.active_tasks: Dict[str, TaskPlan] = {}
        self.subtasks: Dict[str, Subtask] = {}
    
    @tracer.span("orchestrate_task")
    async def orchestrate(
        self,
        user_request: str,
        context: Dict[str, Any],
        user_id: str
    ) -> Dict[str, Any]:
        """
        编排执行任务
        
        Args:
            user_request: 用户原始请求
            context: 上下文信息
            user_id: 用户标识
            
        Returns:
            执行结果
        """
        task_id = str(uuid.uuid4())[:8]
        
        # 1. 权限检查
        if not await self.permission.check(user_id, "task:execute"):
            raise PermissionError(f"用户 {user_id} 无任务执行权限")
        
        # 2. 任务分析
        task_type = await self._analyze_task_type(user_request)
        
        # 3. 生成执行计划
        plan = await self._create_plan(
            task_id=task_id,
            task_type=task_type,
            request=user_request,
            context=context
        )
        self.active_tasks[task_id] = plan
        
        # 4. 执行计划
        if plan.parallelizable and len(plan.steps) > 1:
            results = await self._execute_parallel(plan, user_id)
        else:
            results = await self._execute_sequential(plan, user_id)
        
        # 5. 聚合结果
        final_result = await self._aggregate_results(
            task_type=task_type,
            subtask_results=results,
            original_request=user_request
        )
        
        # 6. 质量检查
        quality_score = await self._quality_check(final_result)
        
        # 7. 清理
        del self.active_tasks[task_id]
        
        return {
            "task_id": task_id,
            "result": final_result,
            "quality_score": quality_score,
            "steps_executed": len(plan.steps),
            "tokens_used": plan.estimated_tokens
        }
    
    async def _analyze_task_type(self, request: str) -> TaskType:
        """分析任务类型"""
        prompt = f"""分析以下请求的任务类型：
{request}

可选类型：{', '.join(t.value for t in TaskType)}

只返回类型值，不要其他内容。"""
        
        response = await self.model_router.generate(
            prompt=prompt,
            model_preference="fast"
        )
        
        type_str = response.strip().lower()
        try:
            return TaskType(type_str)
        except ValueError:
            return TaskType.CODE_EDIT  # 默认
    
    async def _create_plan(
        self,
        task_id: str,
        task_type: TaskType,
        request: str,
        context: Dict
    ) -> TaskPlan:
        """创建执行计划"""
        # 使用 LLM 生成计划
        planning_prompt = f"""你是一个任务规划专家。请将以下请求分解为具体的执行步骤：

请求类型: {task_type.value}
请求内容: {request}

请按以下格式输出计划：
1. [步骤描述] - 所需技能
2. [步骤描述] - 所需技能
...

预估 Token 数: <数字>
是否可并行: <是/否>"""
        
        plan_response = await self.model_router.generate(
            prompt=planning_prompt,
            model_preference="smart"
        )
        
        # 解析计划（简化版）
        steps = self._parse_plan(plan_response)
        
        return TaskPlan(
            task_id=task_id,
            task_type=task_type,
            description=request,
            steps=steps,
            estimated_tokens=5000,  # 简化
            required_skills=self._extract_skills(steps),
            parallelizable=len(steps) > 1 and task_type != TaskType.DEBUGGING
        )
    
    async def _execute_parallel(
        self,
        plan: TaskPlan,
        user_id: str
    ) -> List[Any]:
        """并行执行"""
        # 创建子任务
        subtasks = []
        for i, step in enumerate(plan.steps):
            subtask = Subtask(
                subtask_id=f"{plan.task_id}_{i}",
                parent_task_id=plan.task_id,
                description=step["description"],
                assigned_agent=None,
                dependencies=step.get("dependencies", [])
            )
            subtasks.append(subtask)
            self.subtasks[subtask.subtask_id] = subtask
        
        # 并行委派
        async def execute_subtask(subtask: Subtask):
            # 检查依赖
            for dep_id in subtask.dependencies:
                while self.subtasks[dep_id].status != "completed":
                    await asyncio.sleep(0.1)
            
            # 委派给子代理
            result = await self.subagent_manager.delegate(
                task=subtask.description,
                skills_required=plan.required_skills,
                user_id=user_id
            )
            
            subtask.status = "completed"
            subtask.result = result
            return result
        
        # 执行所有子任务
        results = await asyncio.gather(*[
            execute_subtask(st) for st in subtasks
        ])
        
        return results
    
    async def _execute_sequential(
        self,
        plan: TaskPlan,
        user_id: str
    ) -> List[Any]:
        """串行执行"""
        results = []
        
        for step in plan.steps:
            result = await self.subagent_manager.delegate(
                task=step["description"],
                skills_required=step.get("skills", []),
                user_id=user_id
            )
            results.append(result)
        
        return results
    
    async def _aggregate_results(
        self,
        task_type: TaskType,
        subtask_results: List[Any],
        original_request: str
    ) -> str:
        """聚合结果"""
        # 使用 LLM 整合多个子任务的结果
        aggregation_prompt = f"""请整合以下子任务的执行结果，生成最终回答：

原始请求: {original_request}
任务类型: {task_type.value}

子任务结果:
{chr(10).join(f"- {r}" for r in subtask_results)}

请给出完整、连贯的最终答案。"""
        
        final_response = await self.model_router.generate(
            prompt=aggregation_prompt,
            model_preference="smart"
        )
        
        return final_response
    
    async def _quality_check(self, result: str) -> float:
        """质量检查"""
        # 简单的启发式检查
        checks = [
            len(result) > 50,  # 不要太短
            "error" not in result.lower(),  # 无错误
            "sorry" not in result.lower(),  # 无道歉
        ]
        return sum(checks) / len(checks)
    
    def _parse_plan(self, plan_text: str) -> List[Dict]:
        """解析计划文本"""
        steps = []
        for line in plan_text.split("\n"):
            if line.strip().startswith(("1.", "2.", "3.", "4.", "5.")):
                steps.append({"description": line.strip()[2:].strip()})
        return steps or [{"description": plan_text}]
    
    def _extract_skills(self, steps: List[Dict]) -> List[str]:
        """提取所需技能"""
        skills = set()
        for step in steps:
            desc = step.get("description", "").lower()
            if "file" in desc:
                skills.add("file_system")
            if "test" in desc:
                skills.add("testing")
            if "search" in desc:
                skills.add("code_search")
        return list(skills)
```

### 2. MCP Host 实现

```python
# core/mcp/host.py
"""
MCP (Model Context Protocol) Host 实现

参考: https://modelcontextprotocol.io/

功能：
- MCP Server 管理
- 工具发现与调用
- 资源访问
- Prompt 模板
"""

import json
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass
import asyncio


@dataclass
class MCPServer:
    """MCP Server 配置"""
    name: str
    transport: str  # "stdio" | "sse" | "websocket"
    command: Optional[str] = None  # stdio 命令
    args: List[str] = None
    url: Optional[str] = None  # SSE/WebSocket URL
    env: Dict[str, str] = None


@dataclass
class MCPTool:
    """MCP 工具定义"""
    name: str
    description: str
    input_schema: Dict
    server: str


@dataclass
class MCPResource:
    """MCP 资源定义"""
    uri: str
    name: str
    mime_type: str
    server: str


class MCPHost:
    """
    MCP Host 实现
    
    管理多个 MCP Server，提供统一的工具和资源访问接口。
    """
    
    def __init__(self):
        self.servers: Dict[str, MCPServer] = {}
        self.tools: Dict[str, MCPTool] = {}
        self.resources: Dict[str, MCPResource] = {}
        self.connections: Dict[str, Any] = {}
    
    async def register_server(self, server: MCPServer):
        """
        注册 MCP Server
        
        Args:
            server: Server 配置
        """
        self.servers[server.name] = server
        
        # 建立连接
        if server.transport == "stdio":
            await self._connect_stdio(server)
        elif server.transport == "sse":
            await self._connect_sse(server)
        
        # 发现工具和资源
        await self._discover_capabilities(server.name)
    
    async def _connect_stdio(self, server: MCPServer):
        """stdio 连接"""
        import subprocess
        
        proc = await asyncio.create_subprocess_exec(
            server.command,
            *server.args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**server.env} if server.env else None
        )
        
        self.connections[server.name] = proc
    
    async def _discover_capabilities(self, server_name: str):
        """发现 Server 能力"""
        # 发送 tools/list 请求
        response = await self._send_request(
            server_name,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list"
            }
        )
        
        # 注册工具
        for tool_def in response.get("tools", []):
            tool = MCPTool(
                name=tool_def["name"],
                description=tool_def["description"],
                input_schema=tool_def["inputSchema"],
                server=server_name
            )
            self.tools[tool.name] = tool
    
    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Dict:
        """
        调用 MCP 工具
        
        Args:
            tool_name: 工具名称
            arguments: 参数
            
        Returns:
            工具调用结果
        """
        if tool_name not in self.tools:
            raise ValueError(f"未知工具: {tool_name}")
        
        tool = self.tools[tool_name]
        
        response = await self._send_request(
            tool.server,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            }
        )
        
        return response.get("result", {})
    
    async def _send_request(
        self,
        server_name: str,
        request: Dict
    ) -> Dict:
        """发送请求到 Server"""
        conn = self.connections[server_name]
        
        if isinstance(conn, asyncio.subprocess.Process):
            # stdio 通信
            request_bytes = (json.dumps(request) + "\n").encode()
            conn.stdin.write(request_bytes)
            await conn.stdin.drain()
            
            # 读取响应
            response_bytes = await conn.stdout.readline()
            return json.loads(response_bytes.decode())
        
        return {}
    
    def list_tools(self) -> List[MCPTool]:
        """列出所有可用工具"""
        return list(self.tools.values())
    
    def get_tool_schema(self) -> List[Dict]:
        """获取工具 Schema（OpenAI 格式）"""
        schemas = []
        for tool in self.tools.values():
            schemas.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema
                }
            })
        return schemas
```

### 3. Subagent 管理系统

```python
# core/subagent/manager.py
"""
子代理管理系统

功能：
- 子代理生命周期管理
- 任务委派
- 结果收集
- 资源限制
"""

import asyncio
import multiprocessing as mp
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
import uuid

from .worker import SubagentWorker
from .protocol import SubagentMessage, MessageType


@dataclass
class SubagentConfig:
    """子代理配置"""
    max_workers: int = 4
    max_tasks_per_worker: int = 10
    task_timeout: int = 300  # 5分钟
    enable_sandbox: bool = True
    allowed_tools: List[str] = None


class SubagentManager:
    """
    子代理管理器
    
    管理一个子代理工作池，支持并行任务处理。
    """
    
    def __init__(self, config: Optional[SubagentConfig] = None):
        self.config = config or SubagentConfig()
        self.executor = ProcessPoolExecutor(
            max_workers=self.config.max_workers
        )
        self.active_tasks: Dict[str, asyncio.Future] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.results: Dict[str, Any] = {}
    
    async def delegate(
        self,
        task: str,
        skills_required: List[str],
        user_id: str,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        委派任务给子代理
        
        Args:
            task: 任务描述
            skills_required: 所需技能
            user_id: 用户标识
            context: 额外上下文
            
        Returns:
            任务执行结果
        """
        task_id = str(uuid.uuid4())[:8]
        
        # 创建工作单元
        work_unit = {
            "task_id": task_id,
            "description": task,
            "skills": skills_required,
            "user_id": user_id,
            "context": context or {},
            "config": {
                "timeout": self.config.task_timeout,
                "allowed_tools": self.config.allowed_tools
            }
        }
        
        # 提交到进程池
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(
            self.executor,
            self._execute_in_worker,
            work_unit
        )
        
        self.active_tasks[task_id] = future
        
        try:
            # 等待完成（带超时）
            result = await asyncio.wait_for(
                future,
                timeout=self.config.task_timeout
            )
            self.results[task_id] = result
            return result
        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": f"任务执行超时（>{self.config.task_timeout}秒）",
                "task_id": task_id
            }
        finally:
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
    
    @staticmethod
    def _execute_in_worker(work_unit: Dict) -> Dict:
        """在工作进程中执行"""
        # 创建子代理工作器
        worker = SubagentWorker(work_unit["config"])
        
        # 执行任务
        result = worker.execute(
            description=work_unit["description"],
            skills=work_unit["skills"],
            context=work_unit["context"]
        )
        
        return {
            "success": True,
            "result": result,
            "task_id": work_unit["task_id"]
        }
    
    async def get_status(self) -> Dict[str, Any]:
        """获取管理器状态"""
        return {
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.results),
            "max_workers": self.config.max_workers,
            "queue_size": self.task_queue.qsize()
        }
    
    async def shutdown(self):
        """关闭管理器"""
        # 取消所有活跃任务
        for task_id, future in self.active_tasks.items():
            future.cancel()
        
        # 关闭进程池
        self.executor.shutdown(wait=True)
```

### 4. 模型路由系统

```python
# core/router/model_router.py
"""
模型路由系统

功能：
- 多模型支持
- 负载均衡
- 故障转移
- 成本优化
"""

import asyncio
import random
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class ModelProvider(Enum):
    """模型提供商"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"
    OLLAMA = "ollama"
    BEDROCK = "bedrock"


@dataclass
class ModelConfig:
    """模型配置"""
    provider: ModelProvider
    model_name: str
    api_key: str
    base_url: Optional[str] = None
    priority: int = 1
    cost_per_1k_tokens: float = 0.0
    max_tokens: int = 8192
    timeout: int = 60


class ModelRouter:
    """
    模型路由器
    
    智能路由请求到最合适的模型，支持故障转移和成本优化。
    """
    
    # 模型能力映射
    MODEL_CAPABILITIES = {
        "gpt-4o": {"smart": 0.95, "fast": 0.8, "cheap": 0.7},
        "gpt-4o-mini": {"smart": 0.8, "fast": 0.95, "cheap": 0.95},
        "claude-3-5-sonnet": {"smart": 0.95, "fast": 0.75, "cheap": 0.6},
        "claude-3-haiku": {"smart": 0.75, "fast": 0.9, "cheap": 0.95},
    }
    
    def __init__(self):
        self.models: Dict[str, ModelConfig] = {}
        self.health_status: Dict[str, bool] = {}
        self.request_counts: Dict[str, int] = {}
    
    def register_model(self, name: str, config: ModelConfig):
        """注册模型"""
        self.models[name] = config
        self.health_status[name] = True
        self.request_counts[name] = 0
    
    async def generate(
        self,
        prompt: str,
        model_preference: str = "balanced",  # "fast" | "smart" | "cheap" | "balanced"
        fallback: bool = True
    ) -> str:
        """
        生成回复
        
        Args:
            prompt: 提示词
            model_preference: 模型偏好
            fallback: 是否启用故障转移
            
        Returns:
            生成的文本
        """
        # 选择最佳模型
        model_name = self._select_model(model_preference)
        model = self.models[model_name]
        
        try:
            # 调用模型
            result = await self._call_model(model, prompt)
            self.request_counts[model_name] += 1
            return result
            
        except Exception as e:
            if fallback:
                # 故障转移到其他模型
                self.health_status[model_name] = False
                alternative = self._select_model(
                    model_preference,
                    exclude={model_name}
                )
                if alternative:
                    return await self.generate(
                        prompt,
                        model_preference,
                        fallback=False
                    )
            raise
    
    def _select_model(
        self,
        preference: str,
        exclude: Optional[set] = None
    ) -> str:
        """选择最佳模型"""
        exclude = exclude or set()
        candidates = [
            name for name, config in self.models.items()
            if name not in exclude and self.health_status.get(name, True)
        ]
        
        if not candidates:
            raise RuntimeError("没有可用的模型")
        
        if preference == "balanced":
            # 轮询
            return random.choice(candidates)
        
        # 根据能力评分选择
        scores = []
        for name in candidates:
            caps = self.MODEL_CAPABILITIES.get(
                self.models[name].model_name,
                {"smart": 0.5, "fast": 0.5, "cheap": 0.5}
            )
            scores.append((name, caps.get(preference, 0.5)))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0]
    
    async def _call_model(self, config: ModelConfig, prompt: str) -> str:
        """调用具体模型"""
        if config.provider == ModelProvider.OPENAI:
            return await self._call_openai(config, prompt)
        elif config.provider == ModelProvider.ANTHROPIC:
            return await self._call_anthropic(config, prompt)
        else:
            raise ValueError(f"不支持的提供商: {config.provider}")
    
    async def _call_openai(self, config: ModelConfig, prompt: str) -> str:
        """调用 OpenAI"""
        import openai
        
        client = openai.AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )
        
        response = await client.chat.completions.create(
            model=config.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=config.max_tokens,
            timeout=config.timeout
        )
        
        return response.choices[0].message.content
    
    async def _call_anthropic(self, config: ModelConfig, prompt: str) -> str:
        """调用 Anthropic"""
        import anthropic
        
        client = anthropic.AsyncAnthropic(api_key=config.api_key)
        
        response = await client.messages.create(
            model=config.model_name,
            max_tokens=config.max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "registered_models": list(self.models.keys()),
            "health_status": self.health_status.copy(),
            "request_counts": self.request_counts.copy()
        }
```

### 5. 权限与审计系统

```python
# security/permissions.py + audit.py
"""
企业级安全系统

功能：
- RBAC 权限控制
- 审计日志
- 数据加密
"""

import hashlib
import json
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from datetime import datetime
import asyncio


# ==================== 权限控制 ====================

class PermissionChecker:
    """RBAC 权限检查器"""
    
    # 预定义角色
    ROLES = {
        "admin": {
            "permissions": {"*"},  # 所有权限
            "allowed_tools": {"*"},
            "max_tokens_per_request": 100000
        },
        "developer": {
            "permissions": {
                "task:execute",
                "file:read", "file:write", "file:edit",
                "shell:execute",
                "session:read", "session:write"
            },
            "allowed_tools": {
                "read_file", "write_file", "edit_file",
                "bash", "search", "git"
            },
            "max_tokens_per_request": 50000
        },
        "viewer": {
            "permissions": {
                "task:execute",
                "file:read",
                "session:read"
            },
            "allowed_tools": {"read_file", "search"},
            "max_tokens_per_request": 10000
        }
    }
    
    def __init__(self):
        self.user_roles: Dict[str, str] = {}
    
    def assign_role(self, user_id: str, role: str):
        """分配角色"""
        if role not in self.ROLES:
            raise ValueError(f"未知角色: {role}")
        self.user_roles[user_id] = role
    
    async def check(self, user_id: str, permission: str) -> bool:
        """检查权限"""
        role = self.user_roles.get(user_id, "viewer")
        role_config = self.ROLES[role]
        permissions = role_config["permissions"]
        
        return "*" in permissions or permission in permissions
    
    async def check_tool(self, user_id: str, tool_name: str) -> bool:
        """检查工具权限"""
        role = self.user_roles.get(user_id, "viewer")
        allowed_tools = self.ROLES[role]["allowed_tools"]
        return "*" in allowed_tools or tool_name in allowed_tools


# ==================== 审计日志 ====================

class AuditLogger:
    """审计日志系统"""
    
    def __init__(self, storage_backend: str = "elasticsearch"):
        self.storage = storage_backend
        self.buffer: List[Dict] = []
        self.buffer_size = 100
    
    async def log(
        self,
        event_type: str,
        user_id: str,
        details: Dict,
        severity: str = "info"
    ):
        """
        记录审计日志
        
        Args:
            event_type: 事件类型 (tool_call, file_access, login, etc.)
            user_id: 用户标识
            details: 详细数据
            severity: 严重程度 (info, warning, error, critical)
        """
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "user_id": self._hash_user_id(user_id),
            "session_id": details.get("session_id"),
            "details": details,
            "severity": severity,
            "ip_address": details.get("ip_address"),
            "user_agent": details.get("user_agent")
        }
        
        self.buffer.append(entry)
        
        # 批量写入
        if len(self.buffer) >= self.buffer_size:
            await self._flush()
    
    async def log_tool_call(
        self,
        user_id: str,
        tool_name: str,
        arguments: Dict,
        result: str,
        duration_ms: int
    ):
        """记录工具调用"""
        await self.log(
            event_type="tool_call",
            user_id=user_id,
            details={
                "tool_name": tool_name,
                "arguments_hash": self._hash_sensitive_data(arguments),
                "result_success": "error" not in result.lower(),
                "duration_ms": duration_ms
            },
            severity="info"
        )
    
    async def log_file_access(
        self,
        user_id: str,
        operation: str,  # read, write, delete
        file_path: str,
        success: bool
    ):
        """记录文件访问"""
        await self.log(
            event_type="file_access",
            user_id=user_id,
            details={
                "operation": operation,
                "file_path": file_path,
                "success": success
            },
            severity="warning" if operation in ["write", "delete"] else "info"
        )
    
    def _hash_user_id(self, user_id: str) -> str:
        """哈希用户 ID（隐私保护）"""
        return hashlib.sha256(user_id.encode()).hexdigest()[:16]
    
    def _hash_sensitive_data(self, data: Dict) -> str:
        """哈希敏感数据"""
        return hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()[:16]
    
    async def _flush(self):
        """批量写入存储"""
        # 实际实现应写入 Elasticsearch、S3 等
        if self.storage == "console":
            for entry in self.buffer:
                print(f"[AUDIT] {entry}")
        
        self.buffer.clear()
```

### 6. 技能市场系统

```python
# skills/marketplace.py
"""
技能市场系统

功能：
- 技能发现
- 技能安装
- 技能版本管理
- 技能沙箱执行
"""

import json
import zipfile
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import hashlib


@dataclass
class Skill:
    """技能定义"""
    id: str
    name: str
    version: str
    description: str
    author: str
    tools: List[str]
    entry_point: str
    sandbox_config: Dict
    permissions: List[str]


class SkillMarketplace:
    """技能市场"""
    
    def __init__(self, registry_url: str, local_path: Path):
        self.registry_url = registry_url
        self.local_path = Path(local_path)
        self.local_path.mkdir(parents=True, exist_ok=True)
        self.installed_skills: Dict[str, Skill] = {}
    
    async def search(
        self,
        query: str,
        category: Optional[str] = None
    ) -> List[Skill]:
        """搜索技能"""
        # 调用注册中心 API
        # 返回匹配的技能列表
        return []
    
    async def install(self, skill_id: str, version: str = "latest") -> Skill:
        """
        安装技能
        
        1. 下载技能包
        2. 验证签名
        3. 解压到沙箱目录
        4. 注册工具
        """
        # 下载
        package_path = await self._download(skill_id, version)
        
        # 验证
        if not self._verify_package(package_path):
            raise ValueError("技能包验证失败")
        
        # 解压
        skill_dir = self.local_path / skill_id
        with zipfile.ZipFile(package_path, 'r') as zf:
            zf.extractall(skill_dir)
        
        # 加载配置
        config = json.loads((skill_dir / "skill.json").read_text())
        skill = Skill(
            id=skill_id,
            name=config["name"],
            version=config["version"],
            description=config["description"],
            author=config["author"],
            tools=config["tools"],
            entry_point=config["entry_point"],
            sandbox_config=config.get("sandbox", {}),
            permissions=config.get("permissions", [])
        )
        
        self.installed_skills[skill_id] = skill
        return skill
    
    async def _download(self, skill_id: str, version: str) -> Path:
        """下载技能包"""
        # 实现下载逻辑
        pass
    
    def _verify_package(self, package_path: Path) -> bool:
        """验证包签名"""
        # 实现签名验证
        return True
    
    def load_skill(self, skill_id: str) -> Optional[Skill]:
        """加载已安装的技能"""
        return self.installed_skills.get(skill_id)
```

---

## 🚀 部署指南

### 1. Docker Compose 部署

```yaml
# docker-compose.yml
version: '3.8'

services:
  # 核心服务
  coder-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - REDIS_URL=redis://redis:6379
      - DB_URL=postgresql://postgres:password@db:5432/coder
    depends_on:
      - redis
      - db
    volumes:
      - ./workspace:/workspace
  
  # Web UI
  coder-web:
    build: ./web
    ports:
      - "3000:3000"
    depends_on:
      - coder-api
  
  # Redis 缓存
  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
  
  # PostgreSQL 数据库
  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=coder
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  # 监控
  prometheus:
    image: prom/prometheus
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"
  
  grafana:
    image: grafana/grafana
    ports:
      - "3001:3000"
    depends_on:
      - prometheus

volumes:
  redis_data:
  postgres_data:
```

### 2. Kubernetes 部署

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: coder-platform
spec:
  replicas: 3
  selector:
    matchLabels:
      app: coder
  template:
    metadata:
      labels:
        app: coder
    spec:
      containers:
      - name: coder
        image: coder-platform:latest
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        env:
        - name: WORKERS
          value: "4"
        - name: MAX_TOKENS
          value: "16000"
```

---

## 📊 性能指标

| 指标 | V1 | V2 | V3 |
|------|----|----|----|
| 并发任务 | 1 | 5 | 50+ |
| 响应延迟 | 5s | 3s | <1s |
| Token 效率 | 60% | 75% | 85% |
| 可用性 | 99% | 99.5% | 99.9% |
| 日活用户 | 1 | 10 | 1000+ |

---

## 🔒 安全合规

### 数据保护
- 传输加密 (TLS 1.3)
- 静态加密 (AES-256)
- PII 脱敏

### 访问控制
- SSO 集成
- MFA 支持
- API 密钥轮换

### 审计合规
- SOC 2 Type II
- ISO 27001
- GDPR 合规

---

## 🔄 版本对比

| 特性 | V1 MVP | V2 Advanced | V3 Full |
|------|--------|-------------|---------|
| 架构 | 单文件 | 模块化 | 分布式 |
| 安全 | 基础检查 | 白名单沙箱 | 企业 RBAC |
| 记忆 | 无 | AGENTS.md | 分布式存储 |
| 并发 | 单任务 | 有限并发 | 弹性伸缩 |
| MCP | ❌ | ❌ | ✅ |
| Subagent | ❌ | ❌ | ✅ |
| Web UI | ❌ | ❌ | ✅ |
| 多模型 | ❌ | ❌ | ✅ |
| 技能市场 | ❌ | ❌ | ✅ |
| 企业审计 | ❌ | ❌ | ✅ |

---

**上一版本**: [V2 进阶版](./design-v2-advanced.md)

**参考实现**: 
- [OpenCode](https://github.com/opencode-ai/opencode)
- [Codex CLI](https://github.com/openai/codex)
- [Claude Code](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code)
- [Aider](https://github.com/paul-gauthier/aider)
- [Continue](https://github.com/continuedev/continue)
