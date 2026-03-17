# CodeAgent 系统设计文档

**版本**: 1.0  
**日期**: 2026-03-17  
**目标**: 对标 OpenCode/Claude Code 的全功能终端 AI 编程助手

---

## 1. 概述

### 1.1 项目定位

CodeAgent 是一个基于 Python 的终端 AI 编程助手，对标 OpenCode 和 Claude Code，提供以下核心能力：

- **多模式 Agent**: Ask(问答)、Code(编码)、Architect(架构)三种模式自动/手动切换
- **完整工具链**: 文件操作、代码搜索、Bash 执行、LSP 集成、MCP 协议支持
- **子 Agent 委派**: 并行执行独立任务，提升效率
- **持久化会话**: SQLite 存储对话历史和文件版本
- **Terminal UI**: 基于 Textual 的现代化终端界面

### 1.2 技术栈

| 层级 | 技术 |
|------|------|
| 语言 | Python 3.11+ |
| Agent 框架 | 自研 (基于 ToolUseAgent) |
| LLM 接口 | OpenAI API / Anthropic / 阿里云等 |
| TUI 框架 | Textual |
| 持久化 | SQLite + Pydantic |
| LSP 客户端 | python-lsp-client |
| MCP 协议 | FastMCP |

### 1.3 参考实现

- **OpenCode**: https://github.com/opencode-ai/opencode (Go, Bubble Tea)
- **Claude Code**: Anthropic 官方代码助手

---

## 2. 架构设计

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        Terminal UI Layer                         │
│  (Textual - 聊天界面、文件树、工具输出、快捷键)                    │
├─────────────────────────────────────────────────────────────────┤
│                         Agent Layer                              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    CodeAgent                             │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │   │
│  │  │IntentRouter │→ │  Mode Impl  │→ │  Tool Selection │  │   │
│  │  │  (模式路由)  │  │(Ask/Code/  │  │   (工具选择)     │  │   │
│  │  │             │  │ Architect)  │  │                 │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                        Tool Layer                                │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │  File    │ │  Code    │ │   LSP    │ │ SubAgent │          │
│  │  Tools   │ │  Search  │ │  Tools   │ │   Tool   │          │
│  │(read/   │ │(grep/   │ │(diagnost-│ │(并行任务)│          │
│  │write/   │ │glob)    │ │ics/goto) │ │          │          │
│  │edit)    │ │         │ │          │ │          │          │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘          │
│  ┌──────────┐ ┌──────────┐                                     │
│  │   Bash   │ │   MCP    │                                     │
│  │   Tool   │ │  Client  │                                     │
│  │(shell   │ │(外部工具)│                                     │
│  │执行)    │ │          │                                     │
│  └──────────┘ └──────────┘                                     │
├─────────────────────────────────────────────────────────────────┤
│                      Session Layer                               │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐    │
│  │SessionManager  │  │ CodeSession    │  │ FileVersion    │    │
│  │(会话管理)       │  │(代码会话)      │  │(文件版本控制)   │    │
│  └────────────────┘  └────────────────┘  └────────────────┘    │
├─────────────────────────────────────────────────────────────────┤
│                      LLM Layer                                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │  OpenAI  │ │Anthropic │ │  Qwen    │ │  Ollama  │          │
│  │  Model   │ │  Model   │ │  Model   │ │  Model   │          │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 核心类图

```
BaseAgent (abstract)
    │
    ├── name, description, model, max_steps
    ├── run(), arun()
    │
    ↓
ToolUseAgent
    │
    ├── tool_manager: ToolManager
    ├── tool_executor: ToolExecutor
    ├── message_history: List[Dict]
    │
    ├── invoke(), stream()
    ├── call_tool(), _execute_tool_calls()
    │
    ↓
SessionAgent
    │
    ├── session_manager: SessionManager
    ├── current_session_id: str
    │
    ├── run(session_id, input)
    ├── load_session_history()
    │
    ↓
McpAgent
    │
    ├── mcp_client: FastMCP Client
    ├── _call_mcp_tool()
    │
    ↓
CodeAgent (本系统核心)
    │
    ├── intent_router: IntentRouter
    ├── mode: AgentMode (ASK/CODE/ARCHITECT)
    ├── lsp_client: LspClient
    │
    ├── run(), chat(), execute_task()
    ├── switch_mode(), detect_intent()
    └── delegate_to_subagent()
```

### 2.3 组件职责

| 组件 | 职责 | 关键类 |
|------|------|--------|
| **CodeAgent** | 主 Agent，协调所有功能 | `CodeAgent` |
| **IntentRouter** | 识别用户意图，路由到对应模式 | `IntentRouter` |
| **ToolManager** | 工具注册、查找、执行 | `ToolManager`, `ToolExecutor` |
| **SessionManager** | 会话创建、保存、恢复 | `CodeSessionManager` |
| **LspClient** | 语言服务器协议客户端 | `LspClient`, `LspTools` |
| **McpClient** | MCP 协议客户端 | `FastMCP` |
| **TUI** | 终端用户界面 | `CodeApp` (Textual) |

---

## 3. 模式设计 (Agent Modes)

### 3.1 三种核心模式

```python
class AgentMode(Enum):
    ASK = "ask"              # 问答模式
    CODE = "code"            # 编码模式
    ARCHITECT = "architect"  # 架构模式
```

### 3.2 Ask 模式

**用途**: 快速问答、代码解释、概念说明

**特点**:
- 不执行文件修改操作
- 可用工具有限: `read`, `grep` (只读)
- 响应快速，适合简单查询

**系统提示词策略**:
```
You are a helpful coding assistant. Answer the user's questions clearly.
You can read files and search code to provide accurate answers.
DO NOT modify any files in this mode.
```

**自动触发条件**:
- 查询以 "What", "How", "Why", "Explain" 开头
- 查询包含 "什么是", "怎么", "为什么", "解释"
- 没有要求修改代码的明确指令

### 3.3 Code 模式

**用途**: 文件操作、代码编辑、bug 修复

**特点**:
- 完整的文件操作工具链
- 代码搜索和导航
- Bash 命令执行
- LSP 诊断支持

**系统提示词策略**:
```
You are a code editing assistant. Help users modify and improve code.
Available tools:
- read: View file contents
- write: Create or overwrite files
- edit: Make precise changes to files
- grep: Search code patterns
- glob: Find files
- bash: Execute shell commands

Follow best practices:
1. Read relevant files before editing
2. Make minimal, focused changes
3. Verify changes with diagnostics
4. Run tests if available
```

**自动触发条件**:
- 查询包含 "添加", "修改", "修复", "实现", "删除"
- 查询包含 "add", "fix", "implement", "change", "update"
- 用户明确提到文件或代码修改

### 3.4 Architect 模式

**用途**: 复杂任务规划、多文件重构、系统设计

**特点**:
- 集成 TaskAgent 的规划能力
- 自动分解任务为子任务
- 并行执行独立子任务
- 完整的项目上下文管理

**系统提示词策略**:
```
You are a software architect. Break down complex tasks into manageable steps.

Workflow:
1. Analyze the requirements
2. Create a plan with subtasks
3. Delegate independent tasks to subagents
4. Coordinate the execution
5. Verify and integrate results

You can:
- Create task plans
- Delegate to specialized subagents
- Review and refine implementations
- Ensure consistency across changes
```

**自动触发条件**:
- 查询包含 "重构", "重新设计", "添加功能", "项目"
- 查询包含 "refactor", "redesign", "implement feature", "project"
- 任务明显需要多个步骤完成
- 涉及多个文件或模块

### 3.5 模式切换

```python
class IntentRouter:
    """自动识别意图并路由到对应模式"""

    def route(self, query: str, context: dict) -> AgentMode:
        # 1. 检查用户是否明确指定模式
        if context.get("force_mode"):
            return context["force_mode"]

        # 2. 使用启发式规则判断
        if self._is_simple_question(query):
            return AgentMode.ASK
        elif self._requires_planning(query):
            return AgentMode.ARCHITECT
        else:
            return AgentMode.CODE

    def _is_simple_question(self, query: str) -> bool:
        patterns = [
            r"^(what|how|why|explain|describe)",
            r"^(什么是|怎么|为什么|解释|描述)",
            r"(meaning|purpose|用法|作用)",
        ]
        return any(re.search(p, query, re.I) for p in patterns)

    def _requires_planning(self, query: str) -> bool:
        patterns = [
            r"(implement|refactor|redesign|添加功能|重构|重新设计)",
            r"(multiple files|several modules|多个文件|几个模块)",
            r"(project|application|项目|应用)",
        ]
        return any(re.search(p, query, re.I) for p in patterns)
```

---

## 4. 工具系统设计

### 4.1 工具分类

| 类别 | 工具 | 用途 |
|------|------|------|
| **文件操作** | read, write, edit | 文件读写和编辑 |
| **代码搜索** | grep, glob | 代码查找和文件定位 |
| **命令执行** | bash | Shell 命令执行 |
| **代码智能** | diagnostics, goto_def, find_refs | LSP 功能 |
| **任务委派** | subagent | 创建子 Agent 并行任务 |
| **外部工具** | mcp_* | MCP 服务器提供的工具 |

### 4.2 核心工具定义

#### FileReadTool

```python
class ReadFileInput(BaseModel):
    file_path: str = Field(description="要读取的文件路径")
    offset: Optional[int] = Field(None, description="起始行号(1-based)")
    limit: Optional[int] = Field(None, description="读取的最大行数")

class FileReadTool(BaseTool):
    name = "read"
    description = "读取文件内容，支持指定行范围"
    args_schema = ReadFileInput

    def _run(self, file_path: str, offset: Optional[int] = None,
             limit: Optional[int] = None) -> str:
        # 安全检查: 路径是否在 workspace 内
        full_path = self._resolve_path(file_path)

        if not full_path.exists():
            return f"Error: File not found: {file_path}"

        lines = full_path.read_text().splitlines()

        if offset:
            lines = lines[offset-1:]
        if limit:
            lines = lines[:limit]

        # 添加行号
        numbered = []
        for i, line in enumerate(lines, start=offset or 1):
            numbered.append(f"{i:4d}| {line}")

        return "\n".join(numbered)
```

#### FileWriteTool

```python
class WriteFileInput(BaseModel):
    file_path: str = Field(description="要写入的文件路径")
    content: str = Field(description="文件内容")

class FileWriteTool(BaseTool):
    name = "write"
    description = "写入文件(覆盖原有内容)"
    args_schema = WriteFileInput

    def _run(self, file_path: str, content: str) -> str:
        full_path = self._resolve_path(file_path)

        # 保存旧版本用于撤销
        if full_path.exists():
            self._backup_file(full_path)

        # 确保目录存在
        full_path.parent.mkdir(parents=True, exist_ok=True)

        full_path.write_text(content)
        return f"Successfully wrote {len(content)} characters to {file_path}"
```

#### FileEditTool

```python
class EditFileInput(BaseModel):
    file_path: str = Field(description="要编辑的文件路径")
    old_string: str = Field(description="要替换的字符串(必须精确匹配)")
    new_string: str = Field(description="新的字符串")

class FileEditTool(BaseTool):
    name = "edit"
    description = "精确编辑文件中的字符串"
    args_schema = EditFileInput

    def _run(self, file_path: str, old_string: str,
             new_string: str) -> str:
        full_path = self._resolve_path(file_path)

        if not full_path.exists():
            return f"Error: File not found: {file_path}"

        content = full_path.read_text()

        if old_string not in content:
            return f"Error: Could not find the specified text in {file_path}"

        # 保存备份
        self._backup_file(full_path)

        # 执行替换
        new_content = content.replace(old_string, new_string, 1)
        full_path.write_text(new_content)

        return f"Successfully edited {file_path}"
```

#### GrepTool

```python
class GrepInput(BaseModel):
    pattern: str = Field(description="搜索的正则表达式")
    path: Optional[str] = Field(None, description="搜索路径")
    include: Optional[str] = Field(None, description="文件匹配模式(如 '*.py')")
    output_mode: str = Field("content", description="输出模式: content/files/count")

class GrepTool(BaseTool):
    name = "grep"
    description = "搜索文件内容"
    args_schema = GrepInput

    def _run(self, pattern: str, path: Optional[str] = None,
             include: Optional[str] = None, output_mode: str = "content") -> str:
        search_path = self._resolve_path(path or ".")

        import re
        import fnmatch

        matches = []

        for root, dirs, files in os.walk(search_path):
            # 跳过隐藏目录和常见非代码目录
            dirs[:] = [d for d in dirs if not d.startswith('.')
                      and d not in ['node_modules', '__pycache__', '.git']]

            for filename in files:
                if include and not fnmatch.fnmatch(filename, include):
                    continue

                file_path = Path(root) / filename
                try:
                    content = file_path.read_text()
                    for i, line in enumerate(content.splitlines(), 1):
                        if re.search(pattern, line):
                            matches.append({
                                'file': str(file_path.relative_to(search_path)),
                                'line': i,
                                'content': line.strip()
                            })
                except:
                    continue

        if output_mode == "files":
            files = set(m['file'] for m in matches)
            return "\n".join(sorted(files))
        elif output_mode == "count":
            return f"Found {len(matches)} matches"
        else:
            results = []
            for m in matches[:50]:  # 限制结果数量
                results.append(f"{m['file']}:{m['line']}: {m['content']}")
            return "\n".join(results)
```

#### GlobTool

```python
class GlobInput(BaseModel):
    pattern: str = Field(description="文件匹配模式")
    path: Optional[str] = Field(None, description="搜索路径")

class GlobTool(BaseTool):
    name = "glob"
    description = "查找匹配模式的文件"
    args_schema = GlobInput

    def _run(self, pattern: str, path: Optional[str] = None) -> str:
        search_path = self._resolve_path(path or ".")
        matches = list(search_path.rglob(pattern))
        return "\n".join(str(m.relative_to(search_path)) for m in matches)
```

#### BashTool

```python
class BashInput(BaseModel):
    command: str = Field(description="要执行的命令")
    timeout: int = Field(60, description="超时时间(秒)")
    workdir: Optional[str] = Field(None, description="工作目录")

class BashTool(BaseTool):
    name = "bash"
    description = "执行 shell 命令"
    args_schema = BashInput

    # 危险命令模式
    DANGEROUS_PATTERNS = [
        r"rm\s+-rf\s+/",
        r">\s*/",
        r"dd\s+if=",
        r"mkfs",
        r":(){ :|:& };:",  # Fork bomb
    ]

    def _run(self, command: str, timeout: int = 60,
             workdir: Optional[str] = None) -> str:
        # 安全检查
        if self._is_dangerous(command):
            return "Error: Command rejected for security reasons"

        import subprocess

        cwd = self._resolve_path(workdir) if workdir else None

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd
            )

            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]: {result.stderr}"

            if result.returncode != 0:
                output += f"\n[Exit code: {result.returncode}]"

            return output
        except subprocess.TimeoutExpired:
            return f"Error: Command timed out after {timeout} seconds"
        except Exception as e:
            return f"Error: {str(e)}"

    def _is_dangerous(self, command: str) -> bool:
        import re
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, command, re.I):
                return True
        return False
```

#### SubAgentTool

```python
class SubAgentInput(BaseModel):
    description: str = Field(description="任务描述")
    prompt: str = Field(description="给子 Agent 的详细指令")
    category: str = Field("quick", description="任务类别: quick/deep/unspecified")

class SubAgentTool(BaseTool):
    name = "subagent"
    description = "创建子 Agent 并行执行任务"
    args_schema = SubAgentInput

    def _run(self, description: str, prompt: str,
             category: str = "quick") -> str:
        # 使用 task() 创建后台 Agent
        from opencode import task

        result = task(
            category=category,
            description=description,
            prompt=prompt,
            run_in_background=True,
            load_skills=[]
        )

        return f"Subagent created: {result}"
```

### 4.3 LSP 工具

#### LspDiagnosticsTool

```python
class LspDiagnosticsInput(BaseModel):
    file_path: Optional[str] = Field(None, description="文件路径(可选)")

class LspDiagnosticsTool(BaseTool):
    name = "diagnostics"
    description = "获取代码诊断信息(错误/警告)"
    args_schema = LspDiagnosticsInput

    def _run(self, file_path: Optional[str] = None) -> str:
        # 连接到 LSP 服务器
        # 返回诊断信息
        pass
```

---

## 5. 会话系统设计

### 5.1 会话数据结构

```python
class CodeSession(Session):
    """增强的代码 Agent 会话"""

    session_id: str
    workspace: str                    # 工作目录
    git_branch: Optional[str]         # Git 分支
    open_files: List[str]             # 打开的文件列表
    file_versions: Dict[str, List[FileVersion]]  # 文件版本历史
    mode: AgentMode                   # 当前模式
    created_at: datetime
    updated_at: datetime
    messages: List[Message]
    metadata: Dict[str, Any]

class FileVersion(BaseModel):
    """文件版本记录"""
    timestamp: datetime
    content_hash: str
    content: str                      # 完整内容(压缩存储)
    operation: str                    # "write", "edit", "delete"
    agent_message: Optional[str]      # 触发修改的 Agent 消息
```

### 5.2 会话管理器

```python
class CodeSessionManager:
    """代码会话管理器 - SQLite 持久化"""

    def __init__(self, db_path: str = ".codeagent/sessions.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """初始化数据库表"""
        # 创建 sessions, messages, file_versions 表
        pass

    def create_session(self, workspace: str, user_id: Optional[str] = None) -> CodeSession:
        """创建新会话"""
        session = CodeSession(
            session_id=str(uuid4()),
            workspace=workspace,
            git_branch=self._get_git_branch(workspace),
            open_files=[],
            file_versions={},
            mode=AgentMode.CODE,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            messages=[],
            metadata={}
        )
        self._save_session_to_db(session)
        return session

    def get_session(self, session_id: str) -> Optional[CodeSession]:
        """获取会话"""
        pass

    def save_message(self, session_id: str, message: Message):
        """保存消息"""
        pass

    def save_file_version(self, session_id: str, file_path: str,
                         version: FileVersion):
        """保存文件版本"""
        pass

    def get_file_version(self, session_id: str, file_path: str,
                        timestamp: datetime) -> Optional[FileVersion]:
        """获取特定版本的文件"""
        pass

    def list_sessions(self, workspace: Optional[str] = None) -> List[SessionSummary]:
        """列出会话"""
        pass

    def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        pass
```

### 5.3 文件版本控制

```python
class FileVersionManager:
    """管理文件版本历史，支持撤销"""

    def __init__(self, session: CodeSession):
        self.session = session

    def backup(self, file_path: str, operation: str, agent_message: Optional[str] = None):
        """备份文件当前状态"""
        full_path = Path(self.session.workspace) / file_path
        if not full_path.exists():
            return

        content = full_path.read_text()
        version = FileVersion(
            timestamp=datetime.now(),
            content_hash=hashlib.md5(content.encode()).hexdigest(),
            content=content,
            operation=operation,
            agent_message=agent_message
        )

        if file_path not in self.session.file_versions:
            self.session.file_versions[file_path] = []

        self.session.file_versions[file_path].append(version)

        # 限制历史长度(保留最近 20 个版本)
        self.session.file_versions[file_path] = \
            self.session.file_versions[file_path][-20:]

    def restore(self, file_path: str, steps_back: int = 1) -> bool:
        """恢复到之前的版本"""
        versions = self.session.file_versions.get(file_path, [])
        if len(versions) < steps_back:
            return False

        version = versions[-steps_back]
        full_path = Path(self.session.workspace) / file_path
        full_path.write_text(version.content)
        return True

    def undo_last_change(self) -> Optional[str]:
        """撤销最后一次修改"""
        # 找到最近修改的文件
        all_versions = []
        for file_path, versions in self.session.file_versions.items():
            for v in versions:
                all_versions.append((file_path, v))

        if not all_versions:
            return None

        # 按时间排序
        all_versions.sort(key=lambda x: x[1].timestamp, reverse=True)
        latest_file, _ = all_versions[0]

        if self.restore(latest_file, steps_back=2):
            return f"Restored {latest_file} to previous version"
        return None
```

---

## 6. TUI 界面设计

### 6.1 界面布局

```
┌────────────────────────────────────────────────────────────────┐
│  CodeAgent v0.1.0                    [Ask|Code|Architect] 🟢    │
├──────────────────────────┬─────────────────────────────────────┤
│                          │                                     │
│  📁 File Tree            │  💬 Chat History                   │
│  ├── src/                │  ─────────────────────────────     │
│  │   ├── main.py         │  👤 User: 帮我添加一个用户认证功能    │
│  │   └── utils.py        │                                     │
│  ├── tests/              │  🤖 Assistant:                     │
│  │   └── test_auth.py    │  我来帮你实现用户认证功能。首先让     │
│  └── README.md           │  我查看一下项目结构...               │
│                          │                                     │
│  🔧 Recent Tools         │  🔍 Tool: glob("**/*.py")           │
│  ✓ read src/main.py      │  → Found 12 files                   │
│  ⏳ edit src/auth.py     │                                     │
│  ✗ bash git status       │  💭 Thinking:                       │
│                          │  项目使用Flask框架，我需要...        │
│                          │                                     │
│                          │  🔧 Tool: write("src/auth.py")      │
│                          │  ✓ Done (45ms)                      │
│                          │                                     │
│                          │  ✅ 已完成：创建了auth模块          │
│                          │     - login/logout路由              │
│                          │     - JWT token支持                 │
│                          │     - 密码哈希                      │
├──────────────────────────┴─────────────────────────────────────┤
│  > 输入消息...                                            [Send] │
│  Ctrl+Enter: 换行  |  Ctrl+K: 命令  |  Ctrl+L: 日志  |  ?: 帮助 │
└────────────────────────────────────────────────────────────────┘
```

### 6.2 组件结构

```python
class CodeApp(App):
    """CodeAgent TUI 应用"""

    CSS = """
    /* Textual CSS */
    """

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(id="sidebar"):
                yield FileTree()
                yield ToolHistory()
            with Vertical(id="main"):
                yield ChatHistory()
                yield InputBox()
        yield Footer()

class ChatHistory(ScrollView):
    """聊天历史显示"""

    def add_user_message(self, content: str):
        """添加用户消息"""
        pass

    def add_assistant_message(self, content: str):
        """添加助手消息"""
        pass

    def add_tool_call(self, tool_name: str, args: dict):
        """添加工具调用显示"""
        pass

    def add_tool_result(self, result: str):
        """添加工具结果"""
        pass

    def add_thinking(self, content: str):
        """添加思考过程(流式)"""
        pass

class FileTree(DirectoryTree):
    """文件树组件"""

    def on_file_selected(self, event):
        """文件被选中"""
        pass

class InputBox(TextArea):
    """输入框组件"""

    BINDINGS = [
        Binding("ctrl+s", "send", "Send"),
        Binding("ctrl+enter", "newline", "New Line"),
    ]

    def action_send(self):
        """发送消息"""
        pass

class ToolHistory(Static):
    """最近工具调用历史"""
    pass
```

### 6.3 快捷键设计

| 快捷键 | 功能 |
|--------|------|
| `Ctrl+C` | 退出应用 |
| `Ctrl+K` | 打开命令面板 |
| `Ctrl+L` | 查看日志 |
| `Ctrl+M` | 切换模式 (Ask/Code/Architect) |
| `Ctrl+N` | 新会话 |
| `Ctrl+O` | 打开文件对话框 |
| `Ctrl+S` | 发送消息 |
| `Ctrl+Enter` | 输入框换行 |
| `Ctrl+Z` | 撤销最后一次文件修改 |
| `i` | 聚焦输入框 |
| `Esc` | 返回聊天视图 |
| `?` | 显示帮助 |

---

## 7. 配置系统

### 7.1 配置文件结构

```json
{
  "version": "1.0",

  "agents": {
    "coder": {
      "model": "claude-3-sonnet-20240229",
      "provider": "anthropic",
      "maxTokens": 4000,
      "temperature": 0.7
    },
    "planner": {
      "model": "claude-3-haiku-20240307",
      "provider": "anthropic",
      "maxTokens": 2000,
      "temperature": 0.3
    }
  },

  "tools": {
    "bash": {
      "requireConfirmation": true,
      "dangerousPatterns": [
        "rm\\s+-rf",
        ">\\s*/",
        "dd\\s+if=",
        "mkfs",
        ":\\(\\)\\{"
      ],
      "timeout": 60
    },
    "file": {
      "maxSize": 1048576,
      "backupVersions": 20
    },
    "lsp": {
      "python": {
        "command": "pylsp",
        "args": [],
        "disabled": false
      },
      "typescript": {
        "command": "typescript-language-server",
        "args": ["--stdio"],
        "disabled": false
      },
      "go": {
        "command": "gopls",
        "disabled": false
      }
    }
  },

  "mcpServers": {
    "github": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "${GITHUB_TOKEN}"
      }
    },
    "fetch": {
      "type": "stdio",
      "command": "uvx",
      "args": ["mcp-server-fetch"]
    }
  },

  "ui": {
    "theme": "dark",
    "fontSize": 14,
    "showThinking": true,
    "showToolResults": true,
    "messageWindow": 20
  },

  "session": {
    "autoSave": true,
    "compactThreshold": 0.8,
    "maxMessages": 100
  }
}
```

### 7.2 配置加载优先级

1. 默认配置 (内置)
2. `~/.config/codeagent/config.json` (用户配置)
3. `./.codeagent/config.json` (项目配置)
4. 环境变量 (覆盖配置)

### 7.3 环境变量

| 变量名 | 用途 |
|--------|------|
| `OPENAI_API_KEY` | OpenAI API 密钥 |
| `ANTHROPIC_API_KEY` | Anthropic API 密钥 |
| `DASHSCOPE_API_KEY` | 阿里云 Dashscope |
| `GITHUB_TOKEN` | GitHub API 令牌 |
| `CODEAGENT_CONFIG` | 配置文件路径 |
| `CODEAGENT_WORKSPACE` | 默认工作目录 |

---

## 8. 实现路线图

### Phase 1: 核心框架 (2 周)

- [ ] 搭建项目结构
- [ ] 实现基础 Agent 类 (继承现有 ToolUseAgent)
- [ ] 实现文件操作工具 (read/write/edit)
- [ ] 实现代码搜索工具 (grep/glob)
- [ ] 实现 Bash 工具
- [ ] 基础 CLI 交互 (非 TUI)

### Phase 2: 会话与持久化 (1 周)

- [ ] 实现 CodeSession 数据结构
- [ ] 实现 SQLite 持久化
- [ ] 文件版本控制
- [ ] 会话恢复功能

### Phase 3: 模式系统 (1 周)

- [ ] 实现 IntentRouter
- [ ] 三种模式的系统提示词
- [ ] 模式切换功能
- [ ] 模式特定工具集

### Phase 4: 高级功能 (2 周)

- [ ] LSP 客户端集成
- [ ] MCP 协议支持
- [ ] 子 Agent 委派
- [ ] 并行任务执行

### Phase 5: TUI 界面 (2 周)

- [ ] Textual 基础界面
- [ ] 聊天历史显示
- [ ] 文件树组件
- [ ] 工具输出显示
- [ ] 快捷键系统

### Phase 6: 测试与优化 (1 周)

- [ ] 单元测试
- [ ] 集成测试
- [ ] 性能优化
- [ ] 文档完善

---

## 9. 目录结构

```
apps/codeagent/
├── codeagent/                  # 主包
│   ├── __init__.py
│   ├── __main__.py            # 入口点
│   ├── app.py                 # TUI 应用
│   ├── config.py              # 配置管理
│   ├── agent/                 # Agent 实现
│   │   ├── __init__.py
│   │   ├── base.py            # CodeAgent 主类
│   │   ├── modes.py           # 模式定义
│   │   └── router.py          # 意图路由
│   ├── tools/                 # 工具实现
│   │   ├── __init__.py
│   │   ├── file.py            # 文件工具
│   │   ├── search.py          # 搜索工具
│   │   ├── bash.py            # Bash 工具
│   │   ├── lsp.py             # LSP 工具
│   │   └── subagent.py        # 子 Agent 工具
│   ├── session/               # 会话管理
│   │   ├── __init__.py
│   │   ├── models.py          # 数据模型
│   │   ├── manager.py         # 会话管理器
│   │   └── version.py         # 版本控制
│   ├── lsp/                   # LSP 客户端
│   │   ├── __init__.py
│   │   ├── client.py          # LSP 客户端
│   │   └── tools.py           # LSP 工具封装
│   ├── ui/                    # TUI 组件
│   │   ├── __init__.py
│   │   ├── app.py             # 主应用
│   │   ├── chat.py            # 聊天组件
│   │   ├── filetree.py        # 文件树
│   │   └── widgets.py         # 其他组件
│   └── utils/                 # 工具函数
│       ├── __init__.py
│       ├── paths.py           # 路径处理
│       ├── git.py             # Git 操作
│       └── format.py          # 格式化
├── tests/                     # 测试
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── docs/                      # 文档
├── examples/                  # 示例
├── pyproject.toml            # 项目配置
└── README.md                 # 说明文档
```

---

## 10. 关键设计决策

### 10.1 为什么继承而不是组合?

选择从 `McpAgent` 继承而不是组合，因为：

1. **功能依赖**: CodeAgent 需要所有父类的功能 (ToolUse + Session + MCP)
2. **代码复用**: 复用现有的工具执行、会话管理、MCP 连接逻辑
3. **一致性**: 保持与现有 Agent 框架一致的开发模式

### 10.2 文件版本控制策略

- **保存时机**: 每次 `write` 或 `edit` 前自动备份
- **存储方式**: SQLite 中压缩存储 (使用 zlib)
- **保留数量**: 每个文件保留最近 20 个版本
- **撤销粒度**: 一次撤销恢复到上一个版本

### 10.3 模式切换策略

- **自动检测**: 使用轻量级规则匹配，不调用 LLM
- **手动覆盖**: 用户可通过快捷键或命令强制切换
- **模式记忆**: 同一会话内保持用户选择的模式

### 10.4 安全策略

- **Bash 命令**: 危险模式需要确认
- **文件操作**: 限制在 workspace 目录内
- **路径解析**: 使用 `_resolve_path()` 规范化路径
- **MCP 工具**: 继承 OpenCode 的权限系统

---

## 11. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| LSP 集成复杂 | 高 | 使用现有库，先实现基础诊断 |
| TUI 性能 | 中 | 使用 Textual 的优化模式，限制历史长度 |
| 工具调用循环 | 高 | 设置 max_steps，检测循环模式 |
| 文件冲突 | 中 | 文件版本控制，修改前读取确认 |
| 配置错误 | 低 | 配置验证，提供默认配置 |

---

## 12. 附录

### 12.1 术语表

| 术语 | 解释 |
|------|------|
| **Agent** | 能使用工具完成任务的 AI 实体 |
| **MCP** | Model Context Protocol，模型上下文协议 |
| **LSP** | Language Server Protocol，语言服务器协议 |
| **TUI** | Terminal User Interface，终端用户界面 |
| **Tool** | Agent 可调用的功能单元 |
| **Session** | 一次连续的对话上下文 |

### 12.2 参考资源

- OpenCode: https://github.com/opencode-ai/opencode
- Textual: https://textual.textualize.io/
- FastMCP: https://github.com/jlowin/fastmcp
- Python LSP: https://github.com/python-lsp/python-lsp-server

---

**文档版本**: 1.0  
**最后更新**: 2026-03-17  
**作者**: AI Assistant
