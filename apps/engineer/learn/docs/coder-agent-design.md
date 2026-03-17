# Coder Agent 架构设计文档

> 一个简化但实用的 AI 代码助手，参考 OpenCode、Codex CLI、Claude Code 的设计理念

**版本**: v1.0  
**日期**: 2026-03-17  
**状态**: 设计阶段  

---

## 📋 目录

1. [概述](#1-概述)
2. [架构设计](#2-架构设计)
3. [核心组件](#3-核心组件)
4. [数据流](#4-数据流)
5. [工具系统](#5-工具系统)
6. [安全机制](#6-安全机制)
7. [实现优先级](#7-实现优先级)
8. [接口定义](#8-接口定义)

---

## 1. 概述

### 1.1 设计目标

Coder Agent 是一个**简化但实用**的 AI 代码助手，旨在提供：

- ✅ **代码编辑**: 安全、可验证的文件修改
- ✅ **文件操作**: 读取、搜索、导航代码库
- ✅ **终端命令**: 受控的 shell 执行
- ✅ **对话管理**: 多轮对话和上下文追踪
- ✅ **可扩展性**: 支持 MCP 协议和自定义工具

### 1.2 参考系统

| 系统 | 核心借鉴点 |
|------|-----------|
| **OpenCode** | Agent 架构、工具权限系统、AGENTS.md 记忆 |
| **Codex CLI** | 沙箱安全机制、命令白名单 |
| **Claude Code** | 字符串替换编辑、子代理委派模式 |

### 1.3 设计原则

1. **简化优先**: 不过度工程化，优先实现核心功能
2. **安全默认**: 危险操作需要显式确认
3. **渐进增强**: MVP → 完整功能 → 高级特性
4. **代码清晰**: 易于理解和维护

---

## 2. 架构设计

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     CoderAgent (主入口)                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   Session    │  │   Memory     │  │   Tool Registry      │  │
│  │   Manager    │  │  (AGENTS.md) │  │   (Built-in + MCP)   │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                      Agent Loop (ReAct)                         │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  User Input → Intent Detect → Plan → Execute → Verify    │ │
│  └───────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                      工具层 (Tool Layer)                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │  read    │  │  write   │  │   edit   │  │    bash      │   │
│  │  glob    │  │  grep    │  │  skills  │  │   (sandbox)  │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 模块依赖关系

```
coder_agent.py (主入口)
    ├── ToolUseAgent (继承)
    │       └── BaseAgent
    ├── SessionManager (组合)
    ├── MemoryManager (组合)
    ├── ContextManager (组合)
    └── ToolRegistry (组合)
            ├── FileSystem Tools
            ├── SandboxShell
            └── MCP Client
```

---

## 3. 核心组件

### 3.1 CoderAgent

**职责**: 主入口类，协调所有组件

```python
class CoderAgent(ToolUseAgent):
    """
    简化版 Coder Agent - 核心代码助手
    
    核心能力：
    1. 读取和分析代码文件
    2. 安全地编辑文件（使用精确的字符串替换）
    3. 执行 shell 命令（在沙箱环境中）
    4. 搜索代码库
    """
    
    def __init__(
        self,
        name: str = "Coder",
        model: Optional[Model] = None,
        workspace: str = ".",
        enable_sandbox: bool = True,
        auto_confirm: bool = False,
    )
    
    def chat(self, message: str) -> str
    """主入口：处理用户消息"""
    
    def run_task(self, task_description: str) -> TaskResult
    """执行复杂任务（带规划和跟踪）"""
```

### 3.2 SessionManager

**职责**: 管理对话会话

```python
class SessionManager:
    """
    会话管理器
    
    功能：
    - 创建和跟踪会话
    - 管理对话历史
    - Token 预算控制
    """
    
    def create(self, session_id: Optional[str] = None) -> Session
    def get(self, session_id: str) -> Optional[Session]
    def add_message(self, session_id: str, message: Message)
    def get_history(self, session_id: str, limit: int = 10) -> List[Message]
```

### 3.3 MemoryManager

**职责**: 管理项目记忆（借鉴 OpenCode AGENTS.md）

```python
class MemoryManager:
    """
    项目记忆管理器
    
    自动加载：
    - ~/.config/coder/AGENTS.md (全局记忆)
    - {workspace}/AGENTS.md (项目记忆)
    """
    
    def __init__(self, workspace: Path)
    def load_context(self) -> str
    def save_project_memory(self, content: str)
```

**AGENTS.md 格式示例**:

```markdown
# 项目记忆

## 技术栈
- Python 3.11+
- FastAPI + SQLModel
- PostgreSQL

## 代码规范
- 使用 Ruff 进行代码格式化
- 类型注解必需
- 异步优先

## 常用命令
- `uv run pytest` - 运行测试
- `uv run ruff check .` - 代码检查
```

### 3.4 ContextManager

**职责**: 智能上下文管理

```python
class ContextManager:
    """
    上下文管理器
    
    功能：
    - 自动跟踪已读取的文件
    - Token 预算管理
    - 上下文压缩（当接近限制时）
    """
    
    def __init__(self, workspace: Path, token_budget: int = 8000)
    def add_file(self, path: str) -> str
    def get_context(self) -> str
    def compact(self) -> None  # 压缩上下文
```

### 3.5 SandboxShell

**职责**: 沙箱化的 Shell 执行（借鉴 Codex CLI）

```python
class SandboxShellTool(ShellTool):
    """
    沙箱化 Shell 工具
    
    安全特性：
    1. 命令白名单
    2. 路径限制
    3. 危险命令拦截
    4. 超时控制
    """
    
    DANGEROUS_PATTERNS = [
        r"rm\s+-rf\s+/",
        r"mkfs\.",
        r"dd\s+if=/dev/zero",
    ]
    
    def __init__(
        self,
        allowed_paths: List[str],
        allowed_commands: List[str],
        timeout: int = 60,
    )
```

---

## 4. 数据流

### 4.1 对话流程

```
用户输入
    │
    ▼
┌─────────────────────────────────────┐
│ 1. 接收输入                          │
│    CoderAgent.chat(message)         │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 2. 构建系统提示                      │
│    - 加载 AGENTS.md 记忆             │
│    - 获取已加载文件上下文             │
│    - 可用工具列表                     │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 3. LLM 决策                          │
│    可选操作：                        │
│    - 调用 read_file 获取上下文       │
│    - 调用 edit_file 修改代码         │
│    - 调用 bash 执行命令              │
│    - 直接回复                        │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 4. 权限检查                          │
│    - 危险命令拦截                     │
│    - 路径访问验证                     │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 5. 用户确认（如需要）                 │
│    需要确认的操作：                   │
│    - write_file                      │
│    - edit_file                       │
│    - bash（特定命令）                 │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 6. 执行工具                          │
│    - 文件操作                         │
│    - Shell 命令                       │
│    - 搜索                             │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 7. 返回结果                          │
│    - 更新对话历史                     │
│    - 继续或结束                       │
└─────────────────────────────────────┘
    │
    ▼
用户收到回复
```

### 4.2 文件编辑流程

```
用户: "修改 user.py 中的登录函数"

     │
     ▼
CoderAgent 分析需求
     │
     ▼
调用 read_file("user.py")
     │
     ▼
返回文件内容 → 添加到上下文
     │
     ▼
LLM 生成 edit_file 调用
     │
     ▼
安全验证：
- old_string 存在且唯一？
- 路径在工作区内？
     │
     ▼
用户确认（非自动模式）
     │
     ▼
执行替换
     │
     ▼
验证修改结果
     │
     ▼
返回成功/失败
```

---

## 5. 工具系统

### 5.1 内置工具列表

| 工具 | 功能 | 权限 | 需要确认 |
|------|------|------|---------|
| `read_file` | 读取文件内容 | read | 否 |
| `write_file` | 写入文件 | write | ✅ 是 |
| `edit_file` | 字符串替换编辑 | write | ✅ 是 |
| `glob` | 文件搜索 | read | 否 |
| `grep` | 内容搜索 | read | 否 |
| `list_dir` | 列出目录 | read | 否 |
| `bash` | 执行 shell | execute | 视命令而定 |

### 5.2 工具调用协议

```python
# 工具定义（OpenAI 格式）
tool_definition = {
    "type": "function",
    "function": {
        "name": "edit_file",
        "description": "编辑文件内容，替换指定的字符串",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
            },
            "required": ["file_path", "old_string", "new_string"]
        }
    }
}

# 工具执行结果
ToolResult = {
    "output": str,       # 执行输出
    "success": bool,     # 是否成功
    "error": str,        # 错误信息（如有）
    "elapsed_ms": float, # 执行时间
}
```

---

## 6. 安全机制

### 6.1 多层安全模型

```
Layer 1: 危险命令拦截（硬编码黑名单）
    ↓
Layer 2: 命令白名单检查
    ↓
Layer 3: 路径访问验证
    ↓
Layer 4: 用户确认（敏感操作）
```

### 6.2 安全策略配置

```python
# 安全策略
SecurityPolicy = {
    # 危险命令黑名单
    "blocked_commands": [
        "rm -rf /",
        "mkfs",
        "dd if=/dev/zero",
        ":(){ :|:& };:",
    ],
    
    # 命令白名单（可选）
    "allowed_commands": [
        "git", "ls", "cat", "grep", "find",
        "python", "pytest", "npm", "pip", "uv"
    ],
    
    # 路径限制
    "allowed_paths": ["/home/user/projects"],
    
    # 需要确认的操作
    "confirm_tools": ["write_file", "edit_file"],
}
```

### 6.3 编辑安全验证

```python
def validate_edit(file_path: str, old_string: str) -> ValidationResult:
    """
    验证编辑操作的安全性
    
    检查项：
    1. 文件在工作区内
    2. old_string 存在
    3. old_string 唯一（推荐）
    4. 替换后语法正确（可选）
    """
    pass
```

---

## 7. 实现优先级

### Phase 1: MVP（第 1 周）

**目标**: 最小可用版本，核心功能完整

- [ ] `CoderAgent` 基础框架
- [ ] 集成现有工具（read/write/edit/glob/grep）
- [ ] 基础对话循环
- [ ] 简单的权限检查

**验收标准**:
```python
coder = CoderAgent(model=Model())
result = coder.chat("读取 main.py 并告诉我它的功能")
# 应该能读取文件并给出分析
```

### Phase 2: 增强（第 2-3 周）

- [ ] `SandboxShell`（命令白名单 + 路径限制）
- [ ] `ContextManager`（文件跟踪 + Token 管理）
- [ ] `MemoryManager`（AGENTS.md 支持）
- [ ] 用户确认机制
- [ ] 流式输出支持

### Phase 3: 高级（后续）

- [ ] MCP 协议支持
- [ ] 上下文压缩
- [ ] Task 规划与跟踪
- [ ] SubAgent 委派
- [ ] 持久化存储（SQLite）

---

## 8. 接口定义

### 8.1 核心接口

```python
# ============ 主入口 ============

class CoderAgent:
    def __init__(
        self,
        name: str = "Coder",
        model: Optional[Model] = None,
        workspace: str = ".",
        enable_sandbox: bool = True,
        auto_confirm: bool = False,
    )
    
    def chat(self, message: str) -> str:
        """处理用户消息，返回回复"""
        pass
    
    def run_task(self, task: str) -> TaskResult:
        """执行复杂任务"""
        pass


# ============ 会话管理 ============

class SessionManager:
    def create(self, session_id: Optional[str] = None) -> Session
    def get(self, session_id: str) -> Optional[Session]
    def delete(self, session_id: str) -> bool


# ============ 记忆管理 ============

class MemoryManager:
    def load_context(self) -> str
    def save_project_memory(self, content: str) -> None


# ============ 上下文管理 ============

class ContextManager:
    def add_file(self, path: str) -> str
    def get_context(self) -> str
    def clear(self) -> None
    def compact(self) -> None
```

### 8.2 使用示例

```python
# examples/coder_demo.py

from apps.engineer.learn.coder.agents.coder_agent import CoderAgent
from apps.engineer.learn.coder.core.model import Model

def main():
    # 创建 Coder Agent
    coder = CoderAgent(
        name="MyCoder",
        model=Model(provider="openai", model="gpt-4o"),
        workspace="/path/to/your/project",
        enable_sandbox=True,
        auto_confirm=False,  # 安全模式
    )
    
    # 交互式对话
    print("=" * 60)
    print("Coder Agent - 输入 'quit' 退出")
    print("=" * 60)
    
    while True:
        user_input = input("\n👤 你: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            break
        
        response = coder.chat(user_input)
        print(f"\n🤖 Agent: {response}")

if __name__ == "__main__":
    main()
```

---

## 9. 文件结构

```
apps/engineer/learn/coder/
├── agents/
│   ├── __init__.py
│   ├── coder_agent.py          # ← 主入口（新建）
│   └── tool_use_agent.py       # 已有基础
├── core/
│   ├── editor/
│   │   ├── __init__.py
│   │   ├── code_editor.py      # 代码编辑器
│   │   ├── sandbox_shell.py    # 沙箱 shell
│   │   └── context_manager.py  # 上下文管理
│   ├── memory/
│   │   ├── __init__.py
│   │   └── project_memory.py   # 项目记忆
│   └── tools/
│       └── builtin/            # 已有内置工具
├── examples/
│   └── coder_demo.py           # 使用示例
└── AGENTS.md                   # 项目记忆文件
```

---

## 10. 附录

### 10.1 参考资源

- [OpenCode Documentation](https://opencode.ai/docs/)
- [OpenCode GitHub](https://github.com/anomalyco/opencode)
- [Codex CLI GitHub](https://github.com/openai/codex)
- [Claude Code Documentation](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview)

### 10.2 术语表

| 术语 | 说明 |
|------|------|
| **Agent** | 自主执行任务的 AI 实体 |
| **MCP** | Model Context Protocol，模型上下文协议 |
| **ReAct** | Reasoning + Acting，推理和行动结合的 Agent 模式 |
| **Sandbox** | 沙箱，受控的执行环境 |
| **Tool Use** | 工具使用，LLM 调用外部工具的能力 |

---

**作者**: AI Assistant  
**审核**: 待审核  
**状态**: 设计完成，待实现
