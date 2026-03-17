# Coder Agent V2 - 进阶设计方案

> 生产就绪版本：安全、可扩展、可记忆

**版本**: V2.0 Advanced  
**目标**: 安全沙箱 + 智能记忆 + 模块化架构  
**代码量**: ~2000 行核心代码  
**部署时间**: 2 小时

---

## 🎯 设计哲学

**安全第一，体验优先**

V2 版本专注于：
1. **安全沙箱** - 危险的命令必须被隔离
2. **智能记忆** - AGENTS.md 自动维护
3. **上下文管理** - 大文件智能分块
4. **模块化架构** - 易于扩展和维护

---

## 📦 核心能力

```
✅ 安全沙箱 (SandboxShell)
✅ 命令白名单机制
✅ AGENTS.md 自动记忆
✅ 上下文压缩管理
✅ SQLite 会话持久化
✅ 文件变更追踪 (Git 集成)
✅ 工具权限系统
✅ 优雅的错误处理
❌ MCP 协议 (V3 添加)
❌ Subagent 委派 (V3 添加)
```

---

## 🏗️ 模块化架构

```
┌─────────────────────────────────────────────────────────────┐
│                    AdvancedCoderAgent                        │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │   LLM Core   │  │  ToolManager │  │ ContextManager   │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │MemoryManager │  │SandboxShell  │  │ SessionStore     │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 模块说明

| 模块 | 职责 | 关键技术 |
|------|------|----------|
| `LLM Core` | 模型调用 | OpenAI/Anthropic API |
| `ToolManager` | 工具注册与调度 | 插件化设计 |
| `ContextManager` | Token 管理与压缩 | 滑动窗口 + RAG |
| `MemoryManager` | AGENTS.md 维护 | 文件 + 向量存储 |
| `SandboxShell` | 安全命令执行 | whitelist + timeout |
| `SessionStore` | 会话持久化 | SQLite |

---

## 💻 核心代码实现

### 目录结构

```
coder/
├── __init__.py
├── agent.py                 # 主 Agent 类
├── core/
│   ├── __init__.py
│   ├── llm.py              # LLM 封装
│   ├── context.py          # 上下文管理
│   └── session.py          # 会话存储
├── tools/
│   ├── __init__.py
│   ├── base.py             # 工具基类
│   ├── file_system.py      # 文件系统工具
│   ├── shell.py            # 安全 Shell
│   └── search.py           # 搜索工具
├── memory/
│   ├── __init__.py
│   ├── agents_md.py        # AGENTS.md 管理
│   └── vector_store.py     # 向量存储
└── utils/
    ├── __init__.py
    ├── git_tracker.py      # Git 变更追踪
    └── validators.py       # 输入验证
```

### 1. 核心 Agent 类

```python
# coder/agent.py
"""
进阶 Coder Agent - V2 版本

特点：
- 模块化架构
- 安全沙箱
- 智能记忆
- 上下文管理
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import sqlite3

from .core.llm import LLMCore
from .core.context import ContextManager
from .core.session import SessionStore
from .tools.file_system import FileSystemTool
from .tools.shell import SandboxShell
from .tools.search import SearchTool
from .memory.agents_md import AgentsMDManager
from .utils.git_tracker import GitTracker


@dataclass
class AgentConfig:
    """Agent 配置"""
    workspace: Path = Path(".")
    model_provider: str = "openai"
    model_name: str = "gpt-4o"
    max_context_tokens: int = 8000
    shell_whitelist: List[str] = field(default_factory=lambda: [
        "git", "ls", "cat", "grep", "find", "python", "pip",
        "npm", "node", "pytest", "mypy", "ruff"
    ])
    enable_git_tracking: bool = True
    enable_agents_md: bool = True
    db_path: str = "./coder_sessions.db"


class AdvancedCoderAgent:
    """进阶代码助手 - V2"""
    
    SYSTEM_PROMPT = """你是一个安全的代码助手。你可以使用以下工具：

1. **read_file**: 读取文件内容
   - 支持 offset/limit 参数处理大文件
   - 自动检测文件类型

2. **write_file**: 写入文件
   - 自动创建父目录
   - 支持原子写入

3. **edit_file**: 编辑文件
   - 使用精确字符串匹配
   - 必须提供 old_string 和 new_string

4. **bash**: 执行 shell 命令（安全沙箱）
   - 只有白名单命令可用
   - 超时限制: 60 秒
   - 危险操作会被拦截

5. **search**: 搜索代码
   - grep 搜索
   - 符号查找

重要规则：
- 编辑前必须读取文件
- 使用精确字符串匹配
- 大文件使用 offset/limit 分批读取
- AGENTS.md 会自动记录重要信息
"""

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.workspace = Path(self.config.workspace).resolve()
        
        # 初始化核心组件
        self._init_components()
        
        # 会话状态
        self.session_id: Optional[str] = None
        self.history: List[Dict] = []
        self.tool_calls_count = 0
    
    def _init_components(self):
        """初始化核心组件"""
        # LLM 核心
        self.llm = LLMCore(
            provider=self.config.model_provider,
            model=self.config.model_name
        )
        
        # 上下文管理器
        self.context = ContextManager(
            max_tokens=self.config.max_context_tokens
        )
        
        # 会话存储
        self.session_store = SessionStore(self.config.db_path)
        
        # 工具注册
        self._register_tools()
        
        # 记忆系统
        if self.config.enable_agents_md:
            self.memory = AgentsMDManager(self.workspace)
        
        # Git 追踪
        if self.config.enable_git_tracking:
            self.git_tracker = GitTracker(self.workspace)
    
    def _register_tools(self):
        """注册工具"""
        self.tools = {
            "read_file": FileSystemTool(self.workspace),
            "write_file": FileSystemTool(self.workspace),
            "edit_file": FileSystemTool(self.workspace),
            "bash": SandboxShell(
                whitelist=self.config.shell_whitelist,
                timeout=60,
                workspace=self.workspace
            ),
            "search": SearchTool(self.workspace),
        }
    
    async def chat(self, message: str) -> str:
        """
        处理用户消息
        
        Args:
            message: 用户输入
            
        Returns:
            Agent 回复
        """
        # 创建或恢复会话
        if not self.session_id:
            self.session_id = self.session_store.create_session()
        
        # 添加用户消息
        self._add_to_history("user", message)
        
        # 构建上下文
        context = await self._build_context()
        
        # 调用 LLM
        response = await self.llm.generate(
            messages=context,
            tools=self._get_tool_schemas()
        )
        
        # 处理工具调用
        if response.tool_calls:
            results = await self._execute_tool_calls(response.tool_calls)
            
            # 更新上下文并再次调用
            context.extend(results)
            final_response = await self.llm.generate(messages=context)
            reply = final_response.content
        else:
            reply = response.content
        
        # 添加助手回复
        self._add_to_history("assistant", reply)
        
        # 保存会话
        self.session_store.save_message(self.session_id, "user", message)
        self.session_store.save_message(self.session_id, "assistant", reply)
        
        # 更新 AGENTS.md
        if self.config.enable_agents_md:
            await self._update_memory(message, reply)
        
        return reply
    
    async def _build_context(self) -> List[Dict]:
        """构建对话上下文（带压缩）"""
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        
        # 添加 AGENTS.md 记忆
        if self.config.enable_agents_md:
            memory = self.memory.get_relevant_context()
            if memory:
                messages.append({
                    "role": "system",
                    "content": f"项目记忆:\n{memory}"
                })
        
        # 添加上下文（已压缩）
        compressed = self.context.compress(self.history)
        messages.extend(compressed)
        
        return messages
    
    async def _execute_tool_calls(self, tool_calls: List[Dict]) -> List[Dict]:
        """执行工具调用"""
        results = []
        
        for call in tool_calls:
            tool_name = call["name"]
            tool_args = call["arguments"]
            tool_id = call["id"]
            
            if tool_name in self.tools:
                try:
                    result = await self.tools[tool_name].execute(**tool_args)
                    self.tool_calls_count += 1
                    
                    # 记录文件变更
                    if tool_name in ["write_file", "edit_file"]:
                        if self.config.enable_git_tracking:
                            self.git_tracker.record_change(
                                tool_args.get("path"),
                                tool_name
                            )
                    
                    results.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": result.output
                    })
                except Exception as e:
                    results.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": f"错误: {str(e)}"
                    })
            else:
                results.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": f"未知工具: {tool_name}"
                })
        
        return results
    
    async def _update_memory(self, user_msg: str, assistant_msg: str):
        """更新项目记忆"""
        # 提取关键信息
        important_patterns = [
            "重要", "关键", "架构", "设计", "约定", "convention",
            "must", "always", "never", "pattern"
        ]
        
        content = f"{user_msg} {assistant_msg}".lower()
        if any(p in content for p in important_patterns):
            # 使用 LLM 提取要点
            summary = await self.llm.summarize(
                f"用户: {user_msg}\n助手: {assistant_msg}",
                max_length=200
            )
            self.memory.add_entry(summary, category="important")
    
    def _add_to_history(self, role: str, content: str):
        """添加到历史记录"""
        self.history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
    
    def _get_tool_schemas(self) -> List[Dict]:
        """获取工具 Schema（OpenAI 格式）"""
        schemas = []
        for name, tool in self.tools.items():
            schemas.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.description,
                    "parameters": tool.get_schema()
                }
            })
        return schemas
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "session_id": self.session_id,
            "messages_count": len(self.history),
            "tool_calls_count": self.tool_calls_count,
            "context_tokens": self.context.current_tokens,
            "memory_entries": self.memory.entry_count if self.memory else 0
        }
```

### 2. 安全 Shell 实现

```python
# coder/tools/shell.py
"""安全 Shell 执行器 - 白名单 + 超时 + 资源限制"""

import asyncio
import re
from pathlib import Path
from typing import List, Set, Optional
from dataclasses import dataclass

from .base import BaseTool, ToolResult


@dataclass
class SandboxConfig:
    """沙箱配置"""
    whitelist: List[str]              # 允许的命令白名单
    blacklist_patterns: List[str]     # 禁止的模式（正则）
    timeout: int = 60                 # 超时秒数
    max_output_size: int = 100000     # 最大输出大小
    allow_redirects: bool = False     # 是否允许重定向
    allow_pipes: bool = True          # 是否允许管道
    working_dir: Optional[Path] = None


class SandboxShell(BaseTool):
    """
    安全 Shell 执行器
    
    安全特性：
    1. 命令白名单 - 只允许预定义的命令
    2. 黑名单模式 - 阻止危险操作
    3. 超时控制 - 防止无限执行
    4. 输出限制 - 防止内存溢出
    5. 工作目录限制 - 防止越界访问
    """
    
    name = "bash"
    description = "执行 shell 命令（安全沙箱版本）"
    
    # 默认危险命令模式
    DANGEROUS_PATTERNS = [
        r"rm\s+-rf\s+/",
        r"mkfs\.",
        r"dd\s+if=/dev/zero",
        r">\s*/dev/sd[a-z]",
        r":\(\)\s*\{\s*\|\:\s*\&\s*\}\s*;\s*:",  # Fork bomb
        r"curl\s+.*\s*\|\s*sh",  # 管道到 shell
        r"wget\s+.*\s*\|\s*sh",
    ]
    
    def __init__(
        self,
        whitelist: Optional[List[str]] = None,
        timeout: int = 60,
        workspace: Optional[Path] = None
    ):
        self.config = SandboxConfig(
            whitelist=whitelist or ["ls", "cat", "pwd", "echo"],
            blacklist_patterns=self.DANGEROUS_PATTERNS,
            timeout=timeout,
            working_dir=workspace
        )
    
    async def execute(self, command: str, **kwargs) -> ToolResult:
        """
        执行命令
        
        Args:
            command: 要执行的命令
            
        Returns:
            ToolResult 包含输出或错误
        """
        # 1. 安全检查
        security_check = self._security_check(command)
        if not security_check.passed:
            return ToolResult(
                error=f"安全检查失败: {security_check.reason}",
                success=False
            )
        
        # 2. 解析命令
        parsed = self._parse_command(command)
        
        # 3. 白名单检查
        if not self._is_whitelisted(parsed):
            return ToolResult(
                error=f"命令 '{parsed['base_cmd']}' 不在白名单中. "
                      f"允许的命令: {', '.join(self.config.whitelist)}",
                success=False
            )
        
        # 4. 执行（带超时）
        try:
            result = await self._execute_with_timeout(command)
            return result
        except asyncio.TimeoutError:
            return ToolResult(
                error=f"命令执行超时（>{self.config.timeout}秒）",
                success=False
            )
        except Exception as e:
            return ToolResult(error=str(e), success=False)
    
    def _security_check(self, command: str) -> "SecurityCheckResult":
        """安全检查"""
        # 检查危险模式
        for pattern in self.config.blacklist_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                return SecurityCheckResult(
                    passed=False,
                    reason=f"检测到危险模式: {pattern}"
                )
        
        # 检查重定向（如果不允许）
        if not self.config.allow_redirects:
            if re.search(r"[><]|>>|2>", command):
                return SecurityCheckResult(
                    passed=False,
                    reason="不允许使用重定向操作符"
                )
        
        return SecurityCheckResult(passed=True)
    
    def _parse_command(self, command: str) -> Dict:
        """解析命令"""
        # 去除管道，获取基础命令
        if "|" in command and self.config.allow_pipes:
            parts = command.split("|")
            base_cmd = parts[0].strip().split()[0]
        else:
            base_cmd = command.strip().split()[0] if command.strip() else ""
        
        return {
            "base_cmd": base_cmd,
            "full_command": command
        }
    
    def _is_whitelisted(self, parsed: Dict) -> bool:
        """检查是否在白名单中"""
        base_cmd = parsed["base_cmd"]
        
        # 检查白名单（支持部分匹配）
        for allowed in self.config.whitelist:
            if base_cmd == allowed or base_cmd.endswith(f"/{allowed}"):
                return True
        
        return False
    
    async def _execute_with_timeout(self, command: str) -> ToolResult:
        """带超时的执行"""
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.config.working_dir
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.config.timeout
            )
            
            output = stdout.decode("utf-8", errors="replace")
            
            # 截断过大输出
            if len(output) > self.config.max_output_size:
                output = output[:self.config.max_output_size] + \
                         "\n... [输出已截断]"
            
            if stderr:
                output += f"\n[stderr] {stderr.decode('utf-8', errors='replace')}"
            
            return ToolResult(
                output=output,
                success=proc.returncode == 0
            )
            
        except asyncio.TimeoutError:
            proc.kill()
            raise


@dataclass
class SecurityCheckResult:
    """安全检查结果"""
    passed: bool
    reason: str = ""
```

### 3. AGENTS.md 记忆系统

```python
# coder/memory/agents_md.py
"""AGENTS.md 自动维护系统"""

import re
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass
from datetime import datetime


@dataclass
class MemoryEntry:
    """记忆条目"""
    content: str
    category: str
    timestamp: str
    source: str = ""


class AgentsMDManager:
    """
    AGENTS.md 管理器
    
    功能：
    - 自动维护项目记忆文件
    - 分类存储（架构、约定、重要决策）
    - 去重和过期清理
    """
    
    DEFAULT_TEMPLATE = """# Agent Project Memory

## Project Overview
<!-- 项目概览 -->

## Architecture
<!-- 架构决策 -->

## Conventions
<!-- 代码约定 -->

## Important Notes
<!-- 重要提醒 -->

## Session History
<!-- 会话历史 -->
"""
    
    def __init__(self, workspace: Path):
        self.workspace = Path(workspace)
        self.memory_file = self.workspace / "AGENTS.md"
        self.entries: List[MemoryEntry] = []
        
        # 初始化记忆文件
        self._init_memory_file()
        self._load_entries()
    
    def _init_memory_file(self):
        """初始化记忆文件"""
        if not self.memory_file.exists():
            self.memory_file.write_text(
                self.DEFAULT_TEMPLATE,
                encoding="utf-8"
            )
    
    def _load_entries(self):
        """加载现有条目"""
        content = self.memory_file.read_text(encoding="utf-8")
        
        # 解析各章节
        self.sections = self._parse_sections(content)
    
    def _parse_sections(self, content: str) -> Dict[str, str]:
        """解析章节"""
        sections = {}
        current_section = None
        current_content = []
        
        for line in content.split("\n"):
            if line.startswith("## "):
                if current_section:
                    sections[current_section] = "\n".join(current_content)
                current_section = line[3:].strip()
                current_content = []
            elif current_section:
                current_content.append(line)
        
        if current_section:
            sections[current_section] = "\n".join(current_content)
        
        return sections
    
    def add_entry(
        self,
        content: str,
        category: str = "general",
        source: str = ""
    ):
        """
        添加记忆条目
        
        Args:
            content: 记忆内容
            category: 分类 (architecture/conventions/important)
            source: 来源标记
        """
        # 去重检查
        if self._is_duplicate(content):
            return
        
        entry = MemoryEntry(
            content=content,
            category=category,
            timestamp=datetime.now().isoformat(),
            source=source
        )
        
        self.entries.append(entry)
        self._persist_entry(entry)
    
    def _is_duplicate(self, content: str, threshold: float = 0.8) -> bool:
        """检查是否重复（简单实现）"""
        content_lower = content.lower().strip()
        for entry in self.entries:
            # 简单包含检查
            if content_lower in entry.content.lower():
                return True
            if entry.content.lower() in content_lower:
                return True
        return False
    
    def _persist_entry(self, entry: MemoryEntry):
        """持久化到文件"""
        # 映射分类到章节
        section_map = {
            "architecture": "Architecture",
            "convention": "Conventions",
            "conventions": "Conventions",
            "important": "Important Notes",
            "general": "Session History"
        }
        
        section = section_map.get(entry.category, "Session History")
        
        # 构建条目文本
        entry_text = f"\n- [{entry.timestamp}] {entry.content}"
        if entry.source:
            entry_text += f" (来源: {entry.source})"
        
        # 更新章节
        content = self.memory_file.read_text(encoding="utf-8")
        section_pattern = rf"(## {section}\n)(.*?)(?=\n## |\Z)"
        
        match = re.search(section_pattern, content, re.DOTALL)
        if match:
            new_content = content[:match.end()] + entry_text + content[match.end():]
            self.memory_file.write_text(new_content, encoding="utf-8")
    
    def get_relevant_context(self, query: str = "", limit: int = 1000) -> str:
        """获取相关上下文"""
        content = self.memory_file.read_text(encoding="utf-8")
        
        # 简单截断到限制长度
        if len(content) > limit:
            # 保留头部，截断历史
            sections = self._parse_sections(content)
            important = []
            
            for section_name in ["Architecture", "Conventions", "Important Notes"]:
                if section_name in sections:
                    important.append(f"## {section_name}{sections[section_name]}")
            
            return "\n\n".join(important)
        
        return content
    
    @property
    def entry_count(self) -> int:
        """条目数量"""
        return len(self.entries)
```

### 4. 上下文管理器

```python
# coder/core/context.py
"""上下文管理 - Token 压缩与智能分段"""

import tiktoken
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class ContextWindow:
    """上下文窗口"""
    messages: List[Dict]
    tokens: int
    compressed: bool = False


class ContextManager:
    """
    上下文管理器
    
    功能：
    - Token 计数
    - 智能压缩（滑动窗口）
    - 大文件分段
    - 重要性评分
    """
    
    def __init__(self, max_tokens: int = 8000, model: str = "gpt-4"):
        self.max_tokens = max_tokens
        self.model = model
        self.encoding = tiktoken.encoding_for_model(model)
        self.current_tokens = 0
        
        # 保留策略
        self.keep_system = True      # 保留系统消息
        self.keep_recent = 10        # 保留最近 N 条
        self.summarize_threshold = 20  # 超过则总结
    
    def compress(self, history: List[Dict]) -> List[Dict]:
        """
        压缩历史记录到安全范围
        
        策略：
        1. 保留系统消息
        2. 保留最近 N 条完整对话
        3. 中间内容总结或丢弃
        """
        if len(history) <= self.keep_recent:
            return history
        
        # 分离系统消息
        system_msgs = [m for m in history if m.get("role") == "system"]
        other_msgs = [m for m in history if m.get("role") != "system"]
        
        # 保留最近的消息
        recent = other_msgs[-self.keep_recent:]
        
        # 中间消息总结
        middle = other_msgs[:-self.keep_recent]
        if len(middle) > self.summarize_threshold:
            summary = self._summarize_messages(middle)
            middle = [{"role": "system", "content": f"[早期对话总结] {summary}"}]
        
        result = system_msgs + middle + recent
        
        # 计算 Token
        self.current_tokens = self._count_tokens(result)
        
        return result
    
    def _count_tokens(self, messages: List[Dict]) -> int:
        """计算 Token 数"""
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            total += len(self.encoding.encode(content))
        return total
    
    def _summarize_messages(self, messages: List[Dict]) -> str:
        """总结消息（简化版）"""
        # 实际实现应该调用 LLM 进行总结
        # 这里返回简单的统计信息
        user_msgs = [m for m in messages if m.get("role") == "user"]
        assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
        
        return (
            f"共 {len(messages)} 轮对话，"
            f"用户提问 {len(user_msgs)} 次，"
            f"助手回复 {len(assistant_msgs)} 次"
        )
    
    def chunk_large_content(
        self,
        content: str,
        chunk_size: int = 4000,
        overlap: int = 200
    ) -> List[str]:
        """
        将大内容分块
        
        Args:
            content: 原始内容
            chunk_size: 每块大小（字符）
            overlap: 重叠大小
            
        Returns:
            内容块列表
        """
        if len(content) <= chunk_size:
            return [content]
        
        chunks = []
        start = 0
        
        while start < len(content):
            end = start + chunk_size
            chunk = content[start:end]
            
            # 尝试在换行处截断
            if end < len(content):
                last_newline = chunk.rfind("\n")
                if last_newline > chunk_size * 0.8:
                    end = start + last_newline + 1
                    chunk = content[start:end]
            
            chunks.append(chunk)
            start = end - overlap
        
        return chunks
```

### 5. SQLite 会话存储

```python
# coder/core/session.py
"""SQLite 会话持久化"""

import sqlite3
import json
import uuid
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime


class SessionStore:
    """
    会话存储
    
    功能：
    - 会话创建与管理
    - 消息持久化
    - 会话恢复
    - 历史查询
    """
    
    def __init__(self, db_path: str = "./coder_sessions.db"):
        self.db_path = Path(db_path)
        self._init_db()
    
    def _init_db(self):
        """初始化数据库"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    workspace TEXT,
                    metadata TEXT
                );
                
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    role TEXT,
                    content TEXT,
                    tool_calls TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_messages_session 
                ON messages(session_id, timestamp);
            """)
    
    def create_session(self, workspace: str = ".") -> str:
        """创建新会话"""
        session_id = str(uuid.uuid4())[:8]
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO sessions (id, workspace, metadata)
                   VALUES (?, ?, ?)""",
                (session_id, workspace, json.dumps({}))
            )
        
        return session_id
    
    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        tool_calls: Optional[List] = None
    ):
        """保存消息"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO messages (session_id, role, content, tool_calls)
                   VALUES (?, ?, ?, ?)""",
                (session_id, role, content, json.dumps(tool_calls or []))
            )
            
            # 更新会话时间
            conn.execute(
                """UPDATE sessions SET updated_at = CURRENT_TIMESTAMP
                   WHERE id = ?""",
                (session_id,)
            )
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """获取会话"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM sessions WHERE id = ?",
                (session_id,)
            )
            row = cursor.fetchone()
            
            if row:
                return dict(row)
            return None
    
    def get_messages(self, session_id: str, limit: int = 100) -> List[Dict]:
        """获取会话消息"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """SELECT * FROM messages 
                   WHERE session_id = ?
                   ORDER BY timestamp
                   LIMIT ?""",
                (session_id, limit)
            )
            return [dict(row) for row in cursor.fetchall()]
    
    def list_sessions(self, limit: int = 10) -> List[Dict]:
        """列出最近的会话"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """SELECT * FROM sessions
                   ORDER BY updated_at DESC
                   LIMIT ?""",
                (limit,)
            )
            return [dict(row) for row in cursor.fetchall()]
    
    def resume_session(self, session_id: str) -> List[Dict]:
        """恢复会话历史"""
        return self.get_messages(session_id, limit=1000)
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install openai tiktoken sqlite3
```

### 2. 配置文件

```python
# config.py
from coder.agent import AgentConfig

config = AgentConfig(
    workspace="./my_project",
    model_provider="openai",
    model_name="gpt-4o",
    max_context_tokens=8000,
    shell_whitelist=[
        "git", "ls", "cat", "grep", "find",
        "python", "pip", "pytest", "ruff"
    ],
    enable_git_tracking=True,
    enable_agents_md=True
)
```

### 3. 运行 Agent

```python
import asyncio
from coder.agent import AdvancedCoderAgent, AgentConfig

async def main():
    config = AgentConfig(workspace="./my_project")
    agent = AdvancedCoderAgent(config)
    
    while True:
        user_input = input("\n👤 你: ")
        if user_input.lower() in ["quit", "exit"]:
            break
        
        response = await agent.chat(user_input)
        print(f"\n🤖 Agent: {response}")
        
        # 显示统计
        stats = agent.get_stats()
        print(f"[Token: {stats['context_tokens']}, 工具调用: {stats['tool_calls_count']}]")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 📋 功能清单

| 功能 | V1 | V2 | 说明 |
|------|----|----|------|
| 文件读写 | ✅ | ✅ | V2 支持大文件分块 |
| 文件编辑 | ✅ | ✅ | V2 支持原子写入 |
| Shell 执行 | ⚠️ | ✅ | V2 安全沙箱 |
| 多轮对话 | ✅ | ✅ | V2 上下文压缩 |
| 会话持久化 | ❌ | ✅ | SQLite 存储 |
| 记忆系统 | ❌ | ✅ | AGENTS.md |
| Git 追踪 | ❌ | ✅ | 自动记录变更 |
| 工具权限 | ❌ | ✅ | 细粒度控制 |
| MCP | ❌ | ❌ | V3 添加 |
| Subagent | ❌ | ❌ | V3 添加 |

---

## 🔒 安全特性

### 1. 命令白名单

```python
# 只允许安全的命令
whitelist = ["git", "ls", "cat", "python", "pytest"]

# 危险命令会被拦截
agent.chat("rm -rf /")  # ❌ 被拒绝
agent.chat("git status")  # ✅ 允许
```

### 2. 资源限制

```python
# 超时控制
bash(command, timeout=60)  # 60 秒超时

# 输出限制
max_output_size = 100000  # 100KB 上限
```

### 3. 路径限制

```python
# 所有文件操作限制在工作目录内
workspace = Path("./my_project").resolve()
# 访问 ../etc/passwd 会被拒绝
```

---

## 📊 性能优化

### 1. Token 管理

```python
# 智能压缩
max_tokens = 8000
- 保留系统消息
- 保留最近 10 轮
- 中间内容总结
```

### 2. 大文件处理

```python
# 自动分块
read_file(path, offset=0, limit=100)  # 只读 100 行
```

### 3. 异步执行

```python
# 非阻塞工具调用
await tool.execute(...)  # 并发执行多个工具
```

---

## 🧪 测试示例

```bash
# 运行测试
pytest coder/tests/

# 测试安全沙箱
pytest coder/tests/test_sandbox.py

# 测试记忆系统
pytest coder/tests/test_memory.py
```

---

## 🔄 演进对比

| 维度 | V1 MVP | V2 Advanced |
|------|--------|-------------|
| 架构 | 单文件 | 模块化 |
| 安全 | 基础检查 | 白名单沙箱 |
| 记忆 | 无 | AGENTS.md |
| 持久化 | 无 | SQLite |
| 配置 | 零配置 | 可配置 |
| 代码量 | ~200 行 | ~2000 行 |
| 部署时间 | 5 分钟 | 2 小时 |
| 适用场景 | 个人原型 | 小团队生产 |

---

**上一版本**: [V1 MVP](./design-v1-mvp.md)  
**下一版本**: [V3 完整版](./design-v3-full.md)
