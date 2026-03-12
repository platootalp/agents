"""
Skill系统

架构原则：
1. SkillRepository: 纯数据存储（CRUD）
2. SkillToolSet: 所有工具操作的集合
3. SkillUseAgent: 薄代理层，只负责LLM交互

核心流程：
# 1. 用户输入 -> SkillUseAgent.invoke()
# 2. Agent构建system_prompt（包含可用skill列表）
# 3. LLM决定调用工具 -> SkillToolSet.execute_tool()
# 4. 工具操作文件系统 -> 返回结果
# 5. LLM生成最终回答
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Callable

from dotenv import load_dotenv

from apps.engineer.learn.agent.core.model import Model
from apps.engineer.learn.agent.core.tool import Tool
from apps.engineer.learn.agent.tool_use_agent import ToolUseAgent


# ============================================================================
# Domain: 核心数据模型
# ============================================================================


@dataclass
class Skill:
    """Skill实体 - 纯数据"""

    name: str
    description: str
    base_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_summary(self) -> str:
        return f"- **{self.name}**: {self.description}"


# ============================================================================
# Repository: 数据存储层（纯CRUD，无业务逻辑）
# ============================================================================


class SkillRepository:
    """Skill仓库 - 只负责Skill对象的存储和检索"""

    def __init__(self):
        self._skills: Dict[str, Skill] = {}
        self._loaded: set = set()  # 跟踪已加载的skill

    def register(self, skill: Skill) -> None:
        """注册一个skill"""
        self._skills[skill.name] = skill

    def get(self, name: str) -> Optional[Skill]:
        """获取skill"""
        return self._skills.get(name)

    def list_all(self) -> List[str]:
        """列出所有skill名称"""
        return list(self._skills.keys())

    def mark_loaded(self, name: str) -> None:
        """标记skill为已加载"""
        self._loaded.add(name)

    def is_loaded(self, name: str) -> bool:
        """检查skill是否已加载"""
        return name in self._loaded

    def load_from_directory(self, directory: Union[str, Path]) -> int:
        """从目录加载所有skill"""
        directory = Path(directory)
        if not directory.exists():
            return 0

        for skill_dir in directory.iterdir():
            if not skill_dir.is_dir():
                continue

            skill_file = skill_dir / "SKILL.md"
            if not skill_file.exists():
                continue

            content = skill_file.read_text(encoding="utf-8")
            metadata = self._parse_frontmatter(content)

            self.register(
                Skill(
                    name=metadata.get("name", skill_dir.name),
                    description=metadata.get("description", ""),
                    base_path=str(skill_dir),
                    metadata=metadata,
                )
            )

        return len(self._skills)

    @staticmethod
    def _parse_frontmatter(content: str) -> Dict[str, Any]:
        """解析markdown前置元数据"""
        if not content.startswith("---"):
            return {}

        parts = content.split("---", 2)
        if len(parts) < 3:
            return {}

        metadata = {}
        for line in parts[1].strip().split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                metadata[key.strip()] = value.strip()

        return metadata


# ============================================================================
# ToolSet: 工具集合（所有文件操作工具）
# ============================================================================


class SkillToolSet:
    """
    Skill工具集合

    职责：
    1. 提供所有skill相关的工具操作
    2. 管理工具定义和参数解析
    3. 协调Repository和文件系统操作
    """

    MAX_FILE_SIZE = 1024 * 1024  # 1MB
    MAX_GREP_RESULTS = 50

    # 工具定义元数据
    TOOL_DEFINITIONS = [
        {
            "name": "load_skill_main",
            "description": "Load the main SKILL.md file from a skill. Use this first when you want to understand what a skill does.",
            "params": ["skill_name"],
        },
        {
            "name": "list_skill_files",
            "description": "List all files in a skill directory. Use this to discover what resources are available in a skill.",
            "params": ["skill_name", "pattern"],
        },
        {
            "name": "read_skill_file",
            "description": "Read the content of a specific file from a skill. Use this after listing files to get detailed content.",
            "params": ["skill_name", "file_path"],
        },
        {
            "name": "grep_skill_content",
            "description": "Search for text within skill files. Use this to find specific information across all files in a skill.",
            "params": ["skill_name", "query", "file_pattern"],
        },
        {
            "name": "find_skill_files_by_extension",
            "description": "Find all files with a specific extension in a skill. Use this to locate scripts, templates, or docs.",
            "params": ["skill_name", "extension"],
        },
    ]

    def __init__(self, repository: SkillRepository, base_path: Path):
        self.repository = repository
        self.base_path = Path(base_path)

    def get_tool_definitions(self) -> List[Tool]:
        """获取所有工具定义（供Agent使用）"""
        return [
            Tool(
                name=defn["name"],
                description=defn["description"],
                func=self._create_tool_handler(defn["name"]),
                parameters={
                    "type": "object",
                    "properties": {
                        param: {"type": "string", "description": f"Parameter: {param}"}
                        for param in defn["params"]
                    },
                    "required": defn["params"],
                },
            )
            for defn in self.TOOL_DEFINITIONS
        ]

    def execute_tool(self, tool_name: str, args_json: str) -> str:
        """执行指定工具"""
        args = self._parse_args(args_json)

        handlers: Dict[str, Callable] = {
            "load_skill_main": self._handle_load_skill_main,
            "list_skill_files": self._handle_list_files,
            "read_skill_file": self._handle_read_file,
            "grep_skill_content": self._handle_grep_content,
            "find_skill_files_by_extension": self._handle_find_by_extension,
        }

        handler = handlers.get(tool_name)
        if not handler:
            return f"Unknown tool: {tool_name}"

        try:
            return handler(**args)
        except Exception as e:
            return f"Error executing {tool_name}: {e}"

    def _create_tool_handler(self, tool_name: str) -> Callable[[str], str]:
        """创建工具处理函数（适配Tool接口）"""
        return lambda args_json: self.execute_tool(tool_name, args_json)

    @staticmethod
    def _parse_args(args_json: str) -> Dict[str, Any]:
        """解析工具参数（统一入口，避免重复）"""
        if not args_json:
            return {}
        try:
            return json.loads(args_json) if args_json.startswith("{") else {"skill_name": args_json}
        except json.JSONDecodeError:
            return {"skill_name": args_json}

    def _get_skill_path(self, skill_name: str) -> Optional[Path]:
        """获取skill目录路径（优先从repository查，其次按名称查找）"""
        # 先从repository查
        skill = self.repository.get(skill_name)
        if skill:
            return Path(skill.base_path)

        # 按目录名查找
        skill_dir = self.base_path / skill_name
        if skill_dir.exists():
            return skill_dir

        # 遍历查找metadata匹配的
        for dir_path in self.base_path.iterdir():
            if not dir_path.is_dir():
                continue

            skill_file = dir_path / "SKILL.md"
            if not skill_file.exists():
                continue

            content = skill_file.read_text(encoding="utf-8")
            metadata = SkillRepository._parse_frontmatter(content)
            if metadata.get("name") == skill_name:
                return dir_path

        return None

    # ------------------ 具体工具实现 ------------------

    def _handle_load_skill_main(self, skill_name: str, **_) -> str:
        """加载skill的主SKILL.md文件"""
        skill = self.repository.get(skill_name)
        if not skill:
            available = ", ".join(self.repository.list_all())
            return f"Skill '{skill_name}' not found. Available: {available}"

        skill_file = Path(skill.base_path) / "SKILL.md"
        if not skill_file.exists():
            return f"SKILL.md not found for skill '{skill_name}'."

        content = skill_file.read_text(encoding="utf-8")
        _, body = self._split_frontmatter(content)
        self.repository.mark_loaded(skill_name)
        return f"=== Skill: {skill_name} ===\n\n{body}\n\n=== End of Skill ==="

    def _handle_list_files(self, skill_name: str, pattern: str = "*", **_) -> str:
        """列出skill目录中的文件"""
        skill_path = self._get_skill_path(skill_name)
        if not skill_path:
            return f"Skill '{skill_name}' not found."

        files = sorted(p for p in skill_path.rglob(pattern) if p.is_file())
        if not files:
            return f"No files matching '{pattern}' in {skill_name}."

        return "\n".join(f"{f.relative_to(skill_path)} ({f.stat().st_size} bytes)" for f in files)

    def _handle_read_file(self, skill_name: str, file_path: str, **_) -> str:
        """读取skill中的指定文件"""
        skill_path = self._get_skill_path(skill_name)
        if not skill_path:
            return f"Skill '{skill_name}' not found."

        target_path = (skill_path / file_path).resolve()
        skill_resolved = skill_path.resolve()

        # 安全检查：防止路径遍历
        if not str(target_path).startswith(str(skill_resolved)):
            return "Access denied: path outside skill directory."

        if not target_path.exists():
            return f"File '{file_path}' not found in skill '{skill_name}'."

        if not target_path.is_file():
            return f"'{file_path}' is not a file."

        content = target_path.read_text(encoding="utf-8")
        return f"=== {file_path} ===\n\n{content}\n\n=== End of {file_path} ==="

    def _handle_grep_content(
            self, skill_name: str, query: str, file_pattern: str = "*", **_
    ) -> str:
        """在skill文件中搜索内容"""
        skill_path = self._get_skill_path(skill_name)
        if not skill_path:
            return f"Skill '{skill_name}' not found."

        matches = []
        for f in skill_path.rglob(file_pattern):
            if not f.is_file() or f.stat().st_size > self.MAX_FILE_SIZE:
                continue

            try:
                content = f.read_text(encoding="utf-8")
                for i, line in enumerate(content.split("\n"), 1):
                    if query.lower() in line.lower():
                        rel_path = f.relative_to(skill_path)
                        matches.append(f"{rel_path}:{i}: {line.strip()}")
            except Exception:
                continue

        if matches:
            return "\n".join(matches[: self.MAX_GREP_RESULTS])
        return f"No matches found for '{query}' in {skill_name}."

    def _handle_find_by_extension(self, skill_name: str, extension: str, **_) -> str:
        """按扩展名查找文件"""
        ext = extension if extension.startswith(".") else f".{extension}"

        skill_path = self._get_skill_path(skill_name)
        if not skill_path:
            return f"Skill '{skill_name}' not found."

        files = sorted(p for p in skill_path.rglob(f"*{ext}") if p.is_file())
        if not files:
            return f"No {ext} files found in {skill_name}."

        return "\n".join(str(f.relative_to(skill_path)) for f in files)

    @staticmethod
    def _split_frontmatter(content: str) -> tuple[Dict[str, Any], str]:
        """分离frontmatter和正文"""
        if not content.startswith("---"):
            return {}, content

        parts = content.split("---", 2)
        if len(parts) < 3:
            return {}, content

        metadata = {}
        for line in parts[1].strip().split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                metadata[key.strip()] = value.strip()

        return metadata, parts[2].strip()


# ============================================================================
# Agent: 薄代理层（只负责LLM交互）
# ============================================================================


class SkillUseAgent(ToolUseAgent):
    """
    Skill使用代理

    职责：
    1. 构建system_prompt（包含可用skill信息）
    2. 协调LLM调用和工具执行
    3. 维护对话历史

    注：所有工具逻辑委托给SkillToolSet
    """

    def __init__(
            self,
            name: str,
            description: str = "",
            model: Optional[Model] = None,
            tools: Optional[List[Tool]] = None,
            repository: Optional[SkillRepository] = None,
            max_steps: int = 10,
    ):
        # 初始化repository和toolset
        self.repository = repository or SkillRepository()
        self.toolset: Optional[SkillToolSet] = None

        # 父类初始化（此时还没有toolset，稍后添加工具）
        super().__init__(name, description, model, tools or [], max_steps)

    def setup_skills(self, base_path: Union[str, Path]) -> int:
        """
        设置skill目录

        # 1. 从目录加载skill到repository
        # 2. 创建toolset
        # 3. 将skill工具添加到agent的工具列表
        """
        base_path = Path(base_path)

        # 加载skills到repository
        count = self.repository.load_from_directory(base_path)

        # 创建toolset并获取工具
        self.toolset = SkillToolSet(self.repository, base_path)
        skill_tools = self.toolset.get_tool_definitions()

        # 添加到agent的工具列表
        self.tools.extend(skill_tools)

        return count

    def _build_system_prompt(self) -> str:
        """构建system prompt"""
        base_prompt = (
            "You are a helpful assistant with access to specialized skills. "
            "Skills are directories containing documentation and resources. "
            "Use the available tools to explore and read skill files as needed."
        )

        # 添加可用skill列表
        if self.repository.list_all():
            skills_list = [
                self.repository.get(name).get_summary() for name in self.repository.list_all()
            ]
            base_prompt += f"\n\n## Available Skills\n\n" + "\n".join(skills_list)
            base_prompt += (
                "\n\nTo explore a skill's contents, you can use these tools:\n"
                "- load_skill_main: Load the main SKILL.md\n"
                "- list_skill_files: List all files in a skill\n"
                "- read_skill_file: Read a specific file\n"
                "- grep_skill_content: Search for text\n"
                "- find_skill_files_by_extension: Find files by type"
            )

        return base_prompt

    def invoke(self, input: str) -> str:
        """
        执行用户请求

        流程：
        # 1. 检查model配置
        # 2. 初始化消息历史
        # 3. 循环直到完成或达到最大步数
        #    a. 调用LLM生成回复
        #    b. 如果有工具调用，执行工具并继续
        #    c. 如果有回复内容，返回结果
        """
        if not self.model:
            return "No model configured."

        self.message_history = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": input},
        ]

        for step in range(self.max_steps):
            # 获取工具定义（OpenAI格式）
            openai_tools = self._get_openai_tools()

            # 调用LLM
            response = self.model.generate(self.message_history, tools=openai_tools)
            message = response.choices[0].message

            # 添加到历史
            assistant_msg: Dict[str, Any] = {"role": "assistant", "content": message.content or ""}
            if message.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in message.tool_calls
                ]
            self.message_history.append(assistant_msg)

            # 如果没有工具调用，返回回复内容
            if not message.tool_calls:
                if message.content:
                    return message.content.strip()
                continue

            # 执行工具调用
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                args = tool_call.function.arguments

                # 使用toolset执行工具
                if self.toolset:
                    tool_result = self.toolset.execute_tool(tool_name, args)
                else:
                    tool_result = "Skill toolset not initialized."

                self.message_history.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(tool_result),
                    }
                )

        return "Reached maximum steps without a final answer."

    def stream(self, input: str) -> str:
        pass


# ============================================================================
# Demo
# ============================================================================


def search_tool(query: str) -> str:
    return f"Search: {query}" if query.strip() else "Empty query"


def calculator_tool(expr: str) -> str:
    import re

    expr = expr.strip()
    if not expr:
        return "Empty expression"
    if not re.match(r"^[0-9+\-*/().\s]+$", expr):
        return "Invalid characters"
    try:
        return str(eval(expr, {"__builtins__": {}}))
    except Exception as e:
        return f"Error: {e}"


def demo():
    load_dotenv()

    # 创建agent
    agent = SkillUseAgent(
        name="SkillAgent",
        model=Model(),
        tools=[
            Tool(name="search", description="Web search", func=search_tool),
            Tool(name="calculator", description="Calculator", func=calculator_tool),
        ],
        max_steps=10,  # 增加步数限制
    )

    # 设置skill目录
    skills_dir = Path(__file__).parent / "skills"
    if skills_dir.exists():
        count = agent.setup_skills(skills_dir)
        print(f"Loaded {count} skills from {skills_dir}\n")

    # 运行示例
    result = agent.invoke("生成一个agent介绍PPT")
    print(f"Result:\n{result}")


if __name__ == "__main__":
    demo()
