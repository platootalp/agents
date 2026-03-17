"""
Skill系统 - 模块化技能管理Agent

架构设计:
=========

分层架构 (Clean Architecture):

    Domain层 (Skills)
        ↓ 被Repository依赖
    Repository层 (SkillsRepository)
        ↓ 被ToolSet依赖
    Tool层 (SkillToolSet)
        ↓ 被Agent依赖
    Agent层 (SkillsUseAgent)

组件职责:
    1. Skills (Domain)
       - 纯数据实体，包含name, description, base_path, metadata
       - 提供 get_summary() 方法生成描述
       - 无业务逻辑，仅数据结构

    2. SkillsRepository (Repository)
       - 纯数据存储和检索（CRUD）
       - 管理Skill对象的生命周期
       - 从文件系统加载Skill
       - 无业务逻辑，仅数据访问

    3. SkillToolSet (Tool层)
       - 所有Skill操作的集合
       - 将业务操作封装为工具函数
       - 供LLM调用的工具实现
       - 包含：load_skill, query_skill, list_skills等

    4. SkillsUseAgent (Agent层)
       - 薄代理层，只负责LLM交互
       - 继承 ToolUseAgent 复用对话循环
       - 构建包含可用Skill列表的系统提示

共享组件使用:
    - MessageBuilder: 构建system/user/assistant/tool消息
    - ToolExecutorMixin: 提供工具执行和格式化功能
    - ToolUseAgent: 提供对话循环和Agent基础功能

设计模式:
    - 分层架构: Domain → Repository → Tool → Agent
    - 依赖倒置: 高层不依赖低层，都依赖抽象
    - 单一职责: 每个类只做一件事
    - 组合优于继承: SkillUseAgent组合SkillToolSet

工作流程:
    User Input → SkillsUseAgent.invoke()
                      ↓
              _build_system_prompt() (包含可用skill列表)
                      ↓
              LLM 生成 Tool Call
                      ↓
              SkillToolSet.execute_tool() (调用具体skill工具)
                      ↓
              SkillRepository.get() / 文件系统操作
                      ↓
              返回结果 → LLM 生成最终回答

扩展指南:
    添加新Skill:
        1. 在skills/目录下创建skill目录
        2. 添加SKILL.md和references/
        3. SkillsUseAgent自动发现并加载

    添加新工具:
        1. 在SkillToolSet中添加新方法
        2. 方法自动注册为可用工具
        3. 更新系统提示描述新工具
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Callable

from dotenv import load_dotenv

# 新工具系统导入
try:
    from apps.engineer.learn.coder.core.tools.base import BaseTool, Tool, ToolResult, tool
    from apps.engineer.learn.coder.core.tools.manager import ToolManager
except ImportError:
    from learn.coder.core.tools.base import BaseTool, Tool, ToolResult, tool
    from learn.coder.core.tools.manager import ToolManager

# Agent基础导入
try:
    from apps.engineer.learn.coder.core.model import Model
    from apps.engineer.learn.coder.agents.tool_use_agent import ToolUseAgent
    from apps.engineer.learn.coder.core.utils import MessageBuilder
except ImportError:
    from learn.coder.core.model import Model
    from learn.coder.agents.tool_use_agent import ToolUseAgent
    from learn.coder.core.utils import MessageBuilder


# ============================================================================
# Domain: 核心数据模型
# ============================================================================


@dataclass
class Skills:
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


class SkillsRepository:
    """Skill仓库 - 只负责Skill对象的存储和检索"""

    def __init__(self):
        self._skills: Dict[str, Skills] = {}
        self._loaded: set = set()  # 跟踪已加载的skill

    def register(self, skill: Skills) -> None:
        """注册一个skill"""
        self._skills[skill.name] = skill

    def get(self, name: str) -> Optional[Skills]:
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
                Skills(
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
# ToolSet: 工具集合（使用新工具系统 - @tool装饰器）
# ============================================================================


class SkillsToolSet:
    """
    Skill工具集合

    职责：
    1. 提供所有skill相关的工具操作
    2. 管理工具定义和参数解析
    3. 协调Repository和文件系统操作
    """

    MAX_FILE_SIZE = 1024 * 1024  # 1MB
    MAX_GREP_RESULTS = 50

    def __init__(self, repository: SkillsRepository, base_path: Path):
        self.repository = repository
        self.base_path = Path(base_path)

    def get_tools(self) -> List[BaseTool]:
        """获取所有工具（使用新工具系统）"""
        return [
            self._create_load_skill_main_tool(),
            self._create_list_skill_files_tool(),
            self._create_read_skill_file_tool(),
            self._create_grep_skill_content_tool(),
            self._create_find_skill_files_by_extension_tool(),
        ]

    def _create_load_skill_main_tool(self) -> BaseTool:
        """创建加载skill主文件的工具"""

        @tool(
            name="load_skill_main",
            description="Load the main SKILL.md file from a skill. Use this first when you want to understand what a skill does.",
        )
        def load_skill_main(skill_name: str) -> str:
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

        return load_skill_main

    def _create_list_skill_files_tool(self) -> BaseTool:
        """创建列出skill文件的工具"""

        @tool(
            name="list_skill_files",
            description="List all files in a skill directory. Use this to discover what resources are available in a skill.",
        )
        def list_skill_files(skill_name: str, pattern: str = "*") -> str:
            """列出skill目录中的文件"""
            skill_path = self._get_skill_path(skill_name)
            if not skill_path:
                return f"Skill '{skill_name}' not found."

            files = sorted(p for p in skill_path.rglob(pattern) if p.is_file())
            if not files:
                return f"No files matching '{pattern}' in {skill_name}."

            return "\n".join(
                f"{f.relative_to(skill_path)} ({f.stat().st_size} bytes)" for f in files
            )

        return list_skill_files

    def _create_read_skill_file_tool(self) -> BaseTool:
        """创建读取skill文件的工具"""

        @tool(
            name="read_skill_file",
            description="Read the content of a specific file from a skill. Use this after listing files to get detailed content.",
        )
        def read_skill_file(skill_name: str, file_path: str) -> str:
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

        return read_skill_file

    def _create_grep_skill_content_tool(self) -> BaseTool:
        """创建搜索skill内容的工具"""

        @tool(
            name="grep_skill_content",
            description="Search for text within skill files. Use this to find specific information across all files in a skill.",
        )
        def grep_skill_content(skill_name: str, query: str, file_pattern: str = "*") -> str:
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

        return grep_skill_content

    def _create_find_skill_files_by_extension_tool(self) -> BaseTool:
        """创建按扩展名查找文件的工具"""

        @tool(
            name="find_skill_files_by_extension",
            description="Find all files with a specific extension in a skill. Use this to locate scripts, templates, or docs.",
        )
        def find_skill_files_by_extension(skill_name: str, extension: str) -> str:
            """按扩展名查找文件"""
            ext = extension if extension.startswith(".") else f".{extension}"

            skill_path = self._get_skill_path(skill_name)
            if not skill_path:
                return f"Skill '{skill_name}' not found."

            files = sorted(p for p in skill_path.rglob(f"*{ext}") if p.is_file())
            if not files:
                return f"No {ext} files found in {skill_name}."

            return "\n".join(str(f.relative_to(skill_path)) for f in files)

        return find_skill_files_by_extension

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
            metadata = SkillsRepository._parse_frontmatter(content)
            if metadata.get("name") == skill_name:
                return dir_path

        return None

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


class SkillsUseAgent(ToolUseAgent):
    """
    Skill使用代理 - 薄代理层，负责LLM交互和Skill管理

    架构位置:
        继承 ToolUseAgent → 复用对话循环和工具执行
        组合 SkillToolSet → 委托所有工具操作
        使用 SkillsRepository → 管理Skill数据存储

    核心职责:
        1. LLM交互: 构建系统提示、处理对话、生成回答
        2. Skill发现: 从文件系统自动加载和管理Skill
        3. 工具协调: 将LLM的工具调用委托给SkillToolSet执行

    组件协作:
        SkillsUseAgent (Agent层)
            ↓ 组合
        SkillToolSet (Tool层)
            ↓ 依赖
        SkillsRepository (Repository层)
            ↓ 管理
        Skills (Domain层)

    共享组件:
        - ToolUseAgent: 提供 invoke/stream 接口和对话循环
        - ToolExecutorMixin: 提供工具执行和参数格式化
        - MessageBuilder: 构建包含Skill列表的系统提示

    关键方法:
        __init__(): 初始化repository和toolset，加载skills
        _build_system_prompt(): 构建包含可用Skill信息的系统提示
        discover_skills(): 从目录发现和加载新Skill
        get_loaded_skills(): 获取已加载的Skill列表

    工作流程:
        1. 初始化: 创建repository → 创建toolset → 从目录加载skills
        2. 调用: invoke() → _build_system_prompt() → LLM → 工具调用
        3. 执行: SkillToolSet.execute_tool() → 具体操作Skill
        4. 返回: 工具结果 → LLM → 最终回答

    扩展方式:
        - 添加新Skill: 在skills/目录创建新目录，自动发现
        - 自定义行为: 重写 _build_system_prompt() 修改系统提示
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        model: Optional[Model] = None,
        tools: Optional[List[BaseTool]] = None,
        skills_dir: Optional[Path] = None,
        skills_repository: Optional[SkillsRepository] = None,
        max_steps: int = 10,
    ):
        # 初始化repository和toolset（在父类之前）
        self.repository = skills_repository or SkillsRepository()
        self.toolset: Optional[SkillsToolSet] = None

        # 设置默认skill目录
        if skills_dir is None:
            skills_dir = Path(__file__).parent / "skills"

        # 先设置好skill工具
        skill_tools = []
        if skills_dir.exists():
            count = self._setup_skills(skills_dir)
            print(f"Loaded {count} skills from {skills_dir}\n")
            if self.toolset:
                skill_tools = self.toolset.get_tools()

        # 合并用户提供的工具和skill工具
        all_tools = list(tools or []) + skill_tools

        # 父类初始化
        super().__init__(name, description, model, all_tools, max_steps)

    def _setup_skills(self, base_path: Union[str, Path]) -> int:
        """
        设置skill目录

        # 1. 从目录加载skill到repository
        # 2. 创建toolset
        # 3. 返回加载的skill数量
        """
        base_path = Path(base_path)

        # 加载skills到repository
        count = self.repository.load_from_directory(base_path)

        # 创建toolset
        self.toolset = SkillsToolSet(self.repository, base_path)

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


# ============================================================================
# Demo
# ============================================================================


def demo():
    load_dotenv()

    # 创建agent
    agent = SkillsUseAgent(
        name="SkillAgent",
        model=Model(),
        max_steps=20,  # 增加步数限制
    )

    # 运行示例
    agent.stream("生成一个agent介绍PPT")


if __name__ == "__main__":
    demo()
