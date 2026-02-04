#!/usr/bin/env python3
"""
GitHub Stars Agent - 基于 LangGraph 的智能索引生成器

使用 LLM 智能分析和分类 GitHub starred 仓库，生成高质量的索引文档。

功能：
- 自动获取：直接从 GitHub API 获取 starred 仓库数据
- 智能分类：使用 LLM 分析仓库的技术栈、用途、领域
- 自动总结：为每个分类生成描述性说明
- 智能排序：根据相关性和重要性排序
- 推荐系统：识别关键项目和学习路径
"""

import json
import os
import sys
from datetime import datetime
from typing import Annotated, Dict, List, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.util import get_qwen_model

try:
    import requests
except ImportError:
    print("错误: 需要安装 requests 库")
    print("请运行: pip install requests")
    sys.exit(1)


# ============================================================================
# GitHub API 数据获取
# ============================================================================


class GitHubStarsFetcher:
    """GitHub Stars 信息爬取器"""

    def __init__(self, token: str):
        """
        初始化爬取器

        Args:
            token: GitHub Personal Access Token
        """
        self.token = token
        self.base_url = "https://api.github.com"
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def get_authenticated_user(self) -> Dict:
        """获取当前认证用户信息"""
        url = f"{self.base_url}/user"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()

    def fetch_starred_repos(
            self, username: Optional[str] = None, per_page: int = 100
    ) -> List[Dict]:
        """
        获取用户的所有 starred 仓库

        Args:
            username: GitHub 用户名（如果为 None，则获取认证用户）
            per_page: 每页返回的仓库数量（最大 100）

        Returns:
            包含所有 starred 仓库信息的列表
        """
        if username:
            url = f"{self.base_url}/users/{username}/starred"
        else:
            url = f"{self.base_url}/user/starred"

        all_repos = []
        page = 1

        print(f"🔄 正在获取 starred 仓库信息...")

        while True:
            params = {"per_page": per_page, "page": page}
            response = self.session.get(url, params=params)
            response.raise_for_status()

            repos = response.json()
            if not repos:
                break

            all_repos.extend(repos)
            print(f"  已获取 {len(all_repos)} 个仓库...", end="\r")

            # 检查是否还有下一页
            link_header = response.headers.get("Link", "")
            if "rel=\"next\"" not in link_header:
                break

            page += 1

        print(f"\n✓ 共获取到 {len(all_repos)} 个 starred 仓库")
        return all_repos

    def extract_repo_info(self, repo: Dict) -> Dict:
        """
        从仓库数据中提取需要的信息

        Args:
            repo: GitHub API 返回的仓库数据

        Returns:
            提取后的仓库信息
        """
        return {
            "name": repo["name"],
            "full_name": repo["full_name"],
            "owner": repo["owner"]["login"],
            "description": repo["description"] or "无描述",
            "url": repo["html_url"],
            "homepage": repo["homepage"],
            "stars": repo["stargazers_count"],
            "forks": repo["forks_count"],
            "language": repo["language"] or "-",
            "topics": repo.get("topics", []),
            "license": repo["license"]["name"] if repo["license"] else None,
            "created_at": repo["created_at"],
            "updated_at": repo["updated_at"],
            "pushed_at": repo["pushed_at"],
            "is_fork": repo["fork"],
            "is_archived": repo["archived"],
            "open_issues": repo["open_issues_count"],
        }


# ============================================================================
# 分类节点类（面向对象设计）
# ============================================================================


class CategoryNode:
    """分类节点类 - 支持任意层级的树形结构"""

    def __init__(self, name: str, description: str = "", repos: List[int] = None):
        """
        初始化分类节点
        
        Args:
            name: 分类名称
            description: 分类描述
            repos: 仓库索引列表（只在叶子节点有值）
        """
        self.name = name
        self.description = description
        self.repos = repos or []
        self.children: List['CategoryNode'] = []

    def add_child(self, child: 'CategoryNode') -> 'CategoryNode':
        """添加子节点"""
        self.children.append(child)
        return child

    def is_leaf(self) -> bool:
        """判断是否为叶子节点"""
        return len(self.children) == 0

    def get_all_repos(self) -> List[int]:
        """递归获取本节点及所有子节点的仓库"""
        all_repos = list(self.repos)
        for child in self.children:
            all_repos.extend(child.get_all_repos())
        return all_repos

    def get_path(self, parent_path: List[str] = None) -> List[str]:
        """获取从根到当前节点的路径"""
        if parent_path is None:
            parent_path = []
        return parent_path + [self.name]

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        result = {
            "name": self.name,
            "description": self.description,
        }
        if self.repos:
            result["repos"] = self.repos
        if self.children:
            result["children"] = [child.to_dict() for child in self.children]
        return result

    @classmethod
    def from_dict(cls, data: Dict) -> 'CategoryNode':
        """从字典创建节点"""
        node = cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            repos=data.get("repos", [])
        )
        for child_data in data.get("children", []):
            node.add_child(cls.from_dict(child_data))
        return node

    def traverse(self, callback, depth: int = 0, parent_path: str = ""):
        """递归遍历树"""
        full_path = f"{parent_path} / {self.name}" if parent_path else self.name
        callback(self, depth, full_path)

        for child in self.children:
            child.traverse(callback, depth + 1, full_path)

    def __repr__(self):
        return f"CategoryNode(name={self.name}, repos={len(self.repos)}, children={len(self.children)})"


# ============================================================================
# State 定义
# ============================================================================


class AgentState(TypedDict):
    """Agent 状态"""
    messages: Annotated[List, add_messages]
    github_token: str  # GitHub Token
    username: Optional[str]  # GitHub 用户名
    min_stars: int  # 最小 stars 过滤
    repositories: List[Dict]  # 原始仓库数据
    categories: Dict[str, List[Dict]]  # 分类后的仓库
    category_descriptions: Dict[str, str]  # 分类描述
    recommendations: List[str]  # 推荐和学习路径
    markdown_output: str  # 生成的 Markdown


# ============================================================================
# Agent 节点
# ============================================================================


class GitHubStarsAgent:
    """GitHub Stars 智能分析 Agent"""

    def __init__(self, llm=None):
        """
        初始化 Agent

        Args:
            llm: LLM 实例，默认使用 Qwen 模型
        """
        self.llm = llm or get_qwen_model()

    def fetch_github_stars(self, state: AgentState) -> AgentState:
        """
        第一步：从 GitHub API 获取 starred 仓库数据
        """
        print("\n" + "=" * 70)
        print("🔄 步骤 1/4: 获取 GitHub Stars 数据")
        print("=" * 70)

        github_token = state.get("github_token")
        username = state.get("username")
        min_stars = state.get("min_stars", 0)

        if not github_token:
            raise ValueError("未提供 GitHub Token，请设置 GITHUB_TOKEN 环境变量")

        try:
            # 初始化 fetcher
            fetcher = GitHubStarsFetcher(github_token)

            # 获取用户信息
            if not username:
                user = fetcher.get_authenticated_user()
                username = user["login"]
                print(f"✓ 认证成功: {username}")

            # 获取 starred 仓库
            raw_repos = fetcher.fetch_starred_repos(username)

            # 提取仓库信息
            repos = [fetcher.extract_repo_info(repo) for repo in raw_repos]

            # 过滤
            if min_stars > 0:
                repos = [r for r in repos if r["stars"] >= min_stars]
                print(f"✓ 过滤后剩余 {len(repos)} 个仓库 (>= {min_stars} stars)")

            state["repositories"] = repos
            state["messages"].append(
                AIMessage(content=f"✓ 成功获取 {len(repos)} 个 GitHub starred 仓库")
            )

        except requests.exceptions.HTTPError as e:
            error_msg = f"GitHub API 请求失败: {e}"
            if e.response.status_code == 401:
                error_msg += " (Token 可能无效或已过期)"
            state["messages"].append(AIMessage(content=f"❌ {error_msg}"))
            raise

        except Exception as e:
            error_msg = f"获取数据失败: {e}"
            state["messages"].append(AIMessage(content=f"❌ {error_msg}"))
            raise

        return state

    def analyze_repositories(self, state: AgentState) -> AgentState:
        """
        分析仓库并智能分类

        使用 LLM 分析每个仓库的技术栈、用途和领域，进行智能分类
        """
        print("\n" + "=" * 70)
        print("🤖 步骤 2/4: AI 智能分析与分类")
        print("=" * 70)

        repos = state["repositories"]
        print(f"📊 准备分析 {len(repos)} 个仓库...")

        # 准备仓库摘要（限制数量避免超出上下文）
        max_repos = 105  # 限制最大分析数量（确保 AI 准确分类）
        analyze_repos = repos[:max_repos] if len(repos) > max_repos else repos

        if len(repos) > max_repos:
            print(f"⚠️  仓库数量较多，将分析前 {max_repos} 个高 star 项目")
            # 按 stars 排序，取前 N 个
            analyze_repos = sorted(repos, key=lambda x: x["stars"], reverse=True)[:max_repos]

        repo_summaries = []
        for i, repo in enumerate(analyze_repos):
            summary = {
                "index": i,
                "name": repo["full_name"],
                "description": repo["description"][:100] if repo["description"] else "无描述",  # 限制描述长度
                "language": repo["language"],
                "topics": repo.get("topics", [])[:3],  # 只取前3个topic
                "stars": repo["stars"],
            }
            repo_summaries.append(summary)

        # 创建更简洁的仓库列表用于提示
        repo_list_str = "\n".join([
            f"{i}. {repo['name']} - {repo['description'][:50]}... (⭐{repo['stars']}, {repo['language']})"
            for i, repo in enumerate(repo_summaries)
        ])

        prompt = f"""分析以下 {len(repo_summaries)} 个 GitHub 仓库，按技术领域进行层级分类。

仓库列表：
{repo_list_str}

要求：
1. 使用灵活的多级分类结构（可以是2-4级）
2. 每个仓库只能分配到一个最终分类（叶子节点）
3. 所有 {len(repo_summaries)} 个仓库都必须被分配
4. 使用中文分类名称

返回JSON格式（递归结构，支持任意层级）：
{{
  "categories": [
    {{
      "name": "AI/机器学习",
      "description": "人工智能相关技术",
      "children": [
        {{
          "name": "LLM",
          "description": "大语言模型",
          "children": [
            {{
              "name": "Agent框架",
              "description": "智能代理开发框架",
              "repos": [0, 3, 5]
            }},
            {{
              "name": "向量数据库",
              "repos": [1, 2]
            }}
          ]
        }},
        {{
          "name": "深度学习",
          "repos": [4, 7, 9]
        }}
      ]
    }},
    {{
      "name": "Web开发",
      "repos": [6, 8, 10]
    }}
  ]
}}

说明：
- name: 节点名称（必需）
- description: 节点描述（可选）
- children: 子节点数组（可嵌套任意层级）
- repos: 仓库索引（只在叶子节点）
- 每个 index 只能出现一次"""

        messages = [
            SystemMessage(content="你是一个专业的技术分类专家。"),
            HumanMessage(content=prompt),
        ]

        print("\n🤔 AI 正在分析仓库并智能分类...")
        print("─" * 60)

        # 使用流式输出
        full_content = ""
        for chunk in self.llm.stream(messages):
            content = chunk.content
            print(content, end="", flush=True)
            full_content += content

        print("\n" + "─" * 60)

        try:
            # 解析 LLM 返回的分类结果
            result = json.loads(full_content)
            categories_list = result.get("categories", [])

            # 使用 CategoryNode 类解析
            root_nodes = []
            for cat_data in categories_list:
                root_nodes.append(CategoryNode.from_dict(cat_data))

            print(f"\n✅ AI 分类完成，共 {len(root_nodes)} 个顶层分类")

            # 构建扁平的分类字典
            categories = {}
            category_descriptions = {}
            used_indices = set()

            def process_node(node: CategoryNode, parent_path: str = ""):
                """递归处理节点（使用类方法）"""
                # 构建完整路径
                full_path = f"{parent_path} / {node.name}" if parent_path else node.name

                # 输出层级结构
                depth = len(parent_path.split(" / ")) if parent_path else 0
                indent = "  " * depth
                prefix = "└─ " if depth > 0 else "📁 "

                if depth == 0:
                    print(f"\n{prefix}{node.name}")
                else:
                    print(f"{indent}{prefix}{node.name}")

                # 保存描述
                if node.description:
                    category_descriptions[full_path] = node.description

                # 如果是叶子节点，处理仓库
                if node.repos:
                    # 去重
                    unique_indices = [idx for idx in node.repos if idx not in used_indices and idx < len(analyze_repos)]

                    if len(unique_indices) != len(node.repos):
                        duplicate_count = len(node.repos) - len(unique_indices)
                        print(f"{indent}   ⚠️  移除 {duplicate_count} 个重复")

                    used_indices.update(unique_indices)

                    if unique_indices:
                        categories[full_path] = [analyze_repos[idx] for idx in unique_indices]
                        print(f"{indent}   ({len(unique_indices)} 个仓库)")

                # 递归处理子节点
                for child in node.children:
                    process_node(child, full_path)

            # 处理所有根节点
            for root in root_nodes:
                process_node(root)

            # 检查是否有仓库未被分类
            all_assigned = len(used_indices)
            if all_assigned == 0:
                print("\n⚠️  警告: AI 未分配任何仓库，使用默认语言分类")
                categories = self._default_categorize(repos)
                category_descriptions = {}
            elif all_assigned < len(analyze_repos):
                unassigned_count = len(analyze_repos) - all_assigned
                print(f"\n⚠️  注意: 有 {unassigned_count} 个仓库未被分类")

                # 将未分类的仓库添加到"其他"分类
                unassigned_indices = set(range(len(analyze_repos))) - used_indices
                if unassigned_indices:
                    categories["其他"] = [analyze_repos[idx] for idx in unassigned_indices]
                    category_descriptions["其他"] = "未明确分类的项目"
                    print(f"   已自动归入「其他」分类")

            state["categories"] = categories
            state["category_descriptions"] = category_descriptions
            state["messages"].append(AIMessage(content=f"✓ 完成智能分类，共 {len(categories)} 个分类"))

        except json.JSONDecodeError as e:
            # 如果 LLM 返回格式有误，使用默认按语言分类
            print(f"\n⚠️  警告: LLM 返回格式错误 ({e})，使用默认分类")
            categories = self._default_categorize(repos)
            state["categories"] = categories
            state["category_descriptions"] = {}
            state["messages"].append(AIMessage(content="使用默认语言分类"))

        return state

    def generate_recommendations(self, state: AgentState) -> AgentState:
        """
        生成学习路径和推荐

        分析仓库组合，提供学习建议和项目推荐
        """
        print("\n" + "=" * 70)
        print("💡 步骤 3/4: 生成学习路径和推荐")
        print("=" * 70)

        categories = state["categories"]
        repos = state["repositories"]
        print(f"📚 基于 {len(categories)} 个分类生成推荐...")

        # 准备分类摘要
        category_summary = {}
        for cat_name, cat_repos in categories.items():
            top_repos = sorted(cat_repos, key=lambda x: x["stars"], reverse=True)[:5]
            category_summary[cat_name] = {
                "count": len(cat_repos),
                "top_projects": [
                    {"name": r["full_name"], "stars": r["stars"]}
                    for r in top_repos
                ],
            }

        prompt = f"""基于用户收藏的 GitHub 仓库，提供学习路径和项目推荐。

            分类概览：
            {json.dumps(category_summary, ensure_ascii=False, indent=2)}
            
            请提供：
            1. 学习路径建议（针对主要技术栈）
            2. 关键项目推荐（标注为"⭐ 必看"）
            3. 技术栈组合建议
            
            返回格式（Markdown）：
            ### 学习路径
            - **路径1**: 描述 → 推荐项目1 → 项目2 → 项目3
            - **路径2**: ...
            
            ### 关键项目
            - ⭐ [项目名称](URL) - 推荐理由
            
            ### 技术栈建议
            - 组合1：...
            - 组合2：...
        """

        messages = [
            SystemMessage(content="你是一个技术学习路径规划专家。"),
            HumanMessage(content=prompt),
        ]

        print("\n💡 AI 正在生成学习路径和推荐...")
        print("─" * 60)

        # 使用流式输出
        full_content = ""
        for chunk in self.llm.stream(messages):
            content = chunk.content
            print(content, end="", flush=True)
            full_content += content

        print("\n" + "─" * 60)

        state["recommendations"] = full_content.split("\n")
        state["messages"].append(AIMessage(content="✓ 生成学习路径和推荐"))

        return state

    def generate_markdown(self, state: AgentState) -> AgentState:
        """
        生成最终的 Markdown 索引文档
        """
        print("\n" + "=" * 70)
        print("📝 步骤 4/4: 生成 Markdown 文档")
        print("=" * 70)
        
        categories = state["categories"]
        category_descriptions = state["category_descriptions"]
        recommendations = state["recommendations"]
        repos = state["repositories"]

        md = []
        
        # 渲染各个部分
        self._render_header(md, repos, categories)
        self._render_toc_section(md, categories)
        
        if recommendations:
            self._render_recommendations_section(md, recommendations)
        
        self._render_categories_section(md, categories, category_descriptions)
        self._render_statistics_section(md, repos)
        self._render_footer(md)

        state["markdown_output"] = "\n".join(md)
        state["messages"].append(AIMessage(content="✓ Markdown 文档生成完成"))

        print("\n✅ 文档生成完成！")
        return state
    
    def _render_header(self, md: List[str], repos: List[Dict], categories: Dict) -> None:
        """渲染文档头部"""
        md.append(f"# 🌟 GitHub Stars 智能索引\n")
        md.append(f"> 📚 AI 驱动的个性化收藏库 | 更新时间：{datetime.now().strftime('%Y-%m-%d')}\n")
        md.append("## 📖 关于\n")
        md.append(f"- **总收藏**: {len(repos)} 个项目")
        md.append(f"- **智能分类**: {len(categories)} 个领域")
        md.append("- **AI 分析**: 由 LangGraph Agent (Qwen) 智能整理\n")
        md.append("---\n")
    
    def _render_toc_section(self, md: List[str], categories: Dict) -> None:
        """渲染目录部分"""
        md.append("## 📋 目录\n")
        
        # 构建目录树
        toc_tree = self._build_toc_tree(categories)
        
        # 递归渲染目录
        self._render_toc_level(toc_tree, 0, md)
        md.append("\n---\n")
    
    def _render_recommendations_section(self, md: List[str], recommendations: List[str]) -> None:
        """渲染推荐部分"""
        md.append("## 💡 AI 推荐\n")
        md.extend(recommendations)
        md.append("\n---\n")
    
    def _render_categories_section(self, md: List[str], categories: Dict, 
                                     category_descriptions: Dict) -> None:
        """渲染分类内容部分"""
        # 构建分类树
        category_tree = self._build_category_tree(categories)
        
        # 递归渲染分类内容
        self._render_category_level(category_tree, 2, md, category_descriptions)
    
    def _render_statistics_section(self, md: List[str], repos: List[Dict]) -> None:
        """渲染统计分析部分"""
        md.append("## 📊 统计分析\n")
        
        # 编程语言分布
        lang_stats = self._calculate_language_stats(repos)
        md.append("### 编程语言分布\n")
        for lang, count in list(lang_stats.items())[:8]:
            percentage = (count / len(repos)) * 100
            md.append(f"- **{lang}**: {count} 个 ({percentage:.1f}%)")

        # Stars 分布
        md.append("\n### Stars 分布\n")
        stars_ranges = self._calculate_stars_ranges(repos)
        for range_name, count in stars_ranges.items():
            if count > 0:
                md.append(f"- {range_name}: {count} 个")

        # Top 10
        md.append("\n### 🏆 Top 10 明星项目\n")
        top_repos = sorted(repos, key=lambda x: x["stars"], reverse=True)[:10]
        for i, repo in enumerate(top_repos, 1):
            stars = self._format_stars(repo["stars"])
            md.append(f"{i}. **{repo['full_name']}** - ⭐ {stars}")
    
    def _render_footer(self, md: List[str]) -> None:
        """渲染文档底部"""
        md.append("\n---\n")
        md.append(f"*📅 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
        md.append(f"*🤖 由 GitHub Stars Agent (Qwen) 智能生成*")

    # ========================================================================
    # 辅助方法 - 树构建
    # ========================================================================
    
    def _build_toc_tree(self, categories: Dict[str, List[Dict]]) -> Dict:
        """构建目录树结构"""
        toc_tree = {}
        for cat_path, cat_repos in categories.items():
            parts = [p.strip() for p in cat_path.split(" / ")]
            current = toc_tree
            for part in parts:
                if part not in current:
                    current[part] = {"_count": 0, "_children": {}, "_name": part}
                current = current[part]["_children"]
            
            # 记录仓库数量
            if parts:
                parent = toc_tree
                for part in parts[:-1]:
                    parent = parent[part]["_children"]
                parent[parts[-1]]["_count"] = len(cat_repos)
        
        return toc_tree
    
    def _build_category_tree(self, categories: Dict[str, List[Dict]]) -> Dict:
        """构建分类内容树结构"""
        category_tree = {}
        for cat_path, cat_repos in categories.items():
            parts = [p.strip() for p in cat_path.split(" / ")]
            
            # 将路径插入树中
            current = category_tree
            for i, part in enumerate(parts):
                if part not in current:
                    current[part] = {"_repos": None, "_children": {}, "_path": " / ".join(parts[:i + 1])}
                current = current[part]["_children"]
            
            # 最后一级保存仓库
            if parts:
                last_part = parts[-1]
                parent = category_tree
                for part in parts[:-1]:
                    parent = parent[part]["_children"]
                parent[last_part]["_repos"] = cat_repos
        
        return category_tree
    
    # ========================================================================
    # 辅助方法 - 递归统计
    # ========================================================================
    
    def _get_all_children(self, tree: Dict) -> List[Dict]:
        """递归获取所有子节点"""
        result = []
        for node in tree.values():
            result.append(node)
            if node.get("_children"):
                result.extend(self._get_all_children(node["_children"]))
        return result
    
    def _get_all_repos_from_tree(self, tree: Dict) -> List[Dict]:
        """递归获取树中所有仓库"""
        total = []
        for node in tree.values():
            if node.get("_repos"):
                total.extend(node["_repos"])
            if node.get("_children"):
                total.extend(self._get_all_repos_from_tree(node["_children"]))
        return total

    def _count_repos_recursive(self, tree: Dict) -> int:
        """递归统计树中所有仓库数量"""
        total = 0
        for node in tree.values():
            if node.get("_repos"):
                total += len(node["_repos"])
            if node.get("_children"):
                total += self._count_repos_recursive(node["_children"])
        return total
    
    # ========================================================================
    # 辅助方法 - 递归渲染
    # ========================================================================
    
    def _render_toc_level(self, tree: Dict, depth: int, md: List[str]) -> None:
        """递归渲染目录（支持任意层级）"""
        if not tree:
            return
        
        # 按数量排序
        sorted_items = sorted(
            tree.items(),
            key=lambda x: x[1]["_count"] + sum(
                child["_count"] for child in self._get_all_children(x[1]["_children"])
            ),
            reverse=True
        )
        
        for name, node in sorted_items:
            count = node["_count"]
            children = node["_children"]
            
            # 计算总数（包括子分类）
            total = count + sum(
                child["_count"] for child in self._get_all_children(children)
            )
            
            # 缩进
            indent = "  " * depth
            
            # 生成目录项（都显示总数）
            if depth == 0:
                # 顶层加粗
                md.append(f"{indent}- **[{name}](#{self._anchor(name)})** ({total}个)")
            else:
                # 非顶层也显示总数
                md.append(f"{indent}- [{name}](#{self._anchor(name)}) ({total}个)")
            
            # 递归子目录
            if children:
                self._render_toc_level(children, depth + 1, md)
    
    def _render_category_level(self, tree: Dict, level: int, md: List[str], 
                                 category_descriptions: Dict) -> None:
        """递归渲染分类树（支持任意层级）"""
        if not tree:
            return
        
        # 按仓库数量排序
        sorted_items = sorted(
            tree.items(),
            key=lambda x: len(x[1]["_repos"]) if x[1]["_repos"] else 
                          len(self._get_all_repos_from_tree(x[1]["_children"])),
            reverse=True
        )
        
        for name, node in sorted_items:
            repos = node["_repos"]
            children = node["_children"]
            full_path = node["_path"]
            
            # 计算仓库数
            direct_count = len(repos) if repos else 0
            children_count = self._count_repos_recursive(children) if children else 0
            total_repos = direct_count + children_count
            
            # 渲染标题和描述
            md.extend(self._render_category_header(
                name, level, full_path, category_descriptions,
                total_repos, bool(children), direct_count
            ))
            
            # 如果有仓库（叶子节点），渲染表格
            if repos:
                md.extend(self._render_repo_table(repos))
            
            # 递归渲染子分类
            if children:
                self._render_category_level(children, level + 1, md, category_descriptions)
        
        # 大类之间的分隔线（仅一级分类后）
        if level == 2:
            md.append("---\n")
    
    # ========================================================================
    # 辅助方法 - 内容渲染
    # ========================================================================
    
    def _render_repo_table(self, repos: List[Dict]) -> List[str]:
        """渲染仓库表格"""
        lines = [
            "| 名称 | 简介 | Stars | 语言 | 链接 |",
            "|------|------|-------|------|------|"
        ]

        for repo in sorted(repos, key=lambda x: x["stars"], reverse=True):
            name_col = repo["name"]
            desc = repo["description"][:50] + "..." if len(repo["description"]) > 50 else repo["description"]
            stars = f"⭐ {self._format_stars(repo['stars'])}"
            lang = repo["language"]
            url = f"[🔗]({repo['url']})"
            lines.append(f"| **{name_col}** | {desc} | {stars} | {lang} | {url} |")

        lines.append("")  # 空行
        return lines

    def _render_category_header(self, name: str, level: int, full_path: str,
                                category_descriptions: Dict, total_repos: int,
                                has_children: bool, direct_count: int) -> List[str]:
        """渲染分类标题和描述"""
        lines = []

        # 生成标题
        header = "#" * min(level, 6)
        lines.append(f"{header} {name}\n")

        # 描述
        if full_path in category_descriptions:
            lines.append(f"*{category_descriptions[full_path]}*\n")

        # 显示项目数
        if has_children:
            lines.append(f"共收录 {total_repos} 个项目\n")
        elif direct_count > 0:
            lines.append(f"收录 {direct_count} 个项目\n")

        return lines

    def _default_categorize(self, repos: List[Dict]) -> Dict[str, List[Dict]]:
        """默认按语言分类"""
        categories = {}
        for repo in repos:
            lang = repo["language"]
            if lang not in categories:
                categories[lang] = []
            categories[lang].append(repo)
        return categories

    def _format_stars(self, count: int) -> str:
        """格式化 stars 数量"""
        if count < 1000:
            return str(count)
        elif count < 10000:
            return f"{count / 1000:.1f}k"
        else:
            return f"{count / 1000:.1f}k"

    def _anchor(self, text: str) -> str:
        """生成 Markdown 锚点"""
        import re
        # 移除特殊字符
        text = re.sub(r'[^\w\s-]', '', text)
        return text.lower().replace(" ", "-")

    def _calculate_language_stats(self, repos: List[Dict]) -> Dict[str, int]:
        """计算语言统计"""
        stats = {}
        for repo in repos:
            lang = repo["language"]
            stats[lang] = stats.get(lang, 0) + 1
        return dict(sorted(stats.items(), key=lambda x: x[1], reverse=True))

    def _calculate_stars_ranges(self, repos: List[Dict]) -> Dict[str, int]:
        """计算 stars 范围统计"""
        return {
            "100k+": len([r for r in repos if r["stars"] >= 100000]),
            "50k-100k": len([r for r in repos if 50000 <= r["stars"] < 100000]),
            "10k-50k": len([r for r in repos if 10000 <= r["stars"] < 50000]),
            "1k-10k": len([r for r in repos if 1000 <= r["stars"] < 10000]),
            "<1k": len([r for r in repos if r["stars"] < 1000]),
        }


# ============================================================================
# Graph 构建
# ============================================================================


def create_github_stars_graph(llm=None):
    """创建 GitHub Stars Agent 工作流图"""

    agent = GitHubStarsAgent(llm=llm)

    # 创建图
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("fetch", agent.fetch_github_stars)  # 新增：获取数据
    workflow.add_node("analyze", agent.analyze_repositories)
    # workflow.add_node("recommend", agent.generate_recommendations)
    workflow.add_node("generate", agent.generate_markdown)

    # 添加边
    workflow.add_edge(START, "fetch")  # 从获取数据开始
    workflow.add_edge("fetch", "analyze")
    # workflow.add_edge("analyze", "recommend")
    workflow.add_edge("analyze", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


# ============================================================================
# 公共 API
# ============================================================================


def run_agent(
        github_token: Optional[str] = None,
        username: Optional[str] = None,
        min_stars: int = 0,
        output: str = "stars_index_ai.md",
        llm=None,
) -> Dict:
    """
    运行 GitHub Stars Agent
    
    Args:
        github_token: GitHub Token（如果为 None，从环境变量读取）
        username: GitHub 用户名（如果为 None，使用认证用户）
        min_stars: 最小 stars 数量过滤
        output: 输出文件路径
        llm: LLM 实例（如果为 None，使用 Qwen 模型）
    
    Returns:
        最终状态字典
    """
    # 获取 GitHub Token
    token = github_token or os.environ.get("GITHUB_TOKEN")
    if not token:
        raise ValueError(
            "未提供 GitHub Token\n"
            "请通过以下方式之一提供：\n"
            "1. 设置环境变量: export GITHUB_TOKEN=your_token\n"
            "2. 传递参数: run_agent(github_token='your_token')\n"
            "\n获取 Token: https://github.com/settings/tokens"
        )

    # 创建 Agent 和工作流图
    print("\n" + "╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "🤖 GitHub Stars AI Agent" + " " * 29 + "║")
    print("║" + " " * 20 + "Powered by Qwen" + " " * 33 + "║")
    print("╚" + "=" * 68 + "╝")

    graph = create_github_stars_graph(llm=llm)

    # 执行工作流
    print("\n🚀 开始执行 LangGraph 工作流...")
    print("   流程: fetch → analyze → recommend → generate\n")

    initial_state = {
        "messages": [HumanMessage(content="开始 GitHub Stars 智能分析")],
        "github_token": token,
        "username": username,
        "min_stars": min_stars,
        "repositories": [],
        "categories": {},
        "category_descriptions": {},
        "recommendations": [],
        "markdown_output": "",
    }

    result = graph.invoke(initial_state)

    # 输出结果
    print("\n" + "=" * 70)
    print("💾 保存结果")
    print("=" * 70)

    markdown = result["markdown_output"]
    with open(output, "w", encoding="utf-8") as f:
        f.write(markdown)

    print(f"\n" + "╔" + "=" * 68 + "╗")
    print(f"║  ✅ 完成！索引已生成" + " " * 44 + "║")
    print(f"╚" + "=" * 68 + "╝")
    print(f"\n📄 输出文件: {output}")
    print(f"   - 总仓库: {len(result['repositories'])}")
    print(f"   - 分类数: {len(result['categories'])}")
    print(f"   - 使用模型: Qwen")

    # 打印消息历史
    print("\n📝 处理日志:")
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print(f"   {msg.content}")

    return result


# ============================================================================
# CLI 入口
# ============================================================================


def main():
    """命令行入口（可选）"""
    import argparse

    parser = argparse.ArgumentParser(
        description="GitHub Stars Agent - AI 驱动的智能索引生成（使用 Qwen 模型）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 无参数运行（从环境变量读取 GITHUB_TOKEN）
  python github_stars_agent.py
  
  # 指定输出文件
  python github_stars_agent.py --output my_stars.md
  
  # 过滤高质量项目
  python github_stars_agent.py --min-stars 100
  
  # 获取其他用户的公开 stars
  python github_stars_agent.py --username other-user
        """
    )
    parser.add_argument(
        "--token",
        type=str,
        help="GitHub Personal Access Token (默认从 GITHUB_TOKEN 环境变量读取)",
    )
    parser.add_argument(
        "--username",
        type=str,
        help="GitHub 用户名 (默认使用认证用户)",
    )
    parser.add_argument(
        "--min-stars",
        type=int,
        default=0,
        help="最小 stars 数量过滤 (默认: 0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="stars_index_ai.md",
        help="输出 Markdown 文件路径 (默认: stars_index_ai.md)",
    )

    args = parser.parse_args()

    try:
        result = run_agent(
            github_token=args.token,
            username=args.username,
            min_stars=args.min_stars,
            output=args.output,
        )
        return 0

    except ValueError as e:
        print(f"❌ 配置错误: {e}")
        return 1
    except requests.exceptions.HTTPError as e:
        print(f"❌ GitHub API 错误: {e}")
        if e.response.status_code == 401:
            print("提示: Token 可能无效或已过期")
        return 1
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
