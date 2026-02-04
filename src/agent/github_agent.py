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
        max_repos = 100  # 限制最大分析数量
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

        prompt = f"""分析以下 {len(repo_summaries)} 个 GitHub 仓库，按技术领域分类。

仓库列表：
{repo_list_str}

要求：
1. 创建 5-10 个中文分类（如：AI → LLM-> Agent、Web开发 → 前端框架）
2. 每个分类必须包含至少1个仓库
3. 所有 {len(repo_summaries)} 个仓库都必须被分配

返回JSON格式：
{{
  "categories": {{
    "分类名": {{
      "description": "简短描述",
      "repos": [仓库index数组，如 [0, 3, 5]]
    }}
  }}
}}

重要：repos 数组必须包含实际的仓库编号，不能为空！"""

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
            categories_info = result.get("categories", {})

            # 构建分类字典
            categories = {}
            category_descriptions = {}
            
            print(f"\n✅ AI 分类完成，共 {len(categories_info)} 个分类")

            for cat_name, cat_info in categories_info.items():
                category_descriptions[cat_name] = cat_info["description"]
                repo_indices = cat_info.get("repos", [])
                
                # 使用 analyze_repos 而不是原始 repos
                categories[cat_name] = [
                    analyze_repos[idx] for idx in repo_indices if idx < len(analyze_repos)
                ]
                
                print(f"   - {cat_name}: {len(categories[cat_name])} 个仓库")
            
            # 检查是否有仓库未被分类
            all_assigned = sum(len(cat_info.get("repos", [])) for cat_info in categories_info.values())
            if all_assigned == 0:
                print("\n⚠️  警告: AI 未分配任何仓库，使用默认语言分类")
                categories = self._default_categorize(repos)
                category_descriptions = {}
            elif all_assigned < len(analyze_repos):
                print(f"\n⚠️  注意: 有 {len(analyze_repos) - all_assigned} 个仓库未被分类")

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
        categories = state["categories"]
        category_descriptions = state["category_descriptions"]
        recommendations = state["recommendations"]
        repos = state["repositories"]

        md = []
        username = repos[0].get("owner", "unknown") if repos else "unknown"

        # 头部
        md.append(f"# 🌟 GitHub Stars 智能索引\n")
        md.append(f"> 📚 AI 驱动的个性化收藏库 | 更新时间：{datetime.now().strftime('%Y-%m-%d')}\n")
        md.append("## 📖 关于\n")
        md.append(f"- **总收藏**: {len(repos)} 个项目")
        md.append(f"- **智能分类**: {len(categories)} 个领域")
        md.append("- **AI 分析**: 由 LangGraph Agent (Qwen) 智能整理\n")
        md.append("---\n")

        # 目录
        md.append("## 📋 目录\n")
        sorted_cats = sorted(
            categories.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )
        for cat_name, cat_repos in sorted_cats:
            md.append(f"- [{cat_name}](#{self._anchor(cat_name)}) ({len(cat_repos)}个)")
        md.append("\n---\n")

        # 推荐部分
        if recommendations:
            md.append("## 💡 AI 推荐\n")
            md.extend(recommendations)
            md.append("\n---\n")

        # 分类内容
        for cat_name, cat_repos in sorted_cats:
            md.append(f"## {cat_name}\n")

            # 分类描述
            if cat_name in category_descriptions:
                md.append(f"*{category_descriptions[cat_name]}*\n")

            md.append(f"收录 {len(cat_repos)} 个项目\n")

            # 表格
            md.append("| 名称 | 简介 | Stars | 语言 | 链接 |")
            md.append("|------|------|-------|------|------|")

            # 按 stars 排序
            sorted_repos = sorted(cat_repos, key=lambda x: x["stars"], reverse=True)

            for repo in sorted_repos:
                name = repo["name"]
                desc = repo["description"]
                stars = f"⭐ {self._format_stars(repo['stars'])}"
                lang = repo["language"]
                url = f"[🔗]({repo['url']})"

                md.append(f"| **{name}** | {desc} | {stars} | {lang} | {url} |")

            md.append("\n---\n")

        # 统计
        md.append("## 📊 统计分析\n")
        lang_stats = self._calculate_language_stats(repos)
        md.append("### 编程语言分布\n")
        for lang, count in list(lang_stats.items())[:8]:
            percentage = (count / len(repos)) * 100
            md.append(f"- **{lang}**: {count} 个 ({percentage:.1f}%)")

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

        md.append("\n---\n")
        md.append(f"*📅 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
        md.append(f"*🤖 由 GitHub Stars Agent (Qwen) 智能生成*")

        state["markdown_output"] = "\n".join(md)
        state["messages"].append(AIMessage(content="✓ Markdown 文档生成完成"))
        
        print("✅ 文档生成完成！")

        return state

    # ========================================================================
    # 辅助方法
    # ========================================================================

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
  python github_agent.py
  
  # 指定输出文件
  python github_agent.py --output my_stars.md
  
  # 过滤高质量项目
  python github_agent.py --min-stars 100
  
  # 获取其他用户的公开 stars
  python github_agent.py --username other-user
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
