#!/usr/bin/env python3
"""
GitHub Stars Agent - 基于 LangGraph 的智能索引生成器

使用 LLM 智能分析和分类 GitHub starred 仓库，生成高质量的索引文档。

功能：
- 智能分类：使用 LLM 分析仓库的技术栈、用途、领域
- 自动总结：为每个分类生成描述性说明
- 智能排序：根据相关性和重要性排序
- 推荐系统：识别关键项目和学习路径
"""

import json
import os
from datetime import datetime
from typing import Annotated, Dict, List, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


# ============================================================================
# State 定义
# ============================================================================


class AgentState(TypedDict):
    """Agent 状态"""
    messages: Annotated[List, add_messages]
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
            llm: LLM 实例，默认使用 GPT-4
        """
        self.llm = llm or ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
        )

    def analyze_repositories(self, state: AgentState) -> AgentState:
        """
        分析仓库并智能分类

        使用 LLM 分析每个仓库的技术栈、用途和领域，进行智能分类
        """
        repos = state["repositories"]

        # 准备仓库摘要
        repo_summaries = []
        for i, repo in enumerate(repos[:50]):  # 限制分析数量
            summary = {
                "index": i,
                "name": repo["full_name"],
                "description": repo["description"],
                "language": repo["language"],
                "topics": repo.get("topics", []),
                "stars": repo["stars"],
            }
            repo_summaries.append(summary)

        prompt = f"""你是一个技术栈分析专家。请分析以下 GitHub 仓库，将它们分类到合适的技术领域。
            
            仓库列表：
            {json.dumps(repo_summaries, ensure_ascii=False, indent=2)}
            
            请返回分类结果，格式为 JSON：
            {{
              "categories": {{
                "分类名称1": {{
                  "description": "分类描述",
                  "repos": [0, 3, 5]  // 仓库索引列表
                }},
                "分类名称2": {{
                  "description": "分类描述",
                  "repos": [1, 2, 4]
                }}
              }}
            }}
            
            分类要求：
            1. 使用中文分类名称，清晰准确
            2. 每个分类包含 3-15 个相关仓库
            3. 分类要有层次感：大类（如：AI/机器学习、Web开发）→ 子类（如：LLM工具链、前端框架）
            4. 为每个分类写简短描述（1-2句话）
            5. 热门/重要的项目优先考虑单独分类
            
            只返回 JSON，不要其他内容。
        """

        messages = [
            SystemMessage(content="你是一个专业的技术分类专家。"),
            HumanMessage(content=prompt),
        ]

        response = self.llm.invoke(messages)

        try:
            # 解析 LLM 返回的分类结果
            result = json.loads(response.content)
            categories_info = result.get("categories", {})

            # 构建分类字典
            categories = {}
            category_descriptions = {}

            for cat_name, cat_info in categories_info.items():
                category_descriptions[cat_name] = cat_info["description"]
                categories[cat_name] = [
                    repos[idx] for idx in cat_info["repos"] if idx < len(repos)
                ]

            state["categories"] = categories
            state["category_descriptions"] = category_descriptions
            state["messages"].append(AIMessage(content=f"✓ 完成智能分类，共 {len(categories)} 个分类"))

        except json.JSONDecodeError:
            # 如果 LLM 返回格式有误，使用默认按语言分类
            print("警告: LLM 返回格式错误，使用默认分类")
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
        categories = state["categories"]
        repos = state["repositories"]

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

        response = self.llm.invoke(messages)
        state["recommendations"] = response.content.split("\n")
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
        md.append("- **AI 分析**: 由 LangGraph Agent 智能整理\n")
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
                desc = repo["description"][:50] + "..." if len(repo["description"]) > 50 else repo["description"]
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
        md.append(f"*🤖 由 GitHub Stars Agent 智能生成*")

        state["markdown_output"] = "\n".join(md)
        state["messages"].append(AIMessage(content="✓ Markdown 文档生成完成"))

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


def create_github_stars_graph():
    """创建 GitHub Stars Agent 工作流图"""

    agent = GitHubStarsAgent()

    # 创建图
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("analyze", agent.analyze_repositories)
    workflow.add_node("recommend", agent.generate_recommendations)
    workflow.add_node("generate", agent.generate_markdown)

    # 添加边
    workflow.add_edge(START, "analyze")
    workflow.add_edge("analyze", "recommend")
    workflow.add_edge("recommend", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


# ============================================================================
# CLI 入口
# ============================================================================


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(
        description="GitHub Stars Agent - AI 驱动的智能索引生成"
    )
    parser.add_argument(
        "input",
        type=str,
        help="输入 JSON 文件路径（fetch_stars.py 的输出）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="stars_index_ai.md",
        help="输出 Markdown 文件路径",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="使用的 LLM 模型",
    )

    args = parser.parse_args()

    try:
        # 读取数据
        print("📖 读取仓库数据...")
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)

        repos = data["repositories"]
        print(f"✓ 加载了 {len(repos)} 个仓库")

        # 创建 Agent
        print("\n🤖 启动 AI Agent...")
        llm = ChatOpenAI(model=args.model, temperature=0.3)
        agent = GitHubStarsAgent(llm=llm)

        # 创建图
        graph = create_github_stars_graph()

        # 执行工作流
        print("\n🔄 开始分析...")
        initial_state = {
            "messages": [HumanMessage(content=f"分析 {len(repos)} 个 GitHub stars")],
            "repositories": repos,
            "categories": {},
            "category_descriptions": {},
            "recommendations": [],
            "markdown_output": "",
        }

        result = graph.invoke(initial_state)

        # 输出结果
        markdown = result["markdown_output"]
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(markdown)

        print(f"\n✅ 完成！索引已生成: {args.output}")
        print(f"   - 总仓库: {len(repos)}")
        print(f"   - 分类数: {len(result['categories'])}")
        print(f"   - 使用模型: {args.model}")

        # 打印消息历史
        print("\n📝 处理日志:")
        for msg in result["messages"]:
            if isinstance(msg, AIMessage):
                print(f"   {msg.content}")

    except FileNotFoundError:
        print(f"❌ 错误: 文件不存在 - {args.input}")
        return 1
    except json.JSONDecodeError as e:
        print(f"❌ 错误: JSON 解析失败 - {e}")
        return 1
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
