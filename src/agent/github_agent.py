import os
import requests
import json
from typing import TypedDict, List, Dict, Any, Optional
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from util import model

load_dotenv()


class AgentState(TypedDict):
    github_token: str
    starred_repos: List[Dict[str, Any]]
    categorized_repos: Dict[str, List[Dict[str, Any]]]
    markdown_content: str
    error: Optional[str]


class GitHubStarredRepoAgent:
    def __init__(self):
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.llm = model
        self.graph = self._create_graph()

    def _create_graph(self):
        workflow = StateGraph(AgentState)

        # 定义节点
        workflow.add_node("fetch_starred_repos", self.fetch_starred_repos)
        workflow.add_node("categorize_repos", self.categorize_repos)
        workflow.add_node("generate_markdown", self.generate_markdown)
        workflow.add_node("handle_error", self.handle_error)

        # 设置入口点
        workflow.set_entry_point("fetch_starred_repos")

        # 添加条件边
        workflow.add_conditional_edges(
            "fetch_starred_repos",
            self.check_fetch_success,
            {
                "success": "categorize_repos",
                "error": "handle_error"
            }
        )

        workflow.add_conditional_edges(
            "categorize_repos",
            self.check_categorize_success,
            {
                "success": "generate_markdown",
                "error": "handle_error"
            }
        )

        workflow.add_edge("generate_markdown", END)
        workflow.add_edge("handle_error", END)

        return workflow.compile()

    def fetch_starred_repos(self, state: AgentState) -> AgentState:
        """获取GitHub收藏的仓库"""
        try:
            headers = {
                "Authorization": f"token {self.github_token}",
                "Accept": "application/vnd.github.v3+json"
            }

            repos = []
            page = 1
            per_page = 100

            while True:
                url = f"https://api.github.com/user/starred?per_page={per_page}&page={page}"
                response = requests.get(url, headers=headers)

                if response.status_code != 200:
                    raise Exception(f"Failed to fetch starred repos: {response.text}")

                page_repos = response.json()
                if not page_repos:
                    break

                repos.extend(page_repos)
                page += 1

            # 提取关键信息
            simplified_repos = []
            for repo in repos:
                simplified_repos.append({
                    "name": repo["name"],
                    "full_name": repo["full_name"],
                    "description": repo.get("description", ""),
                    "language": repo.get("language", "Unknown"),
                    "html_url": repo["html_url"],
                    "topics": repo.get("topics", []),
                    "stargazers_count": repo.get("stargazers_count", 0)
                })

            return {
                **state,
                "starred_repos": simplified_repos,
                "error": None
            }

        except Exception as e:
            return {
                **state,
                "error": f"Error fetching starred repos: {str(e)}"
            }

    def categorize_repos(self, state: AgentState) -> AgentState:
        """使用AI对仓库进行分类"""
        try:
            repos = state["starred_repos"]

            # 准备分类提示
            repo_info = "\n".join([
                f"- {repo['full_name']}: {repo['description']} (Language: {repo['language']}, Topics: {', '.join(repo['topics'])})"
                for repo in repos[:50]  # 限制数量，避免token过多
            ])

            prompt = ChatPromptTemplate.from_messages([
                ("system", """你是一个专业的代码仓库分类专家。请根据仓库的名称、描述、编程语言和主题，将以下GitHub仓库分类到合适的类别中。

分类要求：
1. 创建5-8个有意义的类别
2. 每个类别应该有清晰的主题
3. 一个仓库可以属于多个类别
4. 类别名称应该简洁明了

请以JSON格式返回，格式如下：
{
    "categories": {
        "类别名称1": ["仓库全名1", "仓库全名2", ...],
        "类别名称2": ["仓库全名3", ...],
        ...
    }
}"""),
                ("human", f"需要分类的仓库列表：\n{repo_info}")
            ])

            chain = prompt | self.llm
            response = chain.invoke({})

            # 解析JSON响应
            try:
                result = json.loads(response.content)
                categories = result.get("categories", {})
            except:
                # 如果解析失败，使用备用分类方法
                categories = self._fallback_categorization(repos)

            # 构建分类后的仓库字典
            categorized = {}
            for category, repo_names in categories.items():
                categorized[category] = [
                    repo for repo in repos
                    if repo["full_name"] in repo_names
                ]

            return {
                **state,
                "categorized_repos": categorized,
                "error": None
            }

        except Exception as e:
            return {
                **state,
                "error": f"Error categorizing repos: {str(e)}"
            }

    def _fallback_categorization(self, repos: List[Dict]) -> Dict[str, List[str]]:
        """备用分类方法：基于编程语言和主题"""
        categories = {}

        # 按语言分类
        language_categories = {}
        for repo in repos:
            lang = repo["language"] or "Other"
            if lang not in language_categories:
                language_categories[lang] = []
            language_categories[lang].append(repo["full_name"])

        # 按主题分类
        topic_categories = {}
        for repo in repos:
            for topic in repo["topics"]:
                if topic not in topic_categories:
                    topic_categories[topic] = []
                if repo["full_name"] not in topic_categories[topic]:
                    topic_categories[topic].append(repo["full_name"])

        # 合并分类（选择最有意义的）
        all_categories = {**language_categories, **topic_categories}

        # 选择top 8个类别
        sorted_categories = sorted(
            all_categories.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )[:8]

        return dict(sorted_categories)

    def generate_markdown(self, state: AgentState) -> AgentState:
        """生成Markdown文件内容"""
        try:
            categorized = state["categorized_repos"]

            markdown_lines = ["# My GitHub Starred Repositories\n\n"]
            markdown_lines.append(f"**Total Repositories:** {sum(len(repos) for repos in categorized.values())}\n\n")

            for category, repos in categorized.items():
                markdown_lines.append(f"## 📂 {category}\n")
                markdown_lines.append(f"**Count:** {len(repos)}\n\n")

                for repo in sorted(repos, key=lambda x: x["stargazers_count"], reverse=True):
                    desc = repo["description"] or "No description"
                    lang = repo["language"] or "Unknown"
                    stars = repo["stargazers_count"]

                    markdown_lines.append(f"### [{repo['full_name']}]({repo['html_url']})")
                    markdown_lines.append(f"- **Description:** {desc}")
                    markdown_lines.append(f"- **Language:** {lang}")
                    markdown_lines.append(f"- **Stars:** ⭐ {stars}")

                    if repo["topics"]:
                        topics_str = ", ".join([f"`{t}`" for t in repo["topics"]])
                        markdown_lines.append(f"- **Topics:** {topics_str}")

                    markdown_lines.append("")

                markdown_lines.append("---\n")

            markdown_content = "\n".join(markdown_lines)

            # 保存到文件
            with open("starred_repos_categorized.md", "w", encoding="utf-8") as f:
                f.write(markdown_content)

            return {
                **state,
                "markdown_content": markdown_content,
                "error": None
            }

        except Exception as e:
            return {
                **state,
                "error": f"Error generating markdown: {str(e)}"
            }

    def check_fetch_success(self, state: AgentState) -> str:
        """检查获取仓库是否成功"""
        return "success" if state.get("error") is None and state.get("starred_repos") else "error"

    def check_categorize_success(self, state: AgentState) -> str:
        """检查分类是否成功"""
        return "success" if state.get("error") is None and state.get("categorized_repos") else "error"

    def handle_error(self, state: AgentState) -> AgentState:
        """处理错误"""
        print(f"❌ Error occurred: {state['error']}")
        return state

    def run(self):
        """运行Agent"""
        print("🚀 Starting GitHub Starred Repo Categorization Agent...")

        initial_state = {
            "github_token": self.github_token,
            "starred_repos": [],
            "categorized_repos": {},
            "markdown_content": "",
            "error": None
        }

        result = self.graph.invoke(initial_state)

        if result.get("error"):
            print(f"❌ Process failed: {result['error']}")
            return False
        else:
            print("✅ Process completed successfully!")
            print(f"📄 Markdown file saved as: starred_repos_categorized.md")
            return True


# 使用示例
if __name__ == "__main__":
    agent = GitHubStarredRepoAgent()
    agent.run()
