#!/usr/bin/env python3
"""
GitHub Stars 爬取脚本

从 GitHub API 获取当前用户的 star 仓库信息，并可选择性地生成索引文档。

使用方法:
    python fetch_github_stars.py --token YOUR_GITHUB_TOKEN
    python fetch_github_stars.py --token YOUR_GITHUB_TOKEN --output stars.json
    python fetch_github_stars.py --token YOUR_GITHUB_TOKEN --generate-index index.md

环境变量:
    GITHUB_TOKEN: GitHub Personal Access Token (可选，如果未通过 --token 提供)
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlparse

try:
    import requests
except ImportError:
    print("错误: 需要安装 requests 库")
    print("请运行: pip install requests")
    sys.exit(1)


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

        print(f"正在获取 starred 仓库信息...", file=sys.stderr)

        while True:
            params = {"per_page": per_page, "page": page}
            response = self.session.get(url, params=params)
            response.raise_for_status()

            repos = response.json()
            if not repos:
                break

            all_repos.extend(repos)
            print(
                f"  已获取 {len(all_repos)} 个仓库...",
                file=sys.stderr,
                end="\r",
            )

            # 检查是否还有下一页
            link_header = response.headers.get("Link", "")
            if "rel=\"next\"" not in link_header:
                break

            page += 1

        print(f"\n✓ 共获取到 {len(all_repos)} 个 starred 仓库", file=sys.stderr)
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

    def format_stars_count(self, count: int) -> str:
        """
        格式化 stars 数量

        Args:
            count: stars 数量

        Returns:
            格式化后的字符串（如 "12.5k"）
        """
        if count < 1000:
            return str(count)
        elif count < 10000:
            return f"{count / 1000:.1f}k"
        else:
            return f"{count / 1000:.1f}k"

    def categorize_repos(self, repos: List[Dict]) -> Dict[str, List[Dict]]:
        """
        根据语言自动分类仓库

        Args:
            repos: 仓库列表

        Returns:
            按语言分类的仓库字典
        """
        categories = {}
        for repo in repos:
            language = repo["language"]
            if language not in categories:
                categories[language] = []
            categories[language].append(repo)

        # 按 stars 数量排序每个分类
        for lang in categories:
            categories[lang].sort(key=lambda x: x["stars"], reverse=True)

        return categories

    def generate_language_stats(self, repos: List[Dict]) -> Dict[str, int]:
        """
        生成语言统计信息

        Args:
            repos: 仓库列表

        Returns:
            语言统计字典
        """
        stats = {}
        for repo in repos:
            lang = repo["language"]
            stats[lang] = stats.get(lang, 0) + 1
        return dict(sorted(stats.items(), key=lambda x: x[1], reverse=True))


class MarkdownGenerator:
    """Markdown 索引文档生成器"""

    def __init__(self, fetcher: GitHubStarsFetcher):
        self.fetcher = fetcher

    def generate_index(
            self,
            repos: List[Dict],
            username: str,
            group_by: str = "language",
            min_stars: int = 0,
    ) -> str:
        """
        生成 Markdown 索引文档

        Args:
            repos: 仓库列表
            username: GitHub 用户名
            group_by: 分组方式 ("language" 或 "none")
            min_stars: 最小 stars 数量过滤

        Returns:
            Markdown 格式的索引文档
        """
        # 过滤仓库
        filtered_repos = [r for r in repos if r["stars"] >= min_stars]

        # 生成文档
        md = []
        md.append(f"# GitHub Stars Index - @{username}\n")
        md.append(f"> 📚 GitHub 收藏代码库索引 | 最后更新：{datetime.now().strftime('%Y-%m-%d')}\n")
        md.append("## 📖 关于本索引\n")
        md.append(f"- **总收藏数**: {len(filtered_repos)} 个代码库")

        # 语言统计
        lang_stats = self.fetcher.generate_language_stats(filtered_repos)
        md.append(f"- **主要语言**: {', '.join(list(lang_stats.keys())[:5])}")
        md.append("")
        md.append("---\n")

        if group_by == "language":
            # 按语言分组
            categories = self.fetcher.categorize_repos(filtered_repos)

            # 按仓库数量排序分类（保持目录和内容顺序一致）
            sorted_langs = sorted(categories.keys(), key=lambda x: len(categories[x]), reverse=True)

            md.append("## 📋 目录\n")
            for lang in sorted_langs:
                lang_name = lang if lang != "-" else "其他"
                md.append(f"- [{lang_name}](#{self._anchor(lang_name)})")
            md.append("\n---\n")

            # 生成各语言分类
            for lang in sorted_langs:
                lang_name = lang if lang != "-" else "其他"
                repos_in_lang = categories[lang]

                md.append(f"## {lang_name}\n")
                md.append(f"收录 {len(repos_in_lang)} 个项目\n")

                # 生成表格
                md.append("| 名称 | 简介 | Stars | Forks | 最后更新 | 链接 |")
                md.append("|------|------|-------|-------|----------|------|")

                for repo in repos_in_lang:
                    name = repo["name"]
                    desc = repo["description"]
                    stars = f"⭐ {self.fetcher.format_stars_count(repo['stars'])}"
                    forks = f"🔱 {self.fetcher.format_stars_count(repo['forks'])}"
                    updated = repo["updated_at"][:10]
                    url = f"[GitHub]({repo['url']})"

                    md.append(f"| {name} | {desc} | {stars} | {forks} | {updated} | {url} |")

                md.append("\n---\n")
        else:
            # 不分组，直接列出所有仓库
            md.append("## 📚 所有收藏\n")
            sorted_repos = sorted(filtered_repos, key=lambda x: x["stars"], reverse=True)

            md.append("| 名称 | 简介 | Stars | 语言 | 最后更新 | 链接 |")
            md.append("|------|------|-------|------|----------|------|")

            for repo in sorted_repos:
                name = repo["name"]
                desc = repo["description"][:60] + "..." if len(repo["description"]) > 60 else repo["description"]
                stars = f"⭐ {self.fetcher.format_stars_count(repo['stars'])}"
                lang = repo["language"]
                updated = repo["updated_at"][:10]
                url = f"[GitHub]({repo['url']})"

                md.append(f"| {name} | {desc} | {stars} | {lang} | {updated} | {url} |")

            md.append("")

        # 统计信息
        md.append("## 📊 统计信息\n")
        md.append("### 按编程语言统计\n")
        total = len(filtered_repos)
        for lang, count in list(lang_stats.items())[:10]:
            percentage = (count / total) * 100
            lang_name = lang if lang != "-" else "其他"
            md.append(f"- {lang_name}: {count} ({percentage:.1f}%)")

        md.append("")
        md.append("### 按 Stars 范围统计\n")
        ranges = {
            "100k+": len([r for r in filtered_repos if r["stars"] >= 100000]),
            "50k-100k": len([r for r in filtered_repos if 50000 <= r["stars"] < 100000]),
            "10k-50k": len([r for r in filtered_repos if 10000 <= r["stars"] < 50000]),
            "1k-10k": len([r for r in filtered_repos if 1000 <= r["stars"] < 10000]),
            "<1k": len([r for r in filtered_repos if r["stars"] < 1000]),
        }
        for range_name, count in ranges.items():
            if count > 0:
                md.append(f"- {range_name}: {count} 个")

        # Top 10
        md.append("\n### 最受欢迎项目 Top 10\n")
        top_repos = sorted(filtered_repos, key=lambda x: x["stars"], reverse=True)[:10]
        for i, repo in enumerate(top_repos, 1):
            stars = self.fetcher.format_stars_count(repo["stars"])
            md.append(f"{i}. [{repo['full_name']}]({repo['url']}) - ⭐ {stars}")

        md.append("\n---\n")
        md.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        md.append(f"\n*由 [github-stars-indexer](https://github.com/) 自动生成*")

        return "\n".join(md)

    def _anchor(self, text: str) -> str:
        """生成 Markdown 锚点"""
        return text.lower().replace(" ", "-")


def main():
    parser = argparse.ArgumentParser(
        description="从 GitHub API 获取 starred 仓库信息并生成索引文档"
    )
    parser.add_argument(
        "--token",
        type=str,
        help="GitHub Personal Access Token (或通过 GITHUB_TOKEN 环境变量提供)",
    )
    parser.add_argument(
        "--username",
        type=str,
        help="GitHub 用户名 (如果不提供，则获取认证用户的 stars)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="输出 JSON 文件路径",
    )
    parser.add_argument(
        "--generate-index",
        type=str,
        metavar="FILE",
        help="生成 Markdown 索引文档",
    )
    parser.add_argument(
        "--group-by",
        type=str,
        choices=["language", "none"],
        default="language",
        help="分组方式: language (按语言) 或 none (不分组)",
    )
    parser.add_argument(
        "--min-stars",
        type=int,
        default=0,
        help="最小 stars 数量过滤",
    )

    args = parser.parse_args()

    # 获取 token
    token = args.token or os.environ.get("GITHUB_TOKEN")
    if not token:
        print("错误: 未提供 GitHub Token")
        print("请通过 --token 参数或 GITHUB_TOKEN 环境变量提供")
        print("\n如何获取 GitHub Token:")
        print("1. 访问 https://github.com/settings/tokens")
        print("2. 点击 'Generate new token' -> 'Generate new token (classic)'")
        print("3. 选择 'user:read' 权限")
        print("4. 生成并复制 token")
        sys.exit(1)

    try:
        # 初始化爬取器
        fetcher = GitHubStarsFetcher(token)

        # 获取用户信息
        if not args.username:
            user = fetcher.get_authenticated_user()
            username = user["login"]
            print(f"✓ 认证成功: {username}", file=sys.stderr)
        else:
            username = args.username

        # 获取 starred 仓库
        raw_repos = fetcher.fetch_starred_repos(args.username)

        # 提取仓库信息
        repos = [fetcher.extract_repo_info(repo) for repo in raw_repos]

        # 输出 JSON
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(repos, f, ensure_ascii=False, indent=2)
            print(f"✓ 已保存到 {args.output}", file=sys.stderr)
        elif not args.generate_index:
            # 默认输出到 stdout
            print(json.dumps(repos, ensure_ascii=False, indent=2))

        # 生成索引文档
        if args.generate_index:
            generator = MarkdownGenerator(fetcher)
            markdown = generator.generate_index(
                repos,
                username,
                group_by=args.group_by,
                min_stars=args.min_stars,
            )

            with open(args.generate_index, "w", encoding="utf-8") as f:
                f.write(markdown)

            print(f"✓ 索引文档已生成: {args.generate_index}", file=sys.stderr)
            print(f"  - 总仓库数: {len(repos)}", file=sys.stderr)
            print(f"  - 过滤后: {len([r for r in repos if r['stars'] >= args.min_stars])}", file=sys.stderr)

    except requests.exceptions.HTTPError as e:
        print(f"错误: HTTP 请求失败 - {e}", file=sys.stderr)
        if e.response.status_code == 401:
            print("提示: Token 可能无效或已过期", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
