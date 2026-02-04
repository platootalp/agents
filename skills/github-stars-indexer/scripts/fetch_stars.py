#!/usr/bin/env python3
"""
GitHub Stars 数据获取脚本

从 GitHub API 获取用户的 starred 仓库信息，输出为 JSON 格式。

使用方法:
    python fetch_stars.py --token YOUR_GITHUB_TOKEN
    python fetch_stars.py --token YOUR_GITHUB_TOKEN --output stars.json
    python fetch_stars.py --token YOUR_GITHUB_TOKEN --username other-user

环境变量:
    GITHUB_TOKEN: GitHub Personal Access Token
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

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


def main():
    parser = argparse.ArgumentParser(
        description="从 GitHub API 获取 starred 仓库信息"
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
        help="输出 JSON 文件路径（不提供则输出到 stdout）",
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
        print("错误: 未提供 GitHub Token", file=sys.stderr)
        print("请通过 --token 参数或 GITHUB_TOKEN 环境变量提供", file=sys.stderr)
        print("\n如何获取 GitHub Token:", file=sys.stderr)
        print("1. 访问 https://github.com/settings/tokens", file=sys.stderr)
        print("2. 点击 'Generate new token' -> 'Generate new token (classic)'", file=sys.stderr)
        print("3. 选择 'user:read' 权限", file=sys.stderr)
        print("4. 生成并复制 token", file=sys.stderr)
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

        # 过滤
        if args.min_stars > 0:
            repos = [r for r in repos if r["stars"] >= args.min_stars]
            print(f"✓ 过滤后剩余 {len(repos)} 个仓库 (>= {args.min_stars} stars)", file=sys.stderr)

        # 添加元数据
        result = {
            "username": username,
            "total_count": len(repos),
            "fetched_at": raw_repos[0]["updated_at"] if raw_repos else None,
            "repositories": repos,
        }

        # 输出
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"✓ 数据已保存到 {args.output}", file=sys.stderr)
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))

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
