#!/usr/bin/env python3
"""
GitHub Stars 索引生成脚本

从 JSON 数据生成 Markdown 格式的索引文档。

使用方法:
    python generate_index.py stars.json
    python generate_index.py stars.json --output index.md
    python generate_index.py stars.json --group-by language
    python generate_index.py stars.json --sort-by stars
"""

import argparse
import json
import sys
from datetime import datetime
from typing import Dict, List


class MarkdownGenerator:
    """Markdown 索引文档生成器"""

    def format_stars_count(self, count: int) -> str:
        """格式化 stars 数量"""
        if count < 1000:
            return str(count)
        elif count < 10000:
            return f"{count / 1000:.1f}k"
        else:
            return f"{count / 1000:.1f}k"

    def categorize_by_language(self, repos: List[Dict]) -> Dict[str, List[Dict]]:
        """按语言分类仓库"""
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
        """生成语言统计信息"""
        stats = {}
        for repo in repos:
            lang = repo["language"]
            stats[lang] = stats.get(lang, 0) + 1
        return dict(sorted(stats.items(), key=lambda x: x[1], reverse=True))

    def generate_index(
        self,
        data: Dict,
        group_by: str = "language",
        sort_by: str = "stars",
    ) -> str:
        """
        生成 Markdown 索引文档

        Args:
            data: 包含仓库信息的字典
            group_by: 分组方式 ("language" 或 "none")
            sort_by: 排序方式 ("stars" 或 "updated")

        Returns:
            Markdown 格式的索引文档
        """
        repos = data["repositories"]
        username = data["username"]

        md = []
        md.append(f"# GitHub Stars Index - @{username}\n")
        md.append(f"> 📚 GitHub 收藏代码库索引 | 最后更新：{datetime.now().strftime('%Y-%m-%d')}\n")
        md.append("## 📖 关于本索引\n")
        md.append(f"- **总收藏数**: {len(repos)} 个代码库")

        # 语言统计
        lang_stats = self.generate_language_stats(repos)
        md.append(f"- **主要语言**: {', '.join(list(lang_stats.keys())[:5])}")
        md.append("")
        md.append("---\n")

        if group_by == "language":
            # 按语言分组
            categories = self.categorize_by_language(repos)
            
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
                    desc = repo["description"][:50] + "..." if len(repo["description"]) > 50 else repo["description"]
                    stars = f"⭐ {self.format_stars_count(repo['stars'])}"
                    forks = f"🔱 {self.format_stars_count(repo['forks'])}"
                    updated = repo["updated_at"][:10]
                    url = f"[GitHub]({repo['url']})"

                    md.append(f"| {name} | {desc} | {stars} | {forks} | {updated} | {url} |")

                md.append("\n---\n")
        else:
            # 不分组，直接列出所有仓库
            md.append("## 📚 所有收藏\n")
            
            if sort_by == "stars":
                sorted_repos = sorted(repos, key=lambda x: x["stars"], reverse=True)
            elif sort_by == "updated":
                sorted_repos = sorted(repos, key=lambda x: x["updated_at"], reverse=True)
            else:
                sorted_repos = repos

            md.append("| 名称 | 简介 | Stars | 语言 | 最后更新 | 链接 |")
            md.append("|------|------|-------|------|----------|------|")

            for repo in sorted_repos:
                name = repo["name"]
                desc = repo["description"][:60] + "..." if len(repo["description"]) > 60 else repo["description"]
                stars = f"⭐ {self.format_stars_count(repo['stars'])}"
                lang = repo["language"]
                updated = repo["updated_at"][:10]
                url = f"[GitHub]({repo['url']})"

                md.append(f"| {name} | {desc} | {stars} | {lang} | {updated} | {url} |")

            md.append("")

        # 统计信息
        md.append("## 📊 统计信息\n")
        md.append("### 按编程语言统计\n")
        total = len(repos)
        for lang, count in list(lang_stats.items())[:10]:
            percentage = (count / total) * 100
            lang_name = lang if lang != "-" else "其他"
            md.append(f"- {lang_name}: {count} ({percentage:.1f}%)")

        md.append("")
        md.append("### 按 Stars 范围统计\n")
        ranges = {
            "100k+": len([r for r in repos if r["stars"] >= 100000]),
            "50k-100k": len([r for r in repos if 50000 <= r["stars"] < 100000]),
            "10k-50k": len([r for r in repos if 10000 <= r["stars"] < 50000]),
            "1k-10k": len([r for r in repos if 1000 <= r["stars"] < 10000]),
            "<1k": len([r for r in repos if r["stars"] < 1000]),
        }
        for range_name, count in ranges.items():
            if count > 0:
                md.append(f"- {range_name}: {count} 个")

        # Top 10
        md.append("\n### 最受欢迎项目 Top 10\n")
        top_repos = sorted(repos, key=lambda x: x["stars"], reverse=True)[:10]
        for i, repo in enumerate(top_repos, 1):
            stars = self.format_stars_count(repo["stars"])
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
        description="从 JSON 数据生成 GitHub Stars 索引文档"
    )
    parser.add_argument(
        "input",
        type=str,
        help="输入 JSON 文件路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="输出 Markdown 文件路径（不提供则输出到 stdout）",
    )
    parser.add_argument(
        "--group-by",
        type=str,
        choices=["language", "none"],
        default="language",
        help="分组方式: language (按语言) 或 none (不分组)",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        choices=["stars", "updated"],
        default="stars",
        help="排序方式: stars (按星标) 或 updated (按更新时间)",
    )

    args = parser.parse_args()

    try:
        # 读取 JSON 数据
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 生成索引
        generator = MarkdownGenerator()
        markdown = generator.generate_index(
            data,
            group_by=args.group_by,
            sort_by=args.sort_by,
        )

        # 输出
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(markdown)
            print(f"✓ 索引文档已生成: {args.output}", file=sys.stderr)
            print(f"  - 总仓库数: {len(data['repositories'])}", file=sys.stderr)
        else:
            print(markdown)

    except FileNotFoundError:
        print(f"错误: 文件不存在 - {args.input}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"错误: JSON 解析失败 - {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
