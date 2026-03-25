#!/usr/bin/env python3
"""验证PPT内容脚本"""

import argparse
import json
from pathlib import Path


def validate_structure(content: str) -> list:
    """验证PPT结构"""
    issues = []
    lines = content.split("\n")

    slide_count = sum(1 for line in lines if line.startswith("## "))
    if slide_count > 20:
        issues.append(
            f"Too many slides ({slide_count}), consider splitting into multiple presentations"
        )

    if slide_count < 3:
        issues.append(f"Too few slides ({slide_count}), add more content")

    return issues


def main():
    parser = argparse.ArgumentParser(description="Validate PPT content")
    parser.add_argument("content", help="Markdown content or file path")

    args = parser.parse_args()

    content = args.content
    if Path(content).exists():
        content = Path(content).read_text(encoding="utf-8")

    issues = validate_structure(content)

    result = {
        "is_valid": len(issues) == 0,
        "issues": issues,
        "suggestions": [
            "Keep each slide focused on one idea",
            "Use bullet points instead of long paragraphs",
            "Include visuals where appropriate",
        ],
    }

    print(json.dumps(result, indent=2))
    return 0 if result["is_valid"] else 1


if __name__ == "__main__":
    exit(main())
