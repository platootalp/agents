"""
Web工具 - 网络搜索和网页获取

提供web搜索和网页内容获取功能
"""

import json
import urllib.parse
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..base import BaseTool, tool


class WebSearchInput(BaseModel):
    """Web搜索输入参数"""

    query: str = Field(description="搜索查询")
    num_results: int = Field(default=5, description="返回结果数量", ge=1, le=10)


class WebFetchInput(BaseModel):
    """Web获取输入参数"""

    url: str = Field(description="要获取的URL")
    timeout: int = Field(default=30, description="超时时间（秒）", ge=1, le=120)
    headers: Optional[Dict[str, str]] = Field(default=None, description="自定义请求头")


class WebSearchResult(BaseModel):
    """Web搜索结果"""

    title: str = Field(description="标题")
    url: str = Field(description="URL")
    snippet: str = Field(description="摘要")


class WebSearchTool(BaseTool):
    """
    Web搜索工具

    使用DuckDuckGo进行网络搜索（无需API密钥）
    """

    def __init__(self):
        super().__init__(
            name="web_search",
            description="搜索网络信息，返回相关网页的标题、URL和摘要",
            args_schema=WebSearchInput,
        )

    def _run(self, query: str, num_results: int = 5) -> str:
        """
        执行网络搜索

        Args:
            query: 搜索查询
            num_results: 结果数量

        Returns:
            str: JSON格式的搜索结果
        """
        try:
            # 尝试使用duckduckgo-search库
            try:
                from duckduckgo_search import DDGS

                with DDGS() as ddgs:
                    results = list(ddgs.text(query, max_results=num_results))

                    search_results = [
                        {
                            "title": r.get("title", ""),
                            "url": r.get("href", ""),
                            "snippet": r.get("body", ""),
                        }
                        for r in results
                    ]

                    return json.dumps(
                        {
                            "query": query,
                            "results": search_results,
                            "count": len(search_results),
                        },
                        ensure_ascii=False,
                        indent=2,
                    )

            except ImportError:
                # 如果没有duckduckgo-search，使用模拟数据
                return json.dumps(
                    {
                        "query": query,
                        "results": [
                            {
                                "title": f"搜索结果 {i + 1} for '{query}'",
                                "url": f"https://example.com/result/{i + 1}",
                                "snippet": f"这是关于 '{query}' 的模拟搜索结果摘要...",
                            }
                            for i in range(min(num_results, 3))
                        ],
                        "count": min(num_results, 3),
                        "note": "使用模拟数据（未安装duckduckgo-search库）",
                    },
                    ensure_ascii=False,
                    indent=2,
                )

        except Exception as e:
            return json.dumps(
                {
                    "query": query,
                    "error": str(e),
                    "results": [],
                },
                ensure_ascii=False,
            )

    async def _arun(self, query: str, num_results: int = 5) -> str:
        """异步执行网络搜索"""
        # 由于duckduckgo-search不支持异步，使用线程池
        import asyncio

        return await asyncio.to_thread(self._run, query, num_results)


class WebFetchTool(BaseTool):
    """
    Web获取工具

    获取网页内容并转换为markdown或纯文本
    """

    def __init__(self):
        super().__init__(
            name="web_fetch",
            description="获取网页内容，支持转换为markdown格式",
            args_schema=WebFetchInput,
        )

    def _run(
        self,
        url: str,
        timeout: int = 30,
        headers: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        获取网页内容

        Args:
            url: 目标URL
            timeout: 超时时间
            headers: 自定义请求头

        Returns:
            str: JSON格式的获取结果
        """
        try:
            import requests
            from urllib.parse import urlparse

            # 验证URL
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return json.dumps(
                    {
                        "url": url,
                        "error": "无效的URL格式",
                        "success": False,
                    },
                    ensure_ascii=False,
                )

            # 默认请求头
            default_headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }

            if headers:
                default_headers.update(headers)

            # 发送请求
            response = requests.get(
                url,
                headers=default_headers,
                timeout=timeout,
                allow_redirects=True,
            )
            response.raise_for_status()

            # 尝试转换为markdown
            content = response.text
            content_type = response.headers.get("Content-Type", "")

            markdown_content = None
            if "text/html" in content_type:
                try:
                    import html2text

                    h = html2text.HTML2Text()
                    h.ignore_links = False
                    markdown_content = h.handle(content)
                except ImportError:
                    # 如果没有html2text，尝试简单提取
                    try:
                        from bs4 import BeautifulSoup

                        soup = BeautifulSoup(content, "html.parser")
                        # 移除script和style
                        for script in soup(["script", "style"]):
                            script.decompose()
                        markdown_content = soup.get_text(separator="\n", strip=True)
                    except ImportError:
                        markdown_content = content
            else:
                markdown_content = content

            result = {
                "url": url,
                "status_code": response.status_code,
                "content_type": content_type,
                "title": self._extract_title(content) if "text/html" in content_type else None,
                "content_length": len(content),
                "markdown": markdown_content[:10000] if markdown_content else None,  # 限制长度
                "success": True,
            }

            return json.dumps(result, ensure_ascii=False, indent=2)

        except requests.exceptions.Timeout:
            return json.dumps(
                {
                    "url": url,
                    "error": f"请求超时（{timeout}秒）",
                    "success": False,
                },
                ensure_ascii=False,
            )

        except requests.exceptions.RequestException as e:
            return json.dumps(
                {
                    "url": url,
                    "error": f"请求错误: {str(e)}",
                    "success": False,
                },
                ensure_ascii=False,
            )

        except Exception as e:
            return json.dumps(
                {
                    "url": url,
                    "error": str(e),
                    "success": False,
                },
                ensure_ascii=False,
            )

    async def _arun(
        self,
        url: str,
        timeout: int = 30,
        headers: Optional[Dict[str, str]] = None,
    ) -> str:
        """异步获取网页内容"""
        import asyncio

        return await asyncio.to_thread(self._run, url, timeout, headers)

    def _extract_title(self, html: str) -> Optional[str]:
        """从HTML中提取标题"""
        import re

        match = re.search(r"<title[^>]*>([^<]*)</title>", html, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None


# 便捷的函数版本
@tool(name="web_search", description="搜索网络信息")
def web_search(query: str, num_results: int = 5) -> str:
    """便捷的网络搜索"""
    tool = WebSearchTool()
    result = tool.run(query=query, num_results=num_results)
    return str(result.output) if result.success else f"Error: {result.error}"


@tool(name="web_fetch", description="获取网页内容")
def web_fetch(url: str, timeout: int = 30) -> str:
    """便捷的网页获取"""
    tool = WebFetchTool()
    result = tool.run(url=url, timeout=timeout)
    return str(result.output) if result.success else f"Error: {result.error}"
