"""
Wiki MCP Server - Playwright-based Wiki Operations
基于 Playwright 的 Wiki MCP 服务器，提供搜索、读取、创建、修改、列表功能
"""

import asyncio
import json
import os
import re
import socket
import sys
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

from loguru import logger
from mcp.server.fastmcp import FastMCP
from playwright.async_api import Browser, BrowserContext, Page, async_playwright
from pydantic import BaseModel, Field

# Initialize MCP Server
mcp = FastMCP("wiki-mcp-server")

# Global state for browser connection
_browser: Browser | None = None
_context: BrowserContext | None = None
_page: Page | None = None
_wiki_base_url: str = "https://wiki.tuhu.cn"
_cdp_endpoint: str = "http://localhost:9222"


class WikiConfig(BaseModel):
    """Wiki 配置"""

    base_url: str = Field(default="https://wiki.tuhu.cn", description="Wiki 基础 URL")
    space_key: str = Field(default="~lijunyi3", description="默认空间 Key")
    cdp_endpoint: str = Field(
        default="http://localhost:9222", description="Chrome DevTools Protocol 端点"
    )


# ============== Browser Management ==============


async def ensure_browser() -> Page:
    """确保浏览器已连接"""
    global _browser, _context, _page

    if _page is not None:
        try:
            await _page.evaluate("1")
            logger.debug("浏览器页面已连接，复用现有连接")
            return _page
        except Exception:
            logger.warning("浏览器页面已断开，重新连接...")
            _page = None
            _context = None
            _browser = None

    if _browser is None:
        logger.info(f"[BROWSER] 连接到 Chrome CDP: {_cdp_endpoint}")
        try:
            playwright = await async_playwright().start()
            _browser = await playwright.chromium.connect_over_cdp(_cdp_endpoint)
            logger.info("[BROWSER] Chrome 连接成功")
        except Exception as e:
            logger.error(f"[BROWSER] Chrome 连接失败: {e}")
            raise

    if _context is None:
        _context = _browser.contexts[0] if _browser.contexts else await _browser.new_context()

    if _page is None:
        _page = _context.pages[0] if _context.pages else await _context.new_page()

    return _page


async def close_browser():
    """关闭浏览器连接"""
    global _browser, _context, _page
    if _page:
        await _page.close()
        _page = None
    if _context:
        await _context.close()
        _context = None
    if _browser:
        await _browser.close()
        _browser = None


# ============== Helper Functions ==============


def extract_page_id(url: str) -> str | None:
    """从 URL 中提取 pageId"""
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    if "pageId" in query:
        return query["pageId"][0]

    # 尝试从路径中提取
    match = re.search(r"/pages/viewpage\.action\?pageId=(\d+)", url)
    if match:
        return match.group(1)

    # 尝试从 /display/SPACEKEY/PAGETITLE 格式中提取
    match = re.search(r"/display/[^/]+/[^/]+", url)
    if match:
        return url  # 返回完整 URL

    return None


def build_page_url(page_id: str) -> str:
    """构建页面 URL"""
    return f"{_wiki_base_url}/pages/viewpage.action?pageId={page_id}"


def build_create_url(parent_id: str, space_key: str = "~lijunyi3") -> str:
    """构建创建页面 URL"""
    return f"{_wiki_base_url}/pages/createpage.action?spaceKey={space_key}&fromPageId={parent_id}"


def build_edit_url(page_id: str) -> str:
    """构建编辑页面 URL"""
    return f"{_wiki_base_url}/pages/editpage.action?pageId={page_id}"


def build_search_url(query: str, space_key: str = "engineer") -> str:
    """构建搜索 URL"""
    encoded_query = quote(query)
    return f"{_wiki_base_url}/dosearchsite.action?queryString={encoded_query}&spaceKey={space_key}"


def build_children_url(page_id: str) -> str:
    """构建子页面列表 URL"""
    return f"{_wiki_base_url}/pages/listchildren.action?pageId={page_id}"


# ============== MCP Tools ==============


@mcp.tool()
async def wiki_search(query: str, space_key: str = "engineer", limit: int = 10) -> str:
    """
    搜索 Wiki 页面

    Args:
        query: 搜索关键词
        space_key: 空间 Key，默认 "engineer"
        limit: 返回结果数量限制，默认 10

    Returns:
        JSON 格式的搜索结果列表
    """
    page = await ensure_browser()

    try:
        search_url = build_search_url(query, space_key)
        logger.info(f"搜索 Wiki: {query}")

        await page.goto(search_url)
        await page.wait_for_load_state("networkidle")

        # 等待搜索结果加载
        await page.wait_for_selector(".search-results", timeout=10000)

        # 提取搜索结果
        results = await page.evaluate(
            f"""
            () => {{
                const items = [];
                const resultElements = document.querySelectorAll('.search-result');
                resultElements.forEach((el, index) => {{
                    if (index >= {limit}) return;

                    const titleEl = el.querySelector('.search-result-title a');
                    const descEl = el.querySelector('.search-result-description');
                    const spaceEl = el.querySelector('.search-result-space');

                    if (titleEl) {{
                        items.push({{
                            title: titleEl.textContent.trim(),
                            url: titleEl.href,
                            description: descEl ? descEl.textContent.trim() : '',
                            space: spaceEl ? spaceEl.textContent.trim() : ''
                        }});
                    }}
                }});
                return items;
            }}
        """
        )

        # 添加 pageId
        for result in results:
            result["page_id"] = extract_page_id(result["url"])

        return json.dumps(
            {
                "success": True,
                "query": query,
                "count": len(results),
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )

    except Exception as e:
        logger.error(f"搜索失败: {e}")
        return json.dumps({"success": False, "error": str(e), "query": query}, ensure_ascii=False)


@mcp.tool()
async def wiki_read(page_id: str, include_metadata: bool = True) -> str:
    """
    读取 Wiki 页面内容

    Args:
        page_id: 页面 ID
        include_metadata: 是否包含元数据（标题、作者、更新时间等）

    Returns:
        JSON 格式的页面内容
    """
    try:
        logger.info(f"[TOOL] wiki_read 被调用: page_id={page_id}")
        page = await ensure_browser()
        logger.info(f"[TOOL] 浏览器已就绪")

        page_url = build_page_url(page_id)
        logger.info(f"[TOOL] 导航到页面: {page_url}")

        await page.goto(page_url)
        logger.info(f"[TOOL] 页面导航完成，等待加载...")

        await page.wait_for_load_state("networkidle", timeout=30000)
        logger.info(f"[TOOL] 页面加载完成")

        # 等待内容加载
        logger.info(f"[TOOL] 等待内容选择器...")
        await page.wait_for_selector("#main-content, .wiki-content", timeout=10000)
        logger.info(f"[TOOL] 内容已加载")

        # 提取内容
        result = await page.evaluate("""
            () => {
                const result = {
                    title: '',
                    content: '',
                    html: '',
                    metadata: {}
                };

                // 获取标题
                const titleEl = document.querySelector('#title-text, h1#title-text, .pagetitle');
                if (titleEl) {
                    result.title = titleEl.textContent.trim();
                }

                // 获取内容
                const contentEl = document.querySelector('#main-content, .wiki-content, #content');
                if (contentEl) {
                    result.html = contentEl.innerHTML;
                    result.content = contentEl.textContent.trim();
                }

                // 获取元数据
                const authorEl = document.querySelector('.author, .creator, [data-username]');
                if (authorEl) {
                    result.metadata.author = authorEl.textContent.trim();
                }

                const dateEl = document.querySelector('.date, .last-modified, [data-date]');
                if (dateEl) {
                    result.metadata.last_modified = dateEl.textContent.trim();
                }

                return result;
            }
        """)

        result["page_id"] = page_id
        result["url"] = page_url

        if not include_metadata:
            result.pop("metadata", None)

        return json.dumps({"success": True, "page": result}, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.error(f"读取页面失败: {e}")
        return json.dumps(
            {"success": False, "error": str(e), "page_id": page_id}, ensure_ascii=False
        )


@mcp.tool()
async def wiki_list_children(page_id: str, recursive: bool = False) -> str:
    """
    列出页面的子页面

    Args:
        page_id: 父页面 ID
        recursive: 是否递归列出所有子页面，默认 False

    Returns:
        JSON 格式的子页面列表
    """
    page = await ensure_browser()

    try:
        children_url = build_children_url(page_id)
        logger.info(f"列出子页面: {page_id}")

        await page.goto(children_url)
        await page.wait_for_load_state("networkidle")

        # 提取子页面列表
        children = await page.evaluate("""
            () => {
                const items = [];
                const rows = document.querySelectorAll('table tbody tr, .children-list li, .page-list li');

                rows.forEach(row => {
                    const linkEl = row.querySelector('a[href*="pageId="], a[href*="/display/"]');
                    if (linkEl) {
                        const title = linkEl.textContent.trim();
                        const href = linkEl.href;

                        // 提取 pageId
                        let childPageId = null;
                        const match = href.match(/pageId=(\\d+)/);
                        if (match) {
                            childPageId = match[1];
                        }

                        items.push({
                            title: title,
                            url: href,
                            page_id: childPageId
                        });
                    }
                });

                return items;
            }
        """)

        # 如果需要递归
        if recursive:
            all_children = []
            for child in children:
                all_children.append(child)
                if child.get("page_id"):
                    # 递归获取子页面
                    child_result = await wiki_list_children(child["page_id"], recursive=True)
                    child_data = json.loads(child_result)
                    if child_data.get("success"):
                        child["children"] = child_data.get("children", [])

        return json.dumps(
            {
                "success": True,
                "parent_id": page_id,
                "count": len(children),
                "children": children,
            },
            ensure_ascii=False,
            indent=2,
        )

    except Exception as e:
        logger.error(f"列出子页面失败: {e}")
        return json.dumps(
            {"success": False, "error": str(e), "parent_id": page_id},
            ensure_ascii=False,
        )


@mcp.tool()
async def wiki_create(parent_id: str, title: str, content: str, space_key: str = "~lijunyi3") -> str:
    """
    在指定父页面下创建子页面

    Args:
        parent_id: 父页面 ID
        title: 新页面标题
        content: 页面内容（HTML 格式）
        space_key: 空间 Key，默认 "engineer"

    Returns:
        JSON 格式的创建结果
    """
    page = await ensure_browser()

    try:
        create_url = build_create_url(parent_id, space_key)
        logger.info(f"创建页面: '{title}' 在父页面 {parent_id} 下")

        # 导航到创建页面
        await page.goto(create_url)
        await page.wait_for_load_state("networkidle")

        # 等待编辑器加载
        await page.wait_for_selector(
            "input[name='title'], #content-title, [placeholder*='标题']", timeout=10000
        )

        # 填写标题
        title_input = await page.query_selector(
            "input[name='title'], #content-title, [placeholder*='标题']"
        )
        if title_input:
            await title_input.fill(title)
        else:
            # 尝试通过 label 查找
            await page.get_by_label("标题").fill(title)

        await asyncio.sleep(0.5)

        # 填写内容 - 通过 TinyMCE iframe
        # 尝试多种方式设置内容
        content_set = await page.evaluate(
            """
            (content) => {
                // 方式1: 直接查找 TinyMCE iframe
                const iframe = document.querySelector('#wysiwygTextarea_ifr, #tinyMCE_ifr, iframe[id*="wysiwyg"]');
                if (iframe && iframe.contentDocument) {
                    const body = iframe.contentDocument.querySelector('body');
                    if (body) {
                        body.innerHTML = content;
                        return 'iframe';
                    }
                }

                // 方式2: 查找 textarea
                const textarea = document.querySelector('textarea[name="content"], textarea#content, #wysiwygTextarea');
                if (textarea) {
                    textarea.value = content;
                    textarea.dispatchEvent(new Event('input', { bubbles: true }));
                    return 'textarea';
                }

                // 方式3: 使用 TinyMCE API
                if (typeof tinyMCE !== 'undefined') {
                    const editor = tinyMCE.get('wysiwygTextarea') || tinyMCE.get(0);
                    if (editor) {
                        editor.setContent(content);
                        return 'tinymce_api';
                    }
                }

                // 方式4: 查找 contenteditable 元素
                const editable = document.querySelector('[contenteditable="true"]');
                if (editable) {
                    editable.innerHTML = content;
                    return 'contenteditable';
                }

                return null;
            }
        """,
            content,
        )

        if not content_set:
            return json.dumps(
                {
                    "success": False,
                    "error": "无法找到编辑器元素",
                    "parent_id": parent_id,
                    "title": title,
                },
                ensure_ascii=False,
            )

        logger.info(f"内容设置方式: {content_set}")
        await asyncio.sleep(1)

        # 点击发布按钮
        publish_button = await page.query_selector(
            "button[type='submit'], input[type='submit'], button:has-text('发布'), button:has-text('保存')"
        )
        if publish_button:
            await publish_button.click()
        else:
            # 尝试通过文本查找
            try:
                await page.get_by_role("button", name="发布").click()
            except Exception:
                await page.get_by_role("button", name="保存").click()

        # 等待页面保存完成
        await asyncio.sleep(3)
        await page.wait_for_load_state("networkidle")

        # 获取新页面 URL
        current_url = page.url
        new_page_id = extract_page_id(current_url)

        return json.dumps(
            {
                "success": True,
                "message": "页面创建成功",
                "title": title,
                "parent_id": parent_id,
                "page_id": new_page_id,
                "url": current_url,
            },
            ensure_ascii=False,
            indent=2,
        )

    except Exception as e:
        logger.error(f"创建页面失败: {e}")
        return json.dumps(
            {"success": False, "error": str(e), "parent_id": parent_id, "title": title},
            ensure_ascii=False,
        )


@mcp.tool()
async def wiki_update(page_id: str, title: str | None = None, content: str | None = None) -> str:
    """
    更新 Wiki 页面内容

    Args:
        page_id: 页面 ID
        title: 新标题（可选，不提供则保持原样）
        content: 新内容（HTML 格式，可选，不提供则保持原样）

    Returns:
        JSON 格式的更新结果
    """
    page = await ensure_browser()

    try:
        edit_url = build_edit_url(page_id)
        logger.info(f"更新页面: {page_id}")

        # 导航到编辑页面
        await page.goto(edit_url)
        await page.wait_for_load_state("networkidle")

        # 等待编辑器加载
        await page.wait_for_selector("input[name='title'], #content-title", timeout=10000)

        # 更新标题（如果提供）
        if title:
            title_input = await page.query_selector("input[name='title'], #content-title")
            if title_input:
                await title_input.fill(title)
            await asyncio.sleep(0.5)

        # 更新内容（如果提供）
        if content:
            await page.evaluate(
                """
                (content) => {
                    // 方式1: 直接查找 TinyMCE iframe
                    const iframe = document.querySelector('#wysiwygTextarea_ifr, #tinyMCE_ifr, iframe[id*="wysiwyg"]');
                    if (iframe && iframe.contentDocument) {
                        const body = iframe.contentDocument.querySelector('body');
                        if (body) {
                            body.innerHTML = content;
                            return 'iframe';
                        }
                    }

                    // 方式2: 查找 textarea
                    const textarea = document.querySelector('textarea[name="content"], textarea#content, #wysiwygTextarea');
                    if (textarea) {
                        textarea.value = content;
                        textarea.dispatchEvent(new Event('input', { bubbles: true }));
                        return 'textarea';
                    }

                    // 方式3: 使用 TinyMCE API
                    if (typeof tinyMCE !== 'undefined') {
                        const editor = tinyMCE.get('wysiwygTextarea') || tinyMCE.get(0);
                        if (editor) {
                            editor.setContent(content);
                            return 'tinymce_api';
                        }
                    }

                    // 方式4: 查找 contenteditable 元素
                    const editable = document.querySelector('[contenteditable="true"]');
                    if (editable) {
                        editable.innerHTML = content;
                        return 'contenteditable';
                    }

                    return null;
                }
            """,
                content,
            )
            await asyncio.sleep(1)

        # 点击保存按钮
        save_button = await page.query_selector(
            "button[type='submit'], input[type='submit'], button:has-text('保存'), button:has-text('更新')"
        )
        if save_button:
            await save_button.click()
        else:
            try:
                await page.get_by_role("button", name="保存").click()
            except Exception:
                await page.get_by_role("button", name="更新").click()

        # 等待保存完成
        await asyncio.sleep(3)
        await page.wait_for_load_state("networkidle")

        # 获取当前页面 URL
        current_url = page.url

        return json.dumps(
            {
                "success": True,
                "message": "页面更新成功",
                "page_id": page_id,
                "url": current_url,
                "updated_fields": {
                    "title": title is not None,
                    "content": content is not None,
                },
            },
            ensure_ascii=False,
            indent=2,
        )

    except Exception as e:
        logger.error(f"更新页面失败: {e}")
        return json.dumps(
            {"success": False, "error": str(e), "page_id": page_id}, ensure_ascii=False
        )


@mcp.tool()
async def wiki_delete(page_id: str, confirm: bool = False) -> str:
    """
    删除 Wiki 页面（谨慎使用）

    Args:
        page_id: 页面 ID
        confirm: 确认删除，必须设置为 True

    Returns:
        JSON 格式的删除结果
    """
    if not confirm:
        return json.dumps(
            {
                "success": False,
                "error": "删除操作需要设置 confirm=True",
                "page_id": page_id,
            },
            ensure_ascii=False,
        )

    page = await ensure_browser()

    try:
        # 先读取页面信息获取标题
        read_result = await wiki_read(page_id)
        read_data = json.loads(read_result)

        if not read_data.get("success"):
            return json.dumps(
                {"success": False, "error": "无法读取页面信息", "page_id": page_id},
                ensure_ascii=False,
            )

        title = read_data.get("page", {}).get("title", "Unknown")

        # 构建删除 URL
        delete_url = f"{_wiki_base_url}/pages/deletepage.action?pageId={page_id}"
        logger.warning(f"删除页面: {page_id} ({title})")

        await page.goto(delete_url)
        await page.wait_for_load_state("networkidle")

        # 确认删除按钮
        confirm_button = await page.query_selector(
            "button[type='submit'], input[type='submit'], button:has-text('删除'), button:has-text('确认')"
        )
        if confirm_button:
            await confirm_button.click()

        await asyncio.sleep(2)

        return json.dumps(
            {
                "success": True,
                "message": "页面已删除",
                "page_id": page_id,
                "title": title,
            },
            ensure_ascii=False,
            indent=2,
        )

    except Exception as e:
        logger.error(f"删除页面失败: {e}")
        return json.dumps(
            {"success": False, "error": str(e), "page_id": page_id}, ensure_ascii=False
        )


@mcp.tool()
async def wiki_get_spaces() -> str:
    """
    获取可用的 Wiki 空间列表

    Returns:
        JSON 格式的空间列表
    """
    page = await ensure_browser()

    try:
        spaces_url = f"{_wiki_base_url}/spaces.action"
        logger.info("获取空间列表")

        await page.goto(spaces_url)
        await page.wait_for_load_state("networkidle")

        # 提取空间列表
        spaces = await page.evaluate("""
            () => {
                const items = [];
                const rows = document.querySelectorAll('table tbody tr, .space-list li, .spaces-list-item');

                rows.forEach(row => {
                    const linkEl = row.querySelector('a[href*="/display/"], a[href*="/spaces/"]');
                    if (linkEl) {
                        const name = linkEl.textContent.trim();
                        const href = linkEl.href;

                        // 提取 space key
                        let space_key = null;
                        const match = href.match(/\\/display\\/([^\\/]+)/);
                        if (match) {
                            space_key = match[1];
                        }

                        items.push({
                            name: name,
                            key: space_key,
                            url: href
                        });
                    }
                });

                return items;
            }
        """)

        return json.dumps(
            {"success": True, "count": len(spaces), "spaces": spaces},
            ensure_ascii=False,
            indent=2,
        )

    except Exception as e:
        logger.error(f"获取空间列表失败: {e}")
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


@mcp.tool()
async def wiki_configure(
    base_url: str | None = None,
    cdp_endpoint: str | None = None,
    space_key: str | None = None,
) -> str:
    """
    配置 Wiki MCP Server

    Args:
        base_url: Wiki 基础 URL
        cdp_endpoint: Chrome DevTools Protocol 端点
        space_key: 默认空间 Key

    Returns:
        JSON 格式的配置结果
    """
    global _wiki_base_url, _cdp_endpoint

    try:
        if base_url:
            _wiki_base_url = base_url.rstrip("/")
        if cdp_endpoint:
            _cdp_endpoint = cdp_endpoint
        if space_key:
            # 更新默认空间 Key
            pass  # 这个需要在工具调用时传入

        return json.dumps(
            {
                "success": True,
                "message": "配置已更新",
                "config": {"base_url": _wiki_base_url, "cdp_endpoint": _cdp_endpoint},
            },
            ensure_ascii=False,
            indent=2,
        )

    except Exception as e:
        logger.error(f"配置失败: {e}")
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


# ============== Main Entry Point ==============


def is_port_available(host: str, port: int) -> bool:
    """检查端口是否可用"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            return True
    except OSError:
        return False


def find_available_port(host: str, start_port: int, max_attempts: int = 10) -> int | None:
    """查找可用端口"""
    for i in range(max_attempts):
        port = start_port + i
        if is_port_available(host, port):
            return port
    return None


def main():
    """启动 MCP Server (stdio mode by default)"""
    import argparse

    parser = argparse.ArgumentParser(description="Wiki MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="sse",
        help="Transport type (default: sse)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port for SSE mode (default: 8000 or auto-find if busy)",
    )

    args = parser.parse_args()

    if args.transport == "sse":
        host = os.getenv("FASTMCP_HOST", "127.0.0.1")
        default_port = args.port or int(os.getenv("FASTMCP_PORT", "8000"))

        # 检查端口是否可用
        if not is_port_available(host, default_port):
            logger.warning(f"端口 {default_port} 已被占用，尝试查找其他可用端口...")
            available_port = find_available_port(host, default_port + 1)

            if available_port is None:
                logger.error(f"无法找到可用端口 (尝试范围: {default_port + 1}-{default_port + 10})")
                logger.error("请手动指定其他端口: --port <port_number>")
                sys.exit(1)

            logger.info(f"找到可用端口: {available_port}")
            os.environ["FASTMCP_PORT"] = str(available_port)
        else:
            os.environ["FASTMCP_PORT"] = str(default_port)

        port = int(os.environ["FASTMCP_PORT"])
        mcp.settings.port = port
        mcp.settings.host = host

        # 保存端口信息到文件，供客户端读取
        port_info_file = Path.home() / ".wiki_agent" / "mcp_server_port"
        port_info_file.parent.mkdir(parents=True, exist_ok=True)
        port_info_file.write_text(f"{host}:{port}")

        logger.info(f"启动 Wiki MCP Server (SSE mode) on {host}:{port}")

        try:
            mcp.run(transport="sse")
        except Exception as e:
            logger.error(f"服务器启动失败: {e}")
            logger.error("如果端口冲突，请尝试: --port <other_port>")
            sys.exit(1)
    else:
        logger.info("启动 Wiki MCP Server (stdio mode)...")
        try:
            mcp.run()
        except Exception as e:
            logger.error(f"服务器启动失败: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
