"""
Wiki Agent CLI
Wiki Agent 命令行接口
"""

import asyncio
import json
import os
import sys

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config import create_env_template, get_config
from src.wiki_agent import WikiAgent
from src.wiki_mcp_client import WikiMCPClient

app = typer.Typer(help="Wiki Agent - Wiki 文档管理助手")
console = Console()


def setup_logging(log_level: str = "INFO"):
    """设置日志"""
    logger.remove()
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )


@app.callback()
def callback(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="启用详细日志"),
):
    """Wiki Agent CLI"""
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(log_level)


@app.command()
def init():
    """初始化环境变量配置文件"""
    create_env_template()
    console.print("[green]✓[/green] 环境变量模板已创建 (.env)")
    console.print("[yellow]提示:[/yellow] 请编辑 .env 文件，设置 OPENAI_API_KEY 和其他配置")


def create_client(config) -> WikiMCPClient:
    """创建 WikiMCPClient (根据配置选择 transport)"""
    return WikiMCPClient(
        server_path=config.mcp_server_path,
        transport=config.mcp_transport,
        sse_host=config.mcp_sse_host,
        sse_port=config.mcp_sse_port,
        sse_url=config.mcp_sse_url,
    )


@app.command()
def search(
    query: str = typer.Argument(..., help="搜索关键词"),
    space: str = typer.Option("engineer", "--space", "-s", help="空间 Key"),
    limit: int = typer.Option(10, "--limit", "-l", help="结果数量限制"),
):
    """搜索 Wiki 页面"""

    async def _search():
        config = get_config()
        client = create_client(config)

        try:
            result = await client.search(query, space, limit)
            data = json.loads(result)

            if data.get("success"):
                console.print(f"[green]找到 {data.get('count', 0)} 个结果[/green]")

                table = Table(title=f"搜索结果: {query}")
                table.add_column("标题", style="cyan")
                table.add_column("页面 ID", style="magenta")
                table.add_column("空间", style="green")

                for item in data.get("results", []):
                    table.add_row(
                        item.get("title", ""),
                        item.get("page_id", ""),
                        item.get("space", ""),
                    )

                console.print(table)
            else:
                console.print(f"[red]搜索失败: {data.get('error')}[/red]")

        finally:
            await client.close()

    asyncio.run(_search())


@app.command()
def read(
    page_id: str = typer.Argument(..., help="页面 ID"),
    metadata: bool = typer.Option(True, "--metadata/--no-metadata", help="显示元数据"),
):
    """读取 Wiki 页面内容"""

    async def _read():
        config = get_config()
        client = create_client(config)

        try:
            result = await client.read(page_id, metadata)
            data = json.loads(result)

            if data.get("success"):
                page = data.get("page", {})

                console.print(
                    Panel(
                        page.get("title", "无标题"),
                        title="页面标题",
                        border_style="blue",
                    )
                )

                if metadata and page.get("metadata"):
                    meta = page.get("metadata", {})
                    meta_text = " | ".join([f"{k}: {v}" for k, v in meta.items()])
                    console.print(f"[dim]{meta_text}[/dim]")

                content = page.get("content", "")
                if len(content) > 1000:
                    content = content[:1000] + "\n... [内容已截断]"

                console.print(Panel(content, title="内容", border_style="green"))
                console.print(f"[blue]URL: {page.get('url', '')}[/blue]")
            else:
                console.print(f"[red]读取失败: {data.get('error')}[/red]")

        finally:
            await client.close()

    asyncio.run(_read())


@app.command()
def create(
    parent_id: str = typer.Argument(..., help="父页面 ID"),
    title: str = typer.Argument(..., help="页面标题"),
    content: str = typer.Argument(..., help="页面内容（HTML）"),
    space: str = typer.Option("engineer", "--space", "-s", help="空间 Key"),
):
    """创建 Wiki 页面"""

    async def _create():
        config = get_config()
        client = create_client(config)

        try:
            result = await client.create(parent_id, title, content, space)
            data = json.loads(result)

            if data.get("success"):
                console.print("[green]✓ 页面创建成功[/green]")
                console.print(f"标题: {data.get('title')}")
                console.print(f"页面 ID: {data.get('page_id')}")
                console.print(f"URL: {data.get('url')}")
            else:
                console.print(f"[red]创建失败: {data.get('error')}[/red]")

        finally:
            await client.close()

    asyncio.run(_create())


@app.command()
def update(
    page_id: str = typer.Argument(..., help="页面 ID"),
    title: str | None = typer.Option(None, "--title", "-t", help="新标题"),
    content: str | None = typer.Option(None, "--content", "-c", help="新内容（HTML）"),
):
    """更新 Wiki 页面"""

    async def _update():
        if not title and not content:
            console.print("[red]错误: 必须提供 --title 或 --content[/red]")
            return

        config = get_config()
        client = create_client(config)

        try:
            result = await client.update(page_id, title, content)
            data = json.loads(result)

            if data.get("success"):
                console.print("[green]✓ 页面更新成功[/green]")
                console.print(f"URL: {data.get('url')}")
            else:
                console.print(f"[red]更新失败: {data.get('error')}[/red]")

        finally:
            await client.close()

    asyncio.run(_update())


@app.command()
def list_children(
    page_id: str = typer.Argument(..., help="父页面 ID"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="递归列出"),
):
    """列出子页面"""

    async def _list():
        config = get_config()
        client = create_client(config)

        try:
            result = await client.list_children(page_id, recursive)
            data = json.loads(result)

            if data.get("success"):
                console.print(f"[green]找到 {data.get('count', 0)} 个子页面[/green]")

                def print_children(children, indent=0):
                    for child in children:
                        prefix = "  " * indent
                        console.print(
                            f"{prefix}[cyan]{child.get('title', '')}[/cyan] [dim]({child.get('page_id', '')})[/dim]"
                        )
                        if child.get("children"):
                            print_children(child["children"], indent + 1)

                print_children(data.get("children", []))
            else:
                console.print(f"[red]列出失败: {data.get('error')}[/red]")

        finally:
            await client.close()

    asyncio.run(_list())


@app.command()
def agent(
    query: str = typer.Argument(..., help="自然语言查询"),
):
    """使用 Agent 模式执行复杂任务"""

    async def _agent():
        config = get_config()

        if not config.effective_openai_api_key:
            console.print("[red]错误: 未设置 OPENAI_API_KEY[/red]")
            console.print("[yellow]请设置环境变量或编辑 .env 文件[/yellow]")
            return

        mcp_client = create_client(config)
        wiki_agent = WikiAgent(mcp_client=mcp_client)

        try:
            console.print(f"[blue]正在执行: {query}[/blue]")
            console.print("-" * 50)

            result = await wiki_agent.run(query)

            console.print("-" * 50)

            if result.get("success"):
                console.print("[green]✓ 任务完成[/green]")
                console.print(Panel(result.get("output", ""), title="输出", border_style="green"))

                steps = result.get("intermediate_steps", [])
                if steps:
                    console.print(f"\n[dim]执行了 {len(steps)} 个步骤[/dim]")
            else:
                console.print(f"[red]任务失败: {result.get('error')}[/red]")

        finally:
            await wiki_agent.close()

    asyncio.run(_agent())


# ============== Server Commands ==============

server_app = typer.Typer(help="MCP Server 管理")
app.add_typer(server_app, name="server")


@server_app.command("start")
def server_start(
    transport: str = typer.Option(
        "stdio", "--transport", "-t", help="Transport 类型 (stdio 或 sse)"
    ),
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="SSE 主机地址"),
    port: int = typer.Option(8000, "--port", "-p", help="SSE 端口"),
):
    """启动 MCP Server"""
    get_config()

    if transport == "sse":
        os.environ["FASTMCP_HOST"] = host
        os.environ["FASTMCP_PORT"] = str(port)
        console.print(f"[blue]启动 MCP Server (SSE mode) on {host}:{port}...[/blue]")
    else:
        console.print("[blue]启动 MCP Server (stdio mode)...[/blue]")

    from src.wiki_mcp_server import main

    sys.argv = [sys.argv[0], "--transport", transport]
    main()


@server_app.command("run")
def server_run(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="主机地址"),
    port: int = typer.Option(8000, "--port", "-p", help="端口"),
):
    """启动 MCP Server (SSE mode, 独立运行)"""
    os.environ["FASTMCP_HOST"] = host
    os.environ["FASTMCP_PORT"] = str(port)

    console.print(f"[green]启动 MCP Server on {host}:{port}[/green]")
    console.print(f"[dim]URL: http://{host}:{port}/sse[/dim]")
    console.print("[yellow]按 Ctrl+C 停止服务器[/yellow]")

    from wiki_mcp_server import main

    sys.argv = [sys.argv[0], "--transport", "sse"]
    main()


@app.command()
def chat():
    """启动交互式对话模式（支持流式输出）"""

    async def _chat():
        config = get_config()

        if not config.effective_openai_api_key:
            console.print("[red]错误: 未设置 OPENAI_API_KEY[/red]")
            console.print("[yellow]请设置环境变量或编辑 .env 文件[/yellow]")
            return

        mcp_client = create_client(config)
        wiki_agent = WikiAgent(mcp_client=mcp_client)

        try:
            console.print("\n[bold green]🤖 Wiki Agent 对话模式[/bold green]")
            console.print("[dim]支持实时流式输出 | 输入 'exit' 或 'quit' 退出\n")

            await wiki_agent.initialize()
            console.print("[green]✓ Agent 初始化完成\n")

            history = []

            while True:
                user_input = console.input("[bold blue]你:[/bold blue] ")
                user_input = user_input.strip()

                if not user_input:
                    continue

                if user_input.lower() in ("exit", "quit", "退出"):
                    console.print("\n[yellow]再见! 👋[/yellow]")
                    break

                if user_input.lower() in ("clear", "清空"):
                    history.clear()
                    console.print("[dim]对话历史已清空\n")
                    continue

                if user_input.lower() in ("help", "帮助"):
                    console.print("""
[bold]可用命令:[/bold]
  [cyan]exit/quit/退出[/cyan] - 退出对话
  [cyan]clear/清空[/cyan]    - 清空对话历史
  [cyan]help/帮助[/cyan]     - 显示帮助信息

[bold]使用提示:[/bold]
  - 直接输入问题或任务描述
  - Agent 会自动决定使用哪些工具
  - 支持流式实时输出
""")
                    continue

                console.print("[bold green]AI:[/bold green] ", end="")

                try:
                    full_query = "\n".join([f"用户: {q}\nAI: {a}" for q, a in history])
                    if full_query:
                        full_query += f"\n用户: {user_input}"
                    else:
                        full_query = user_input

                    full_response = []
                    in_tool_call = False

                    async for chunk in wiki_agent.run_stream(full_query):
                        chunk_type = chunk.get("type")

                        if chunk_type == "content":
                            if in_tool_call:
                                console.print()
                                in_tool_call = False
                            content = chunk.get("content", "")
                            console.print(content, end="")
                            full_response.append(content)

                        elif chunk_type == "tool_start":
                            if not in_tool_call:
                                console.print()
                            tool_name = chunk.get("tool_name", "unknown")
                            console.print(f"[dim][使用工具: {tool_name}][/dim] ", end="")
                            in_tool_call = True

                        elif chunk_type == "tool_end":
                            in_tool_call = False

                        elif chunk_type == "complete":
                            if in_tool_call:
                                console.print()
                            tool_count = chunk.get("tool_calls_count", 0)
                            if tool_count > 0:
                                console.print(f"\n[dim][完成: {tool_count} 次工具调用][/dim]")

                        elif chunk_type == "error":
                            console.print(f"\n[red]错误: {chunk.get('error')}[/red]")

                    console.print()

                    response_text = "".join(full_response)
                    history.append((user_input, response_text))

                    if len(history) > 10:
                        history = history[-10:]

                except Exception as e:
                    console.print(f"\n[red]发生错误: {e}[/red]")

                console.print()

        except KeyboardInterrupt:
            console.print("\n\n[yellow]用户中断，正在退出...[/yellow]")
        finally:
            await wiki_agent.close()

    asyncio.run(_chat())


if __name__ == "__main__":
    app()
