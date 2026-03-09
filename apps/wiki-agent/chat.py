#!/usr/bin/env python3
"""
Wiki Agent 交互式对话模式
"""

import asyncio
import sys
import os
import traceback
from typing import Optional

sys.path.insert(0, "src")

from dotenv import load_dotenv

load_dotenv()

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from loguru import logger
from src.config import get_config
from src.wiki_agent import WikiAgent
from src.wiki_mcp_client import WikiMCPClient

console = Console()


def setup_logging():
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    )


def create_client(config) -> Optional[WikiMCPClient]:
    try:
        return WikiMCPClient(
            server_path=config.mcp_server_path,
            transport=config.mcp_transport,
            sse_host=config.mcp_sse_host,
            sse_port=config.mcp_sse_port,
            sse_url=config.mcp_sse_url,
        )
    except Exception as e:
        console.print(f"[red]创建 MCP 客户端失败: {e}[/red]")
        return None


async def initialize_agent(config) -> Optional[WikiAgent]:
    mcp_client = create_client(config)
    if not mcp_client:
        return None

    wiki_agent = WikiAgent(mcp_client=mcp_client)

    try:
        console.print("[dim]正在初始化 Agent...[/dim]")
        await wiki_agent.initialize()
        return wiki_agent
    except Exception as e:
        console.print(f"[red]Agent 初始化失败: {e}[/red]")
        await wiki_agent.close()
        return None


async def handle_streaming_response(wiki_agent: WikiAgent, query: str) -> tuple[bool, str]:
    full_response = []
    in_tool_call = False
    has_error = False

    try:
        async for chunk in wiki_agent.run_stream(query):
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
                console.print(f"[dim cyan][使用工具: {tool_name}][/dim cyan] ", end="")
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
                error_msg = chunk.get("error", "未知错误")
                console.print(f"\n[red]执行错误: {error_msg}[/red]")
                has_error = True

        console.print()
        return not has_error, "".join(full_response)

    except Exception as e:
        console.print(f"\n[red]流式输出异常: {e}[/red]")
        if os.getenv("WIKI_DEBUG"):
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
        return False, "".join(full_response)


async def chat():
    setup_logging()

    try:
        config = get_config()
    except Exception as e:
        console.print(f"[red]配置加载失败: {e}[/red]")
        console.print("[yellow]请检查 .env 文件是否存在且格式正确[/yellow]")
        return

    if not config.effective_openai_api_key:
        console.print(
            Panel(
                "[red]错误: 未设置 OPENAI_API_KEY[/red]\n\n"
                "[yellow]解决方法:[/yellow]\n"
                "1. 设置环境变量: [cyan]export OPENAI_API_KEY=your_key[/cyan]\n"
                "2. 或编辑 .env 文件添加 OPENAI_API_KEY",
                title="配置错误",
                border_style="red",
            )
        )
        return

    console.print("\n[bold green]🤖 Wiki Agent 对话模式[/bold green]")
    console.print("[dim]支持实时流式输出 | 输入 'exit' 或 'quit' 退出\n")

    wiki_agent = await initialize_agent(config)
    if not wiki_agent:
        console.print("[red]Agent 初始化失败，无法启动对话[/red]")
        return

    console.print("[green]✓ Agent 初始化完成\n")

    history = []
    session_error_count = 0
    max_session_errors = 3

    try:
        while True:
            if session_error_count >= max_session_errors:
                console.print(
                    f"\n[red]连续错误次数过多 ({session_error_count})，建议重启服务[/red]"
                )
                break

            try:
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

                if user_input.lower() in ("status", "状态"):
                    console.print(
                        f"[dim]历史消息数: {len(history)} | 当前会话错误: {session_error_count}[/dim]\n"
                    )
                    continue

                if user_input.lower() in ("help", "帮助", "?"):
                    console.print("""
[bold]可用命令:[/bold]
  [cyan]exit/quit/退出[/cyan] - 退出对话
  [cyan]clear/清空[/cyan]    - 清空对话历史
  [cyan]status/状态[/cyan]   - 显示会话状态
  [cyan]help/帮助/?[/cyan]   - 显示帮助信息

[bold]使用提示:[/bold]
  - 直接输入问题或任务描述
  - Agent 会自动决定使用哪些工具
  - 支持流式实时输出
""")
                    continue

                console.print("[bold green]AI:[/bold green] ", end="")

                full_query = "\n".join([f"用户: {q}\nAI: {a}" for q, a in history])
                if full_query:
                    full_query += f"\n用户: {user_input}"
                else:
                    full_query = user_input

                success, response_text = await handle_streaming_response(wiki_agent, full_query)

                if success and response_text:
                    history.append((user_input, response_text))
                    session_error_count = 0

                    if len(history) > 10:
                        history = history[-10:]
                elif not success:
                    session_error_count += 1
                    if response_text:
                        history.append((user_input, response_text))

            except KeyboardInterrupt:
                console.print("\n[yellow]\n检测到中断，输入 exit 退出或继续对话[/yellow]\n")
                continue
            except Exception as e:
                session_error_count += 1
                console.print(f"\n[red]输入处理错误: {e}[/red]")
                if os.getenv("WIKI_DEBUG"):
                    console.print(f"[dim]{traceback.format_exc()}[/dim]")
                console.print()

    except KeyboardInterrupt:
        console.print("\n\n[yellow]用户中断，正在退出...[/yellow]")
    except Exception as e:
        console.print(f"\n[red]会话异常: {e}[/red]")
        if os.getenv("WIKI_DEBUG"):
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
    finally:
        try:
            await wiki_agent.close()
            console.print("[dim]Agent 资源已释放[/dim]")
        except Exception as e:
            console.print(f"[dim]关闭时出错: {e}[/dim]")


if __name__ == "__main__":
    try:
        asyncio.run(chat())
    except KeyboardInterrupt:
        console.print("\n[yellow]程序已终止[/yellow]")
    except Exception as e:
        console.print(f"\n[red]程序异常退出: {e}[/red]")
        sys.exit(1)
