"""Rich-based console UI for interactive CLI coder with streaming support."""

import json
from collections.abc import AsyncGenerator

from rich.align import Align
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.style import Style
from rich.syntax import Syntax
from rich.text import Text


class StreamingPanel:
    """A panel that can be updated during streaming."""

    def __init__(self, console: Console, title: str, border_style: str):
        self.console = console
        self.title = title
        self.border_style = border_style
        self.content = ""
        self.live = None

    def start(self):
        """Start the live display."""
        self.live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=15,
            auto_refresh=True,
        )
        self.live.start()

    def update(self, new_content: str):
        """Update the content."""
        self.content = new_content
        if self.live:
            self.live.update(self._render())

    def stop(self):
        """Stop the live display."""
        if self.live:
            self.live.stop()

    def _render(self):
        """Render the panel."""
        md = Markdown(self.content)
        return Panel(
            md,
            title=self.title,
            border_style=self.border_style,
            title_align="left",
        )


class ChatConsole:
    """Rich console for chat interface with streaming support."""

    def __init__(self):
        self.console = Console()
        self.theme = "monokai"
        self.user_style = Style(color="cyan", bold=True)
        self.assistant_style = Style(color="green", bold=True)
        self.system_style = Style(color="yellow", dim=True)
        self.tool_style = Style(color="magenta", dim=True)
        self._current_streaming = None

    def print_header(self, title: str = "🤖 CLI Agent") -> None:
        """Print the application header."""
        header = Panel(
            Align.center(Text(title, style="bold cyan", justify="center")),
            border_style="cyan",
            subtitle="Type 'help' for commands, 'exit' to quit",
        )
        self.console.print(header)

    def print_message(
        self,
        role: str,
        content: str | None,
        is_streaming: bool = False,
    ) -> None:
        """Print a chat message with appropriate styling."""
        if content is None:
            return

        if role == "user":
            prefix = Text("👤 You: ", style=self.user_style)
            panel = Panel(
                Text(content),
                title=prefix,
                border_style="cyan",
                title_align="left",
            )
        elif role == "assistant":
            prefix = Text("🤖 Agent: ", style=self.assistant_style)
            # Render markdown for assistant messages
            md = Markdown(content)
            panel = Panel(
                md,
                title=prefix,
                border_style="green",
                title_align="left",
            )
        elif role == "system":
            prefix = Text("⚙️ System: ", style=self.system_style)
            panel = Panel(
                Text(content, style="dim"),
                title=prefix,
                border_style="yellow",
                title_align="left",
            )
        elif role == "tool":
            prefix = Text("🔧 Tool: ", style=self.tool_style)
            # Try to format as JSON or coder
            try:
                data = json.loads(content)
                syntax = Syntax(
                    json.dumps(data, indent=2, ensure_ascii=False),
                    "json",
                    theme=self.theme,
                    line_numbers=False,
                )
                panel = Panel(
                    syntax,
                    title=prefix,
                    border_style="magenta",
                    title_align="left",
                )
            except (json.JSONDecodeError, ValueError):
                panel = Panel(
                    Text(content, style="dim"),
                    title=prefix,
                    border_style="magenta",
                    title_align="left",
                )
        else:
            panel = Panel(Text(content))

        self.console.print(panel)

    def print_tool_call(self, tool_name: str, arguments: str) -> None:
        """Print a tool call notification."""
        try:
            args = json.loads(arguments)
            args_str = json.dumps(args, indent=2, ensure_ascii=False)
        except (json.JSONDecodeError, ValueError):
            args_str = arguments

        syntax = Syntax(args_str, "json", theme=self.theme, line_numbers=False)

        panel = Panel(
            syntax,
            title=f"🔧 Calling tool: [bold magenta]{tool_name}[/bold magenta]",
            border_style="magenta",
            title_align="left",
        )
        self.console.print(panel)

    async def stream_assistant_response(
        self,
        generator: AsyncGenerator[str, None],
    ) -> tuple[str, list[dict]]:
        """Stream assistant response with real-time display.

        Returns the complete content and any tool calls detected.
        """
        full_content = ""
        tool_calls = []
        current_tool = None

        # Create streaming panel
        prefix = Text("🤖 Agent: ", style=self.assistant_style)
        panel = StreamingPanel(
            self.console,
            str(prefix),
            "green",
        )
        panel.start()

        try:
            async for chunk in generator:
                # Check for tool call markers
                if chunk.startswith("__TOOL_CALL_START__:"):
                    tool_name = chunk.split(":", 1)[1]
                    current_tool = {
                        "id": f"call_{len(tool_calls)}",
                        "type": "function",
                        "function": {"name": tool_name, "arguments": ""},
                    }
                elif chunk.startswith("__TOOL_CALL_ARGS__:"):
                    args = chunk.split(":", 1)[1]
                    if current_tool:
                        current_tool["function"]["arguments"] += args
                elif current_tool and not chunk.startswith("__"):
                    # Collect tool arguments
                    current_tool["function"]["arguments"] += chunk
                    # Check if we have complete JSON
                    try:
                        json.loads(current_tool["function"]["arguments"])
                        tool_calls.append(current_tool)
                        current_tool = None
                    except json.JSONDecodeError:
                        pass
                else:
                    # Regular content
                    full_content += chunk
                    panel.update(full_content)

            # Don't forget the last tool call if we have one
            if current_tool:
                tool_calls.append(current_tool)

        finally:
            panel.stop()

        # Content already displayed during streaming, no need to print again
        return full_content, tool_calls

    async def stream_response(
        self,
        generator: AsyncGenerator[str, None],
        role: str = "assistant",
    ) -> str:
        """Legacy: Stream a response with live updates."""
        content, _ = await self.stream_assistant_response(generator)
        return content

    def print_thinking(self) -> "Progress":
        """Show a thinking indicator, returns Progress to stop later."""
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]Thinking..."),
            console=self.console,
            transient=True,
        )
        progress.add_task("thinking", total=None)
        return progress

    def print_error(self, message: str) -> None:
        """Print an error message."""
        panel = Panel(
            Text(message, style="bold red"),
            title="❌ Error",
            border_style="red",
        )
        self.console.print(panel)

    def print_success(self, message: str) -> None:
        """Print a success message."""
        panel = Panel(
            Text(message, style="bold green"),
            title="✅ Success",
            border_style="green",
        )
        self.console.print(panel)

    def print_info(self, message: str) -> None:
        """Print an info message."""
        self.console.print(f"[dim]{message}[/dim]")

    def print_help(self) -> None:
        """Print help information."""
        help_text = """
# Available Commands

## Conversation
- **exit** / **quit** - Exit the application
- **clear** - Clear the screen
- **history** - Show conversation history
- **new** - Start a new conversation session
- **sessions** - List all saved sessions
- **load <session_id>** - Load a saved session
- **save** - Save current session manually

## Tools
- **tools** - List available tools
- **help** - Show this help message

## Tips
- Use natural language to chat with the AI
- The AI can use tools when needed
- Your conversation is automatically saved
- Type commands without any prefix
        """
        self.console.print(Markdown(help_text))

    def get_input(self, prompt: str = "👤 You") -> str:
        """Get user input."""
        return Prompt.ask(f"[bold cyan]{prompt}[/bold cyan]")

    def clear(self) -> None:
        """Clear the console."""
        self.console.clear()

    def print_sessions(self, sessions: list) -> None:
        """Print list of saved sessions."""

        if not sessions:
            self.console.print("[dim]No saved sessions found.[/dim]")
            return

        table_data = []
        for session in sessions:
            created = session.created_at.strftime("%Y-%m-%d %H:%M")
            updated = session.updated_at.strftime("%Y-%m-%d %H:%M")
            msg_count = len(session.messages)
            table_data.append((session.session_id, session.title, created, updated, str(msg_count)))

        from rich.table import Table

        table = Table(title="Saved Sessions", border_style="cyan")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Title", style="green")
        table.add_column("Created", style="dim")
        table.add_column("Updated", style="dim")
        table.add_column("Messages", style="magenta", justify="right")

        for row in table_data:
            table.add_row(*row)

        self.console.print(table)

    def print_tools(self, tools: list) -> None:
        """Print list of available tools."""
        if not tools:
            self.console.print("[dim]No tools available.[/dim]")
            return

        from rich.table import Table

        table = Table(title="Available Tools", border_style="magenta")
        table.add_column("Name", style="magenta", no_wrap=True)
        table.add_column("Description", style="green")

        for tool in tools:
            table.add_row(tool.name, tool.description)

        self.console.print(table)
