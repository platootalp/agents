"""Textual TUI interface for CLI Agent."""

from __future__ import annotations

from typing import Any

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message as TextualMessage
from textual.reactive import reactive
from textual.widgets import (
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    Static,
)
from textual.widgets import Markdown as TextualMarkdown

from cli_agent.core.memory import MemoryManager
from cli_agent.core.provider import Message, OpenAILLM
from cli_agent.core.tools import ToolManager
from cli_agent.tools import get_default_tools


class MessageWidget(Static):
    """Widget to display a chat message."""

    DEFAULT_CSS = """
    MessageWidget {
        padding: 1 2;
        margin: 1 0;
        border: solid $primary;
        border-title-color: $text;
    }
    MessageWidget.user {
        border: solid $accent;
    }
    MessageWidget.assistant {
        border: solid $success;
    }
    MessageWidget.tool {
        border: solid $warning;
    }
    MessageWidget.system {
        border: solid $error;
    }
    """

    def __init__(self, role: str, content: str, **kwargs: Any) -> None:
        self.role = role
        super().__init__(**kwargs)
        self.content = content

    def compose(self) -> ComposeResult:
        border_title = {
            "user": "👤 You",
            "assistant": "🤖 Agent",
            "tool": "🔧 Tool",
            "system": "⚙️ System",
        }.get(self.role, self.role)

        self.border_title = border_title

        if self.role == "user":
            self.add_class("user")
        elif self.role == "assistant":
            self.add_class("assistant")
        elif self.role == "tool":
            self.add_class("tool")
        elif self.role == "system":
            self.add_class("system")

        yield TextualMarkdown(self.content)


class ChatHistory(VerticalScroll):
    """Scrollable chat history container."""

    DEFAULT_CSS = """
    ChatHistory {
        width: 100%;
        height: 1fr;
        padding: 0 1;
    }
    """

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the chat history."""
        message = MessageWidget(role, content)
        self.mount(message)
        self.scroll_end(animate=False)

    def clear(self) -> None:
        """Clear all messages."""
        for child in list(self.children):
            child.remove()


class StatusBar(Static):
    """Status bar showing session info and tool count."""

    DEFAULT_CSS = """
    StatusBar {
        dock: bottom;
        height: auto;
        padding: 0 2;
        background: $surface;
        color: $text-muted;
    }
    StatusBar.horizontal-layout {
        layout: horizontal;
        height: auto;
    }
    StatusBar Label {
        width: 1fr;
    }
    """

    session_name = reactive("New Conversation")
    tool_count = reactive(0)
    status = reactive("Ready")

    def compose(self) -> ComposeResult:
        with Horizontal(classes="horizontal-layout"):
            yield Label(self.session_name, id="session-label")
            yield Label(f"Tools: {self.tool_count}", id="tools-label")
            yield Label(self.status, id="status-label")

    def watch_session_name(self, value: str) -> None:
        try:
            label = self.query_one("#session-label", Label)
            label.update(f"Session: {value}")
        except Exception:
            pass

    def watch_tool_count(self, value: int) -> None:
        try:
            label = self.query_one("#tools-label", Label)
            label.update(f"Tools: {value}")
        except Exception:
            pass

    def watch_status(self, value: str) -> None:
        try:
            label = self.query_one("#status-label", Label)
            label.update(value)
        except Exception:
            pass


class CommandInput(Input):
    """Input widget with command history."""

    DEFAULT_CSS = """
    CommandInput {
        dock: bottom;
        margin: 1 2;
        height: auto;
    }
    """

    BINDINGS = [
        ("up", "history_previous", "Previous"),
        ("down", "history_next", "Next"),
    ]

    def __init__(self, **kwargs: Any) -> None:
        self.history: list[str] = []
        self.history_index = -1
        self.current_input = ""
        super().__init__(placeholder="Type a message or command...", **kwargs)

    def action_history_previous(self) -> None:
        """Navigate to previous history item."""
        if self.history and self.history_index < len(self.history) - 1:
            if self.history_index == -1:
                self.current_input = self.value
            self.history_index += 1
            self.value = self.history[-(self.history_index + 1)]
            self.cursor_position = len(self.value)

    def action_history_next(self) -> None:
        """Navigate to next history item."""
        if self.history_index > 0:
            self.history_index -= 1
            self.value = self.history[-(self.history_index + 1)]
            self.cursor_position = len(self.value)
        elif self.history_index == 0:
            self.history_index = -1
            self.value = self.current_input
            self.cursor_position = len(self.value)

    def add_to_history(self, value: str) -> None:
        """Add a command to history."""
        if value and (not self.history or self.history[-1] != value):
            self.history.append(value)
        self.history_index = -1
        self.current_input = ""


class CLIAgentTUI(App):
    """Textual TUI for CLI Agent."""

    CSS = """
    Screen {
        align: center middle;
    }
    #main-container {
        width: 100%;
        height: 100%;
    }
    #chat-container {
        width: 100%;
        height: 1fr;
    }
    #sidebar {
        width: 30%;
        height: 100%;
        background: $surface;
        border-right: solid $primary;
        padding: 1;
    }
    #sidebar.collapsed {
        display: none;
    }
    #content-area {
        width: 100%;
        height: 100%;
    }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+q", "quit", "Quit"),
        ("f1", "toggle_help", "Help"),
        ("f2", "new_session", "New Session"),
        ("ctrl+l", "clear_chat", "Clear"),
        ("ctrl+s", "save_session", "Save"),
    ]

    def __init__(self, session_id: str | None = None, **kwargs: Any) -> None:
        self.memory = MemoryManager()
        self.tools = ToolManager()
        self.llm: OpenAILLM | None = None
        self.session_id = session_id
        self.max_autonomous_turns = 10
        self.system_prompt = (
            "You are a helpful AI assistant with access to tools. "
            "Be concise and helpful in your responses."
        )
        super().__init__(**kwargs)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Horizontal(id="main-container"):
            with Vertical(id="sidebar"):
                yield Label("Commands", classes="sidebar-title")
                yield ListView(
                    ListItem(Label("F1 - Help")),
                    ListItem(Label("F2 - New Session")),
                    ListItem(Label("Ctrl+S - Save")),
                    ListItem(Label("Ctrl+L - Clear")),
                    ListItem(Label("Ctrl+Q - Quit")),
                )

            with Vertical(id="content-area"):
                yield ChatHistory(id="chat-history")
                yield StatusBar(id="status-bar")
                yield CommandInput(id="command-input")

        yield Footer()

    async def on_mount(self) -> None:
        """Initialize the app."""
        await self.initialize()

        # Load or create session
        if self.session_id:
            session = self.memory.load_session(self.session_id)
            if session:
                self.memory.set_current_session(session)

        if not self.memory.get_current_session():
            self.memory.create_session("New Conversation")

        session = self.memory.get_current_session()
        status_bar = self.query_one("#status-bar", StatusBar)
        status_bar.session_name = session.title
        status_bar.tool_count = len(self.tools.list_tools())

        # Show welcome message
        chat_history = self.query_one("#chat-history", ChatHistory)
        chat_history.add_message(
            "system",
            "Welcome to CLI Agent TUI!\n\nType your messages below or use:\n"
            "- F1 for help\n- F2 for new session\n- Ctrl+Q to quit",
        )

    async def initialize(self) -> bool:
        """Initialize the agent."""
        import os

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            chat_history = self.query_one("#chat-history", ChatHistory)
            chat_history.add_message(
                "system",
                "⚠️ OPENAI_API_KEY not found. Please set it in your environment.",
            )
            return False

        base_url = os.getenv("OPENAI_BASE_URL")
        self.llm = OpenAILLM(api_key=api_key, base_url=base_url)

        default_tools = get_default_tools()
        for tool in default_tools:
            self.tools.register(tool)

        return True

    def _build_messages(self) -> list[Message]:
        """Build message list for LLM."""
        messages = [Message(role="system", content=self.system_prompt)]
        messages.extend(self.memory.get_messages())
        return messages

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle user input."""
        value = event.value.strip()
        if not value:
            return

        # Add to history
        input_widget = self.query_one("#command-input", CommandInput)
        input_widget.add_to_history(value)

        # Handle commands
        if value.lower() in ("exit", "quit", "q"):
            self.exit()
            return

        if value.lower() == "help":
            self.action_toggle_help()
            return

        if value.lower() == "clear":
            self.action_clear_chat()
            return

        if value.lower() == "new":
            self.action_new_session()
            return

        if value.lower().startswith("load "):
            session_id = value[5:].strip()
            session = self.memory.load_session(session_id)
            if session:
                self.memory.set_current_session(session)
                self._refresh_session_display()
                self.query_one("#chat-history", ChatHistory).add_message(
                    "system", f"Loaded session: {session.title}"
                )
            else:
                self.query_one("#chat-history", ChatHistory).add_message(
                    "system", f"Session '{session_id}' not found."
                )
            return

        if value.lower() == "sessions":
            sessions = self.memory.list_sessions()
            if sessions:
                msg = "Saved sessions:\n\n"
                for s in sessions[:10]:  # Show max 10
                    msg += f"- {s.session_id}: {s.title}\n"
                self.query_one("#chat-history", ChatHistory).add_message("system", msg)
            else:
                self.query_one("#chat-history", ChatHistory).add_message(
                    "system", "No saved sessions."
                )
            return

        if value.lower() == "save":
            self.action_save_session()
            return

        if value.lower() == "tools":
            tools = self.tools.list_tools()
            msg = "Available tools:\n\n"
            for tool in tools:
                msg += f"- {tool.name}: {tool.description}\n"
            self.query_one("#chat-history", ChatHistory).add_message("system", msg)
            return

        # Regular chat
        await self._handle_chat(value)

    async def _handle_chat(self, user_input: str) -> None:
        """Handle chat conversation."""
        chat_history = self.query_one("#chat-history", ChatHistory)
        status_bar = self.query_one("#status-bar", StatusBar)

        # Add user message
        chat_history.add_message("user", user_input)
        user_msg = Message(role="user", content=user_input)
        self.memory.add_message(user_msg)

        if not self.llm:
            chat_history.add_message("system", "Error: LLM not initialized")
            return

        status_bar.status = "Thinking..."

        try:
            # Build messages and stream response
            messages = self._build_messages()
            full_response = ""

            # Create assistant message container
            response_widget = MessageWidget("assistant", "")
            await chat_history.mount(response_widget)

            # Stream the response
            generator = self.llm.chat_stream(
                messages=messages,
                tools=self.tools.list_tools(),
            )

            tool_calls = []
            current_tool = None

            async for chunk in generator:
                # Handle tool call markers
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
                    current_tool["function"]["arguments"] += chunk
                    try:
                        import json

                        json.loads(current_tool["function"]["arguments"])
                        tool_calls.append(current_tool)
                        current_tool = None
                    except json.JSONDecodeError:
                        pass
                else:
                    # Regular content
                    full_response += chunk
                    # Update the response widget
                    await response_widget.remove()
                    response_widget = MessageWidget("assistant", full_response)
                    await chat_history.mount(response_widget)
                    chat_history.scroll_end(animate=False)

            if current_tool:
                tool_calls.append(current_tool)

            # Save assistant message to memory
            assistant_msg = Message(
                role="assistant",
                content=full_response if full_response else None,
                tool_calls=tool_calls if tool_calls else None,
            )
            self.memory.add_message(assistant_msg)

            # Handle tool calls
            if tool_calls:
                for tool_call in tool_calls:
                    function_data = tool_call.get("function", {})
                    tool_name = function_data.get("name", "")
                    arguments = function_data.get("arguments", "{}")

                    # Show tool call
                    chat_history.add_message("tool", f"Calling {tool_name} with args: {arguments}")

                    # Execute tool
                    result = await self.tools.execute_tool_call(tool_call)
                    result_str = result.to_string()

                    # Show result
                    chat_history.add_message("tool", f"Result: {result_str}")

                    # Add to memory
                    tool_msg = Message(
                        role="tool",
                        content=result_str,
                        tool_call_id=tool_call.get("id"),
                        name=tool_name,
                    )
                    self.memory.add_message(tool_msg)

            # Save session
            session = self.memory.get_current_session()
            if session:
                self.memory.save_session(session)

            status_bar.status = "Ready"

        except Exception as e:
            chat_history.add_message("system", f"Error: {e}")
            status_bar.status = "Error"

    def _refresh_session_display(self) -> None:
        """Refresh session info in status bar."""
        session = self.memory.get_current_session()
        if session:
            status_bar = self.query_one("#status-bar", StatusBar)
            status_bar.session_name = session.title

    def action_toggle_help(self) -> None:
        """Show help message."""
        help_text = """
# CLI Agent TUI - Help

## Commands
Type these in the input box:
- **help** - Show this help
- **exit/quit/q** - Exit the application
- **clear** - Clear chat history
- **new** - Start a new session
- **save** - Save current session
- **sessions** - List saved sessions
- **load <id>** - Load a session
- **tools** - List available tools

## Keyboard Shortcuts
- **F1** - Show help
- **F2** - New session
- **Ctrl+S** - Save session
- **Ctrl+L** - Clear chat
- **Ctrl+Q** - Quit
- **Up/Down** - Navigate input history

## Usage
Simply type your message and press Enter to chat with the AI.
The AI can use tools when needed to help you.
        """
        self.query_one("#chat-history", ChatHistory).add_message("system", help_text)

    def action_new_session(self) -> None:
        """Create a new session."""
        self.memory.create_session()
        self.query_one("#chat-history", ChatHistory).clear()
        self.query_one("#chat-history", ChatHistory).add_message(
            "system", "Started a new conversation session."
        )
        self._refresh_session_display()

    def action_clear_chat(self) -> None:
        """Clear chat history display."""
        self.query_one("#chat-history", ChatHistory).clear()
        self.query_one("#chat-history", ChatHistory).add_message("system", "Chat history cleared.")

    def action_save_session(self) -> None:
        """Save current session."""
        session = self.memory.get_current_session()
        if session:
            self.memory.save_session(session)
            self.query_one("#chat-history", ChatHistory).add_message(
                "system", f"Session saved: {session.title}"
            )


def run_tui(session_id: str | None = None) -> None:
    """Run the TUI application."""
    app = CLIAgentTUI(session_id=session_id)
    app.run()
