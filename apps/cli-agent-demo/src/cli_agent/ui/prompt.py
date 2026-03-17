"""prompt_toolkit-based input handler with history and completion."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style


class CommandCompleter(Completer):
    """Custom completer for CLI coder commands."""

    COMMANDS = [
        "help",
        "exit",
        "quit",
        "clear",
        "history",
        "new",
        "sessions",
        "load ",
        "save",
        "tools",
    ]

    def __init__(self) -> None:
        self.word_completer = WordCompleter(
            self.COMMANDS,
            ignore_case=True,
            sentence=True,
            match_middle=False,
        )

    def get_completions(self, document, complete_event):
        """Get completions based on current input."""
        text = document.text

        # Only complete at the start of the line
        if not text or text.startswith(("help", "exit", "quit", "clear")):
            yield from self.word_completer.get_completions(document, complete_event)


class PromptHandler:
    """Handle user input with prompt_toolkit features."""

    def __init__(self, history_file: Path | None = None) -> None:
        self.history_file = history_file or Path.home() / ".cli-coder" / "history.txt"
        self.history_file.parent.mkdir(parents=True, exist_ok=True)

        self.style = Style.from_dict(
            {
                "prompt": "bold cyan",
                "command": "bold magenta",
            }
        )

        self.completer = CommandCompleter()
        self.key_bindings = self._create_key_bindings()

        self.session = PromptSession(
            history=FileHistory(str(self.history_file)),
            auto_suggest=AutoSuggestFromHistory(),
            completer=self.completer,
            complete_while_typing=True,
            key_bindings=self.key_bindings,
            style=self.style,
            multiline=False,
            wrap_lines=True,
        )

    def _create_key_bindings(self) -> KeyBindings:
        """Create custom key bindings."""
        bindings = KeyBindings()

        @bindings.add("c-c")
        def _(event):
            """Ctrl+C: Cancel current input or raise KeyboardInterrupt."""
            if event.app.current_buffer.text:
                event.app.current_buffer.reset()
            else:
                event.app.exit(exception=KeyboardInterrupt)

        @bindings.add("c-d")
        def _(event):
            """Ctrl+D: Exit when buffer is empty."""
            if not event.app.current_buffer.text:
                event.app.exit(exception=EOFError)

        @bindings.add("c-l")
        def _(event):
            """Ctrl+L: Clear screen."""
            import os

            os.system("clear" if os.name != "nt" else "cls")

        return bindings

    def get_input(
        self,
        prompt: str = "👤 You",
        rprompt: str | None = None,
        bottom_toolbar: Callable | None = None,
    ) -> str:
        """Get user input with rich features (synchronous)."""
        try:
            result = self.session.prompt(
                HTML(f"<prompt>{prompt}</prompt>: "),
                rprompt=HTML(f" <command>{rprompt}</command>") if rprompt else None,
                bottom_toolbar=bottom_toolbar,
                mouse_support=True,
            )
            return result.strip()
        except KeyboardInterrupt:
            return ""
        except EOFError:
            raise

    async def get_input_async(
        self,
        prompt: str = "👤 You",
        rprompt: str | None = None,
        bottom_toolbar: Callable | None = None,
    ) -> str:
        """Get user input with rich features (asynchronous).

        Use this when already inside an async context to avoid
        'asyncio.run() cannot be called from a running event loop' errors.
        Falls back to standard input if not in a TTY.
        """
        import sys

        # Check if we're in a TTY
        if not sys.stdin.isatty():
            # Fall back to standard input for non-TTY environments
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: input(f"{prompt}: "))

        try:
            result = await self.session.prompt_async(
                HTML(f"<prompt>{prompt}</prompt>: "),
                rprompt=HTML(f" <command>{rprompt}</command>") if rprompt else None,
                bottom_toolbar=bottom_toolbar,
                mouse_support=True,
            )
            return result.strip()
        except KeyboardInterrupt:
            return ""
        except EOFError:
            raise

    def get_multiline_input(
        self,
        prompt: str = "👤 You",
        instruction: str = "(Enter twice to submit, Ctrl+C to cancel)",
    ) -> str:
        """Get multi-line user input.

        Args:
            prompt: The prompt text to display
            instruction: Instructions shown in toolbar

        Returns:
            User input string (possibly multi-line)
        """

        def toolbar():
            return HTML(f"<b><style bg='ansiblue'> {instruction} </style></b>")

        try:
            result = self.session.prompt(
                HTML(f"<prompt>{prompt}</prompt>: "),
                multiline=True,
                bottom_toolbar=toolbar,
                mouse_support=True,
            )
            return result.strip()
        except (KeyboardInterrupt, EOFError):
            return ""
