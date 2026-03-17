"""Main entry point for CLI Agent with Typer + prompt_toolkit + Rich."""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Annotated

import typer
from dotenv import load_dotenv

from cli_agent.core.memory import MemoryManager
from cli_agent.core.provider import Message, OpenAILLM
from cli_agent.core.tools import ToolManager
from cli_agent.tools import get_default_tools
from cli_agent.ui.console import ChatConsole
from cli_agent.ui.prompt import PromptHandler

load_dotenv()

app = typer.Typer(
    name="cli-coder",
    help="Interactive AI assistant with multi-turn streaming and tool calling",
    add_completion=False,
)


class CLIAgent:
    """Interactive CLI Agent with Typer + prompt_toolkit + Rich."""

    def __init__(self) -> None:
        self.console = ChatConsole()
        self.prompt = PromptHandler()
        self.memory = MemoryManager()
        self.tools = ToolManager()
        self.llm: OpenAILLM | None = None
        self.system_prompt = (
            "You are a helpful AI assistant with access to tools. You can use tools when needed "
            "to help the user accomplish their tasks.\n\n"
            "You can engage in multi-turn conversations and execute long-horizon tasks. "
            "When working on complex tasks:\n"
            "1. Break down the task into steps\n"
            "2. Use tools as needed to gather information or perform actions\n"
            "3. Continue autonomously when you have enough context\n"
            "4. Ask the user for clarification when information is missing\n"
            "5. Summarize progress periodically\n\n"
            "Be proactive in completing tasks while keeping the user informed."
        )
        self.max_autonomous_turns = 10

    async def initialize(self) -> bool:
        """Initialize the coder. Returns False if setup fails."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            self.console.print_error(
                "OPENAI_API_KEY not found. Please set it in your environment or .env file."
            )
            return False

        base_url = os.getenv("OPENAI_BASE_URL")
        self.llm = OpenAILLM(api_key=api_key, base_url=base_url)

        default_tools = get_default_tools()
        for tool in default_tools:
            self.tools.register(tool)

        return True

    def _build_messages(self) -> list[Message]:
        """Build message list for LLM from memory."""
        messages = [Message(role="system", content=self.system_prompt)]
        messages.extend(self.memory.get_messages())
        return messages

    async def _stream_and_handle_tools(
        self,
        messages: list[Message],
        is_follow_up: bool = False,
    ) -> tuple[str, list[dict]]:
        """Stream LLM response and handle any tool calls."""
        if not self.llm:
            raise RuntimeError("LLM not initialized")

        generator = self.llm.chat_stream(
            messages=messages,
            tools=self.tools.list_tools(),
        )

        content, tool_calls = await self.console.stream_assistant_response(generator)

        assistant_msg = Message(
            role="assistant",
            content=content if content else None,
            tool_calls=tool_calls if tool_calls else None,
        )
        self.memory.add_message(assistant_msg)

        return content, tool_calls

    async def _execute_tool_calls(self, tool_calls: list[dict]) -> list[Message]:
        """Execute tool calls and return tool response messages."""
        tool_messages = []

        for tool_call in tool_calls:
            function_data = tool_call.get("function", {})
            tool_name = function_data.get("name", "")
            arguments = function_data.get("arguments", "{}")

            self.console.print_tool_call(tool_name, arguments)

            result = await self.tools.execute_tool_call(tool_call)

            result_str = result.to_string()
            self.console.print_message("tool", result_str)

            tool_msg = Message(
                role="tool",
                content=result_str,
                tool_call_id=tool_call.get("id"),
                name=tool_name,
            )
            self.memory.add_message(tool_msg)
            tool_messages.append(tool_msg)

        return tool_messages

    async def _should_continue_autonomously(
        self,
        content: str,
        tool_calls: list[dict],
        turn_count: int,
    ) -> bool:
        """Determine if the coder should continue autonomously."""
        if turn_count >= self.max_autonomous_turns:
            return False

        if tool_calls:
            return True

        if not content or len(content.strip()) < 10:
            return True

        content_lower = content.strip().lower()

        # Check for questions (English and Chinese)
        if "?" in content or "？" in content:
            return False

        # Check for question phrases
        question_phrases = [
            "what would you like",
            "what do you want",
            "please let me know",
            "can you provide",
            "could you tell me",
            "i need to know",
            "what is your",
            "有什么我可以帮",
            "需要帮助吗",
            "还需要",
        ]

        for phrase in question_phrases:
            if phrase in content_lower:
                return False

        # Check for continuation indicators
        continuation_indicators = [
            "let me",
            "i'll ",
            "i will",
            "next,",
            "now i ",
            "continuing",
            "proceeding",
            "step",
        ]

        for indicator in continuation_indicators:
            if indicator in content_lower[-200:]:
                return True

        # Default: stop after one response to avoid loops
        return False

    async def _process_conversation_turn(
        self,
        user_input: str | None = None,
        is_autonomous: bool = False,
        turn_count: int = 0,
    ) -> bool:
        """Process a single conversation turn with streaming."""
        if not self.llm:
            self.console.print_error("LLM not initialized")
            return False

        if user_input:
            user_msg = Message(role="user", content=user_input)
            self.memory.add_message(user_msg)

        try:
            messages = self._build_messages()
            content, tool_calls = await self._stream_and_handle_tools(messages)

            if tool_calls:
                await self._execute_tool_calls(tool_calls)

                return await self._process_conversation_turn(
                    is_autonomous=True,
                    turn_count=turn_count + 1,
                )

            should_continue = await self._should_continue_autonomously(
                content, tool_calls, turn_count
            )

            return should_continue

        except Exception as e:
            self.console.print_error(f"Failed to process response: {e}")
            return False

    async def chat(self, user_input: str) -> None:
        """Process user input with multi-turn streaming support."""
        turn_count = 0
        should_continue = True

        while should_continue and turn_count < self.max_autonomous_turns:
            should_continue = await self._process_conversation_turn(
                user_input=user_input if turn_count == 0 else None,
                is_autonomous=turn_count > 0,
                turn_count=turn_count,
            )

            turn_count += 1

            if should_continue and turn_count < self.max_autonomous_turns:
                await asyncio.sleep(0.5)

        session = self.memory.get_current_session()
        if session:
            self.memory.save_session(session)

    async def handle_command(self, command: str) -> bool:
        """Handle special commands. Returns False to exit."""
        cmd = command.lower().strip()

        if cmd in ("exit", "quit", "q"):
            self.console.print_info("Goodbye! 👋")
            return False

        elif cmd == "help":
            self.console.print_help()

        elif cmd == "clear":
            self.console.clear()
            self.console.print_header()

        elif cmd == "history":
            messages = self.memory.get_messages()
            if not messages:
                self.console.print_info("No messages in current session.")
            else:
                for msg in messages:
                    self.console.print_message(msg.role, msg.content)

        elif cmd == "new":
            self.memory.create_session()
            self.console.print_success("Started a new conversation session.")

        elif cmd == "sessions":
            sessions = self.memory.list_sessions()
            self.console.print_sessions(sessions)

        elif cmd.startswith("load "):
            session_id = command[5:].strip()
            session = self.memory.load_session(session_id)
            if session:
                self.memory.set_current_session(session)
                self.console.print_success(f"Loaded session: {session.title}")
            else:
                self.console.print_error(f"Session '{session_id}' not found.")

        elif cmd == "save":
            session = self.memory.get_current_session()
            if session:
                self.memory.save_session(session)
                self.console.print_success("Session saved.")

        elif cmd == "tools":
            self.console.print_tools(self.tools.list_tools())

        else:
            self.console.print_error(f"Unknown command: {command}")

        return True

    async def run(self) -> None:
        """Main loop."""
        if not await self.initialize():
            sys.exit(1)

        if not self.memory.get_current_session():
            self.memory.create_session("New Conversation")

        self.console.print_header()
        session = self.memory.get_current_session()
        self.console.print_info(f"Session: {session.title} ({session.session_id})")
        self.console.print_info(f"Tools available: {len(self.tools.list_tools())}")
        self.console.print_info("Type 'help' for available commands\n")

        while True:
            try:
                user_input = await self.prompt.get_input_async()

                if not user_input.strip():
                    continue

                if user_input.lower() in (
                    "exit",
                    "quit",
                    "q",
                    "help",
                    "clear",
                    "history",
                    "new",
                    "sessions",
                    "save",
                    "tools",
                ) or user_input.lower().startswith("load "):
                    if not await self.handle_command(user_input):
                        break
                else:
                    await self.chat(user_input)

            except KeyboardInterrupt:
                self.console.print_info("\nUse 'exit' or 'quit' to exit properly.")
            except EOFError:
                break


@app.callback(invoke_without_command=True)
def default_command(
    ctx: typer.Context,
    session: Annotated[
        str | None,
        typer.Option("--session", "-s", help="Load a specific session by ID"),
    ] = None,
    version: Annotated[
        bool,
        typer.Option("--version", "-v", help="Show version and exit"),
    ] = False,
) -> None:
    """CLI Agent - Interactive AI assistant with multi-turn streaming and tool calling."""
    if version:
        typer.echo("cli-coder 0.1.0")
        raise typer.Exit()

    # Only run main logic if no subcommand was invoked
    if ctx.invoked_subcommand is None:
        agent = CLIAgent()

        if session:
            loaded = agent.memory.load_session(session)
            if loaded:
                agent.memory.set_current_session(loaded)
            else:
                typer.echo(f"Session '{session}' not found. Starting new session.")

        asyncio.run(agent.run())


@app.command()
def tui(
    session: Annotated[
        str | None,
        typer.Option("--session", "-s", help="Load a specific session by ID"),
    ] = None,
) -> None:
    """Launch the Textual TUI interface."""
    from cli_agent.ui.tui import run_tui

    run_tui(session_id=session)


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
