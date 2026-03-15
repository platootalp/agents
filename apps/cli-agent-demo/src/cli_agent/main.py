"""Main entry point for CLI Agent with multi-turn streaming support."""

import asyncio
import os
import sys

import click
from dotenv import load_dotenv

from cli_agent.core.memory import MemoryManager
from cli_agent.core.provider import Message, OpenAILLM
from cli_agent.core.tools import ToolManager
from cli_agent.tools import get_default_tools
from cli_agent.ui.console import ChatConsole

# Load environment variables
load_dotenv()


class CLIAgent:
    """Interactive CLI Agent with multi-turn streaming support."""

    def __init__(self):
        self.console = ChatConsole()
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
        self.max_autonomous_turns = 10  # Prevent infinite loops

    async def initialize(self) -> bool:
        """Initialize the agent. Returns False if setup fails."""
        # Check for API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            self.console.print_error(
                "OPENAI_API_KEY not found. Please set it in your environment or .env file."
            )
            return False

        # Initialize LLM
        base_url = os.getenv("OPENAI_BASE_URL")
        self.llm = OpenAILLM(api_key=api_key, base_url=base_url)

        # Register default tools
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
        """Stream LLM response and handle any tool calls.

        Returns the final content and any tool calls that were made.
        """
        if not self.llm:
            raise RuntimeError("LLM not initialized")

        # Stream the response
        generator = self.llm.chat_stream(
            messages=messages,
            tools=self.tools.list_tools(),
        )

        content, tool_calls = await self.console.stream_assistant_response(generator)

        # Add assistant message to memory
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

            # Show tool call
            self.console.print_tool_call(tool_name, arguments)

            # Execute tool
            result = await self.tools.execute_tool_call(tool_call)

            # Show tool result
            result_str = result.to_string()
            self.console.print_message("tool", result_str)

            # Add tool response to memory
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
        """Determine if the agent should continue autonomously or wait for user input.

        Returns True if the agent should continue without user input.
        """
        # Safety limit
        if turn_count >= self.max_autonomous_turns:
            return False

        # If we just made tool calls, we should continue to process results
        if tool_calls:
            return True

        # If content is empty or very short, likely waiting for tool results
        if not content or len(content.strip()) < 10:
            return True

        # Check if the response ends with a question (needs user input)
        content_lower = content.strip().lower()
        question_indicators = [
            "?",
            "what would you like",
            "what do you want",
            "please let me know",
            "can you provide",
            "could you tell me",
            "i need to know",
            "what is your",
        ]

        for indicator in question_indicators:
            if content_lower.endswith(indicator) or indicator in content_lower[-100:]:
                return False

        # If the agent is in the middle of a task (indicated by certain phrases)
        continuation_indicators = [
            "let me",
            "i'll",
            "i will",
            "next,",
            "now i",
            "continuing",
            "proceeding",
            "step",
        ]

        for indicator in continuation_indicators:
            if indicator in content_lower[-200:]:
                return True

        # Default: wait for user input
        return False

    async def _process_conversation_turn(
        self,
        user_input: str | None = None,
        is_autonomous: bool = False,
        turn_count: int = 0,
    ) -> bool:
        """Process a single conversation turn with streaming.

        Args:
            user_input: User message (None for autonomous continuation)
            is_autonomous: Whether this is an autonomous continuation
            turn_count: Current turn count for safety limits

        Returns:
            True if should continue autonomously, False to wait for user input
        """
        if not self.llm:
            self.console.print_error("LLM not initialized")
            return False

        # Add user message if provided
        if user_input:
            user_msg = Message(role="user", content=user_input)
            self.memory.add_message(user_msg)

        try:
            # Build messages and stream response
            messages = self._build_messages()
            content, tool_calls = await self._stream_and_handle_tools(messages)

            # Handle tool calls if any
            if tool_calls:
                # Execute tools
                await self._execute_tool_calls(tool_calls)

                # Continue the conversation autonomously to process tool results
                return await self._process_conversation_turn(
                    is_autonomous=True,
                    turn_count=turn_count + 1,
                )

            # Decide whether to continue autonomously
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
            # Process this turn
            should_continue = await self._process_conversation_turn(
                user_input=user_input if turn_count == 0 else None,
                is_autonomous=turn_count > 0,
                turn_count=turn_count,
            )

            turn_count += 1

            # If continuing autonomously, add a small delay for readability
            if should_continue and turn_count < self.max_autonomous_turns:
                await asyncio.sleep(0.5)

        # Save session after conversation
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
        # Initialize
        if not await self.initialize():
            sys.exit(1)

        # Create or load session
        if not self.memory.get_current_session():
            self.memory.create_session("New Conversation")

        # Print header
        self.console.print_header()
        session = self.memory.get_current_session()
        self.console.print_info(f"Session: {session.title} ({session.session_id})")
        self.console.print_info(f"Tools available: {len(self.tools.list_tools())}")
        self.console.print_info("Type 'help' for available commands\n")

        # Main loop
        while True:
            try:
                user_input = self.console.get_input()

                if not user_input.strip():
                    continue

                # Check if it's a command
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
                    # Regular chat with multi-turn support
                    await self.chat(user_input)

            except KeyboardInterrupt:
                self.console.print_info("\nUse 'exit' or 'quit' to exit properly.")
            except EOFError:
                break


@click.command()
@click.option(
    "--session",
    "-s",
    help="Load a specific session by ID",
)
@click.version_option(version="0.1.0")
def main(session: str | None):
    """CLI Agent - Interactive AI assistant with multi-turn streaming and tool calling."""
    agent = CLIAgent()

    # Load session if specified
    if session:
        loaded = agent.memory.load_session(session)
        if loaded:
            agent.memory.set_current_session(loaded)
        else:
            click.echo(f"Session '{session}' not found. Starting new session.")

    # Run the agent
    asyncio.run(agent.run())


if __name__ == "__main__":
    main()
