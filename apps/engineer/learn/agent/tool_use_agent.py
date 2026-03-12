import json
import re
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv

from apps.engineer.learn.agent.core.agent import BaseAgent
from apps.engineer.learn.agent.core.model import Model
from apps.engineer.learn.agent.core.tool import Tool


class ToolUseAgent(BaseAgent):
    def __init__(
            self,
            name: str,
            description: str = "",
            model: Optional[Model] = None,
            tools: Optional[List[Tool]] = None,
            max_steps: int = 5,
    ):
        super().__init__(name, description, model, max_steps)
        self.tools = tools or []
        self.SYSTEM_PROMPT = (
            "You are a helpful assistant that can use tools to help answer user queries. "
            "Use the available tools when needed, and provide a clear final answer."
        )

    def _get_openai_tools(self) -> Optional[List[Dict[str, Any]]]:
        """Convert tools to OpenAI function calling format."""
        if not self.tools:
            return None

        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self.tools
        ]

    def invoke(self, input: str) -> str:
        if not self.model:
            return "No model configured."

        # Initialize history
        self.message_history = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": input},
        ]

        step = 0
        while step < self.max_steps:
            # Get available tools in OpenAI format
            openai_tools = self._get_openai_tools()

            # Generate response with tool support
            response = self.model.generate(self.message_history, tools=openai_tools)
            message = response.choices[0].message

            # Add assistant message to history
            assistant_msg: Dict[str, Any] = {"role": "assistant", "content": message.content or ""}
            if message.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in message.tool_calls
                ]
            self.message_history.append(assistant_msg)

            # Check if model wants to call tools
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        tool_args = json.loads(tool_call.function.arguments)
                        query = tool_args.get("query", "")
                    except json.JSONDecodeError:
                        query = tool_call.function.arguments

                    tool_result = self.call_tool(tool_name, query)

                    # Add tool response to history
                    self.message_history.append(
                        {"role": "tool", "tool_call_id": tool_call.id, "content": str(tool_result)}
                    )

                step += 1
                continue

            # Return the final answer
            if message.content:
                return message.content.strip()

            step += 1

        return "Reached maximum steps without a final answer."

    def stream(self, input: str) -> str:
        pass

    def call_tool(self, tool_name: str, args: str) -> str:
        for t in self.tools:
            if t.name.lower() == tool_name.lower():
                if callable(t.func):
                    try:
                        return t.func(args)
                    except Exception as e:
                        return f"Tool {t.name} error: {e}"
                return t.description or f"No callable for tool {t.name}"
        return f"Tool {tool_name} not found"


# --- 示例工具实现（安全、简单） ---
def search_tool(query: str) -> str:
    query = query.strip()
    if not query:
        return "Empty query"
    # 这里只返回伪造结果，实际应调用真实搜索 API
    return f"Search results for '{query}': [Example result 1, Example result 2]"


def calculator_tool(expr: str) -> str:
    expr = expr.strip()
    if not expr:
        return "Empty expression"
    # 安全计算：只允许数字和运算符
    if not re.match(r"^[0-9+\-*/().\s]+$", expr):
        return "Invalid characters in expression"
    try:
        result = eval(expr, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Calc error: {e}"


if __name__ == "__main__":
    # Example usage: API key loaded from OPENAI_API_KEY env var
    load_dotenv()

    tools = [
        Tool(name="search", description="Web search", func=search_tool),
        Tool(name="calculator", description="Simple calculator", func=calculator_tool),
    ]
    agent = ToolUseAgent(name="ExampleAgent", model=Model(), tools=tools, max_steps=5)
    print(
        agent.invoke(
            "What's the population of Paris? Please use the search tool to find this information."
        )
    )
