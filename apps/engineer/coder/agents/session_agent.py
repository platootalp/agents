"""SessionAgent - 支持持久化会话的 Agent

继承自 ToolUseAgent，添加会话持久化功能。
每个对话 session 自动保存到 SessionManager，支持跨对话恢复历史。
"""

import sys
import os

from dotenv import load_dotenv

# 添加项目根目录到路径
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
)
sys.path.insert(0, project_root)

from typing import Any, Dict, List, Optional, Union

# Session 系统导入
try:
    from apps.engineer.coder.core.session import Session, SessionManager
    from apps.engineer.coder.core.tools.base import BaseTool
    from apps.engineer.coder.agents.tool_use_agent import ToolUseAgent
    from apps.engineer.coder.core.model import Model
    from apps.engineer.coder.core.utils import MessageBuilder
except ImportError:
    from apps.engineer.coder.core.session import Session, SessionManager
    from apps.engineer.coder.core.tools.base import BaseTool
    from apps.engineer.coder.agents.tool_use_agent import ToolUseAgent
    from apps.engineer.coder.core.model import Model
    from apps.engineer.coder.core.utils import MessageBuilder


class SessionAgent(ToolUseAgent):
    """支持会话持久化的 Agent

    特性:
        - 自动保存对话历史到持久化存储
        - 支持通过 session_id 恢复历史对话
        - 可配置消息窗口大小（防止上下文过长）
        - 支持文件系统存储或其他 SessionStore 实现

    使用示例:
        # 使用内存存储（默认）
        agent = SessionAgent(name="Assistant", model=model)

        # 使用文件存储
        store = FileSystemSessionStore("./sessions")
        manager = SessionManager(store)
        agent = SessionAgent(name="Assistant", model=model, session_manager=manager)

        # 开始对话
        response = agent.run("session-001", "你好")

        # 恢复对话
        response = agent.run("session-001", "继续刚才的话题")
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        model: Optional[Model] = None,
        tools: Optional[List[Union[BaseTool, Any]]] = None,
        max_steps: int = 10,
        max_window_messages: int = 20,
        session_manager: Optional[SessionManager] = None,
    ):
        """初始化 SessionAgent

        Args:
            name: Agent 名称
            description: Agent 描述
            model: LLM 模型实例
            tools: 工具列表
            max_steps: 最大执行步数
            max_window_messages: 加载历史消息的最大数量
            session_manager: SessionManager 实例（默认内存存储）
        """
        super().__init__(name, description, model, tools, max_steps)
        self.session_manager = session_manager or SessionManager()
        self.max_window_messages = max_window_messages
        self.current_session_id: Optional[str] = None

    def _load_session_history(self, session: Session) -> List[Dict[str, Any]]:
        """将 Session 中的消息转换为 LLM 可用的格式"""
        messages = []
        history = session.get_messages(limit=self.max_window_messages)

        for msg in history:
            if msg.role == "tool":
                messages.append(
                    {
                        "role": "tool",
                        "content": msg.content,
                        "tool_call_id": msg.tool_call_id or "",
                    }
                )
            elif msg.tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": msg.content or "",
                        "tool_calls": msg.tool_calls,
                    }
                )
            else:
                messages.append(
                    {
                        "role": msg.role,
                        "content": msg.content,
                    }
                )

        return messages

    def _save_message_to_session(
        self,
        session_id: str,
        role: str,
        content: str,
        tool_calls: Optional[List[Dict]] = None,
        tool_call_id: Optional[str] = None,
    ) -> None:
        """保存消息到 session"""
        try:
            from apps.engineer.coder.core.message import Message
        except ImportError:
            from apps.engineer.coder.core.message import Message

        session = self.session_manager.get_session(session_id)
        if session:
            message = Message(
                role=role,
                content=content,
                tool_calls=tool_calls,
                tool_call_id=tool_call_id,
            )
            session.add_message(message)
            self.session_manager.save_session(session)

    def run(
        self,
        session_id: str,
        input: str,
        workspace: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> str:
        """执行对话并自动保存到 session

        Args:
            session_id: 会话 ID（用于恢复历史或创建新会话）
            input: 用户输入
            workspace: 工作空间（新会话时有效）
            user_id: 用户 ID（新会话时有效）

        Returns:
            Agent 响应
        """
        if not self.model:
            return "Error: No model configured."

        self.current_session_id = session_id

        # 获取或创建 session
        session = self.session_manager.get_or_create_session(
            session_id=session_id,
            workspace=workspace or "default",
            user_id=user_id or "",
        )

        # 加载历史消息
        history = self._load_session_history(session)

        # 构建系统提示
        system_prompt = self._build_system_prompt()

        # 初始化对话
        if not history:
            self.message_history = [
                MessageBuilder.build_system_message(system_prompt),
                MessageBuilder.build_user_message(input),
            ]
        else:
            # 在历史消息前插入系统提示（如果没有的话）
            if not history or history[0].get("role") != "system":
                self.message_history = [
                    MessageBuilder.build_system_message(system_prompt)
                ] + history
            else:
                self.message_history = history
            self.message_history.append(MessageBuilder.build_user_message(input))

        # 保存用户消息到 session
        self._save_message_to_session(session_id, "user", input)

        # 执行对话循环
        for step in range(self.max_steps):
            openai_tools = self._get_openai_tools()

            # 调用 LLM
            response = self.model.generate(self.message_history, tools=openai_tools)
            message = response.choices[0].message

            # 处理工具调用
            tool_calls = MessageBuilder.convert_api_tool_calls(message.tool_calls)

            # 保存助手消息到 session
            self._save_message_to_session(
                session_id,
                "assistant",
                message.content or "",
                tool_calls=tool_calls if tool_calls else None,
            )

            # 添加到本地历史
            assistant_msg = MessageBuilder.build_assistant_message(
                message.content or "", tool_calls
            )
            self.message_history.append(assistant_msg)

            # 如果没有工具调用，返回结果
            if not tool_calls:
                return message.content.strip() if message.content else ""

            # 执行工具调用
            results = self._execute_tool_calls(tool_calls)

            # 保存工具响应并添加到历史
            for result in results:
                self._save_message_to_session(
                    session_id,
                    "tool",
                    result["result"],
                    tool_call_id=result["tool_call_id"],
                )
                tool_msg = MessageBuilder.build_tool_response_message(
                    result["tool_call_id"], result["result"]
                )
                self.message_history.append(tool_msg)

        return "Reached maximum steps without a final answer."

    def stream(
        self,
        session_id: str,
        input: str,
        workspace: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> str:
        """流式执行对话并自动保存到 session"""
        if not self.model:
            return "Error: No model configured."

        self.current_session_id = session_id

        # 获取或创建 session
        session = self.session_manager.get_or_create_session(
            session_id=session_id,
            workspace=workspace or "default",
            user_id=user_id or "",
        )

        # 加载历史
        history = self._load_session_history(session)

        # 构建系统提示
        system_prompt = self._build_system_prompt()

        # 初始化对话
        if not history:
            print("\n🆕 New Conversation\n")
            self.message_history = [
                MessageBuilder.build_system_message(system_prompt),
                MessageBuilder.build_user_message(input),
            ]
        else:
            print(f"\n📚 Loaded {len(history)} messages from history\n")
            if not history or history[0].get("role") != "system":
                self.message_history = [
                    MessageBuilder.build_system_message(system_prompt)
                ] + history
            else:
                self.message_history = history
            self.message_history.append(MessageBuilder.build_user_message(input))

        # 保存用户消息
        self._save_message_to_session(session_id, "user", input)
        print(f"👤 User: {input}\n")

        # 执行对话循环
        for step in range(self.max_steps):
            openai_tools = self._get_openai_tools()

            # 流式调用
            stream = self.model.stream(self.message_history, tools=openai_tools)

            accumulated_content = ""
            accumulated_tool_calls: Dict[int, Dict[str, Any]] = {}
            in_thinking = False
            in_content = False

            for chunk in stream:
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                # 处理思考内容
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    if not in_thinking:
                        print("\n💭 Thinking:\n", end="", flush=True)
                        in_thinking = True
                        in_content = False
                    print(delta.reasoning_content, end="", flush=True)

                # 处理内容
                if delta.content:
                    if not in_content:
                        print("\n📝 Assistant:\n", end="", flush=True)
                        in_content = True
                        in_thinking = False
                    print(delta.content, end="", flush=True)
                    accumulated_content += delta.content

                # 处理工具调用
                if delta.tool_calls:
                    accumulated_tool_calls.update(
                        MessageBuilder.accumulate_tool_calls(delta.tool_calls)
                    )

            tool_calls_list = list(accumulated_tool_calls.values())

            # 保存助手消息
            self._save_message_to_session(
                session_id,
                "assistant",
                accumulated_content,
                tool_calls=tool_calls_list if tool_calls_list else None,
            )

            # 添加到本地历史
            assistant_msg = MessageBuilder.build_assistant_message(
                accumulated_content, tool_calls_list
            )
            self.message_history.append(assistant_msg)

            # 如果没有工具调用，返回结果
            if not tool_calls_list:
                return accumulated_content.strip() if accumulated_content else ""

            # 执行工具
            print(f"\n🔧 Tool Calls ({len(tool_calls_list)}):")
            for i, tool_call in enumerate(tool_calls_list, 1):
                tool_name = tool_call["function"]["name"]
                args = tool_call["function"]["arguments"]

                print(f"  [{i}] {tool_name}")
                if args:
                    print(f"      Args: {args[:200]}")

                import time

                start_time = time.time()
                print(f"      Executing...", end="", flush=True)

                tool_result = self.call_tool(tool_name, args)
                elapsed = (time.time() - start_time) * 1000
                print(f" ✓ Done ({elapsed:.0f}ms)")

                result_display = (
                    tool_result[:300] + "..." if len(tool_result) > 300 else tool_result
                )
                print(f"      Result: {result_display}")

                # 保存工具响应
                self._save_message_to_session(
                    session_id,
                    "tool",
                    tool_result,
                    tool_call_id=tool_call["id"],
                )

                tool_msg = MessageBuilder.build_tool_response_message(
                    tool_call["id"], str(tool_result)
                )
                self.message_history.append(tool_msg)

            print()

        return "Reached maximum steps without a final answer."

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取 session 信息"""
        session = self.session_manager.get_session(session_id)
        if not session:
            return None

        return {
            "session_id": session.session_id,
            "workspace": session.workspace,
            "user_id": session.user_id,
            "title": session.title,
            "tags": session.tags,
            "message_count": session.get_message_count(),
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "duration_seconds": session.get_duration(),
            "token_estimate": session.get_token_estimate(),
        }

    def update_session_title(self, session_id: str, title: str) -> bool:
        """更新 session 标题"""
        session = self.session_manager.get_session(session_id)
        if session:
            session.update_title(title)
            self.session_manager.save_session(session)
            return True
        return False


if __name__ == "__main__":
    load_dotenv()
    agent = SessionAgent(name="SessionAgent", model=Model())
    result = agent.run("test", "什么是deepagent")
    print(result)
    result2 = agent.run("test", "继续刚才的话题，我们讲到哪里了？")
    print(result2)
