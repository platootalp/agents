"""
Wiki Agent - Modern LangChain Agent with MCP Integration
基于 LangChain 1.0 的 Wiki 操作 Agent，使用 MCP 工具

简化方案:
- 使用 MultiServerMCPClient 加载 MCP 工具
- 使用新的 create_agent 构建 ReAct 循环
- 无需手动包装工具
"""

import os
import re
from typing import Any, AsyncGenerator

from langchain.agents import create_agent


def clean_text_for_logging(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)

    text = re.sub(r"[\ud800-\udfff]", "", text)
    text = "".join(char for char in text if ord(char) >= 32 or char in "\n\t\r")

    return text


from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from loguru import logger

from src.wiki_mcp_client import WikiMCPClient


class WikiAgent:
    """
    Wiki 操作 Agent - 使用 LangChain 1.0 create_agent

    架构:
    1. 使用 WikiMCPClient 加载 MCP 工具 (自动转换为 LangChain 工具)
    2. 使用 create_agent 构建 ReAct 循环 Agent
    3. Agent 自动决定调用哪些工具，循环直到任务完成
    """

    SYSTEM_PROMPT = """你是一个专业的 Wiki 文档管理助手。你可以使用工具来帮助用户搜索、读取、创建、更新和管理 Wiki 页面。

重要提示：
- page_id 是页面的数字 ID，不是标题
- 如果你只有标题，先用 wiki_search 找到 page_id
- 创建页面时必须指定父页面（parent_id），父页面必须存在
- 内容必须是 HTML 格式
- 删除操作不可逆，必须确认 confirm=True

工作原则：
1. 积极主动：不要只是询问，而是先执行你能做的操作
2. 先检查后操作：创建页面前先用 wiki_read 检查父页面是否存在
3. 使用默认值：如果用户没有指定内容，使用 "<p>测试内容</p>" 作为默认内容
4. 一步到位：尽可能在一次对话中完成用户的请求

请使用工具来完成任务，不要只是文字回复。"""

    def __init__(
        self,
        llm: ChatOpenAI | None = None,
        mcp_client: WikiMCPClient | None = None,
    ):
        """
        初始化 Wiki Agent

        Args:
            llm: 语言模型实例，默认使用环境变量配置的模型
            mcp_client: MCP 客户端实例，默认创建新实例
        """
        self.llm = llm or self._create_default_llm()
        self.mcp_client = mcp_client or self._create_default_client()
        self._agent = None

    def _create_default_llm(self) -> ChatOpenAI:
        """创建默认的语言模型"""
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        model = os.getenv("OPENAI_MODEL", "gpt-4o")

        if not api_key:
            raise ValueError(
                "未设置 OPENAI_API_KEY 环境变量。请设置环境变量或使用自定义 llm 参数。"
            )

        return ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=0.1,
        )

    def _create_default_client(self) -> WikiMCPClient:
        """创建默认的 MCP 客户端"""
        from src.config import get_config

        config = get_config()
        return WikiMCPClient(
            server_path=config.mcp_server_path,
            transport=config.mcp_transport,
            sse_host=config.mcp_sse_host,
            sse_port=config.mcp_sse_port,
            sse_url=config.mcp_sse_url,
        )

    async def initialize(self):
        """
        初始化 Agent

        加载 MCP 工具并创建 LangChain Agent
        """
        if self._agent is not None:
            return

        # 从 MCP 服务器加载工具
        logger.info("[AGENT] 正在加载 MCP 工具...")
        tools = await self.mcp_client.connect()
        logger.info(f"[AGENT] 已加载 {len(tools)} 个工具")

        # 使用新的 create_agent 创建 ReAct Agent
        # create_agent 内部使用 LangGraph 构建循环：
        # 1. LLM 决定调用哪个工具
        # 2. 执行工具
        # 3. 将结果返回给 LLM
        # 4. 重复直到任务完成或达到最大步数
        logger.info("[AGENT] 正在创建 Agent...")

        self._agent = create_agent(
            model=self.llm,
            tools=tools,
            system_prompt=self.SYSTEM_PROMPT,
        )
        logger.info("[AGENT] Agent 创建完成")

    async def run(self, query: str) -> dict[str, Any]:
        """
        运行 Agent 执行任务

        Args:
            query: 用户查询

        Returns:
            Agent 执行结果，包含输出和消息历史
        """
        # 确保 Agent 已初始化
        if self._agent is None:
            await self.initialize()

        logger.info(f"[AGENT] 执行查询: {query}")

        try:
            # 调用 Agent
            # create_agent 返回的是 LangGraph CompiledGraph
            # 使用 ainvoke 执行，传入消息列表
            logger.debug(f"[AGENT] Agent type: {type(self._agent)}")
            logger.debug(f"[AGENT] Calling ainvoke with query: {query}")

            # Add recursion limit to prevent infinite loops
            result = await self._agent.ainvoke(
                {"messages": [HumanMessage(content=query)]},
                config={"recursion_limit": 10},
            )

            # 提取结果
            messages = result.get("messages", [])

            # 找到最后一条 AI 消息作为输出
            output = ""
            for msg in reversed(messages):
                if hasattr(msg, "content") and msg.type == "ai":
                    output = msg.content
                    break

            # 统计工具调用次数
            tool_calls = [m for m in messages if getattr(m, "type", None) == "tool"]

            logger.info(f"[AGENT] 任务完成，执行了 {len(tool_calls)} 个工具调用")

            return {
                "success": True,
                "query": query,
                "output": output,
                "messages": messages,
                "tool_calls_count": len(tool_calls),
            }

        except Exception as e:
            logger.error(f"[AGENT] 执行失败: {e}")
            return {
                "success": False,
                "query": query,
                "error": str(e),
            }

    async def run_stream(self, query: str) -> AsyncGenerator[dict[str, Any], None]:
        """
        运行 Agent 执行任务（流式输出）

        Args:
            query: 用户查询

        Yields:
            流式输出块，包含类型和内容
            - type: "content" - LLM 生成的内容
            - type: "tool_start" - 工具调用开始
            - type: "tool_end" - 工具调用结束
            - type: "complete" - 任务完成
        """
        # 确保 Agent 已初始化
        if self._agent is None:
            await self.initialize()

        safe_query = clean_text_for_logging(query)
        logger.info(f"[AGENT] 执行流式查询: {safe_query}")

        try:
            messages = [HumanMessage(content=safe_query)]
            tool_calls_count = 0
            current_tool = None

            # create_agent 返回的是 LangGraph CompiledGraph，使用 stream() 方法
            # stream() 返回同步生成器，需要包装为异步
            import asyncio

            def stream_in_thread():
                return list(
                    self._agent.stream(
                        {"messages": messages},
                        stream_mode=["messages", "updates"],
                        config={"recursion_limit": 10},
                    )
                )

            # 在线程中运行同步 stream
            chunks = await asyncio.to_thread(stream_in_thread)

            for chunk in chunks:
                # 处理不同类型的流式输出
                if isinstance(chunk, tuple) and len(chunk) >= 2:
                    chunk_type, chunk_data = chunk[0], chunk[1]

                    if chunk_type == "messages":
                        for msg in chunk_data:
                            if hasattr(msg, "content") and msg.content:
                                safe_content = clean_text_for_logging(msg.content)
                                yield {
                                    "type": "content",
                                    "content": safe_content,
                                }
                            if hasattr(msg, "tool_calls") and msg.tool_calls:
                                for tc in msg.tool_calls:
                                    tool_name = "unknown"
                                    if isinstance(tc, dict):
                                        tool_name = tc.get("name", "unknown")
                                    elif hasattr(tc, "name"):
                                        tool_name = tc.name
                                    elif hasattr(tc, "function") and hasattr(tc.function, "name"):
                                        tool_name = tc.function.name

                                    current_tool = tool_name
                                    yield {
                                        "type": "tool_start",
                                        "tool_name": tool_name,
                                    }
                                    tool_calls_count += 1

                    # 处理更新流（工具结果）
                    elif chunk_type == "updates":
                        for update in chunk_data:
                            if isinstance(update, dict):
                                # 检测工具完成
                                if "tools" in update or "tool" in update:
                                    yield {
                                        "type": "tool_end",
                                        "tool_name": current_tool or "unknown",
                                    }
                                    current_tool = None

                # 兼容处理：如果是字典格式
                elif isinstance(chunk, dict):
                    if "messages" in chunk:
                        for msg in chunk["messages"]:
                            if hasattr(msg, "content") and msg.content:
                                yield {
                                    "type": "content",
                                    "content": msg.content,
                                }

            # 发送完成事件
            yield {
                "type": "complete",
                "query": query,
                "tool_calls_count": tool_calls_count,
            }

            logger.info(f"[AGENT] 流式任务完成，执行了 {tool_calls_count} 个工具调用")

        except Exception as e:
            safe_error = clean_text_for_logging(str(e))
            logger.error(f"[AGENT] 流式执行失败: {safe_error}")
            yield {
                "type": "error",
                "error": safe_error,
                "query": safe_query,
            }

    async def close(self):
        """关闭 Agent 资源"""
        if self.mcp_client:
            await self.mcp_client.close()
            logger.info("[AGENT] 资源已释放")

    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()
