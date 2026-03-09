"""
Wiki MCP Client - Modern client using langchain-mcp-adapters
使用 langchain-mcp-adapters 的现代 MCP 客户端

简化方案: 使用 MultiServerMCPClient 加载工具，同时保留直接调用方法供 CLI 使用
"""

import json
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from loguru import logger


def get_server_port_from_file() -> tuple[str, int] | None:
    """从文件读取服务器端口信息"""
    port_file = Path.home() / ".wiki_agent" / "mcp_server_port"
    if port_file.exists():
        content = port_file.read_text().strip()
        try:
            host, port_str = content.split(":")
            return host, int(port_str)
        except ValueError:
            return None
    return None


class WikiMCPClient:
    """
    Wiki MCP 客户端 - 使用 langchain-mcp-adapters

    双重功能:
    1. 使用 MultiServerMCPClient 自动管理连接
    2. 提供直接工具调用方法供 CLI 使用
    3. 返回 LangChain BaseTool 列表供 Agent 使用
    """

    def __init__(
        self,
        server_path: str | None = None,
        transport: str = "sse",
        sse_host: str = "127.0.0.1",
        sse_port: int = 8000,
        sse_url: str | None = None,
    ):
        """
        初始化 Wiki MCP 客户端

        Args:
            server_path: MCP 服务器脚本路径 (stdio mode)
            transport: Transport 类型 ("stdio" 或 "sse")
            sse_host: SSE transport 主机地址
            sse_port: SSE transport 端口号
            sse_url: SSE transport 完整 URL (可选)
        """
        self.server_path = server_path or "src/wiki_mcp_server.py"
        self.transport = transport
        self.sse_host = sse_host
        self.sse_port = sse_port
        self.sse_url = sse_url

        self._client: MultiServerMCPClient | None = None
        self._tools: list[BaseTool] = []
        self._tools_map: dict[str, BaseTool] = {}

    def _get_sse_url(self) -> str:
        """获取 SSE URL"""
        if self.sse_url:
            return self.sse_url

        # 尝试从文件读取服务器实际端口
        server_info = get_server_port_from_file()
        if server_info:
            host, port = server_info
            return f"http://{host}:{port}/sse"

        return f"http://{self.sse_host}:{self.sse_port}/sse"

    def _build_server_config(self) -> dict[str, Any]:
        """构建 MultiServerMCPClient 配置"""
        if self.transport == "stdio":
            return {
                "wiki": {
                    "command": "python",
                    "args": [self.server_path, "--transport", "stdio"],
                    "transport": "stdio",
                }
            }
        else:  # sse
            return {
                "wiki": {
                    "url": self._get_sse_url(),
                    "transport": "sse",
                }
            }

    async def connect(self) -> list[BaseTool]:
        """
        连接到 MCP 服务器并返回 LangChain 工具列表

        Returns:
            list[BaseTool]: LangChain 工具列表，可直接传递给 create_agent
        """
        if self._client is not None:
            logger.debug("客户端已连接，返回现有工具")
            return self._tools

        server_config = self._build_server_config()
        logger.info(f"连接到 MCP 服务器 ({self.transport} mode)")

        self._client = MultiServerMCPClient(server_config)
        self._tools = await self._client.get_tools()

        # 构建工具名称映射，便于直接调用
        self._tools_map = {tool.name: tool for tool in self._tools}

        logger.info(f"成功加载 {len(self._tools)} 个工具")
        for tool in self._tools:
            logger.debug(f"  - {tool.name}")

        return self._tools

    async def _call_tool(self, tool_name: str, **kwargs) -> str:
        """
        内部方法: 调用指定工具

        Args:
            tool_name: 工具名称
            **kwargs: 工具参数

        Returns:
            工具返回的 JSON 字符串
        """
        if self._client is None:
            await self.connect()

        tool = self._tools_map.get(tool_name)
        if not tool:
            available = list(self._tools_map.keys())
            return json.dumps(
                {"success": False, "error": f"未知工具: {tool_name}。可用工具: {available}"}
            )

        try:
            result = await tool.ainvoke(kwargs)
            # LangChain 工具返回的是字符串或结构化数据
            if isinstance(result, str):
                return result
            return json.dumps({"success": True, "result": result})
        except Exception as e:
            logger.error(f"工具调用失败 {tool_name}: {e}")
            return json.dumps({"success": False, "error": str(e)})

    @property
    def tools(self) -> list[BaseTool]:
        """获取已加载的工具列表"""
        return self._tools

    # ============== CLI 直接调用方法 ==============

    async def search(self, query: str, space_key: str = "engineer", limit: int = 10) -> str:
        """搜索 Wiki 页面"""
        return await self._call_tool("wiki_search", query=query, space_key=space_key, limit=limit)

    async def read(self, page_id: str, include_metadata: bool = True) -> str:
        """读取页面内容"""
        return await self._call_tool(
            "wiki_read", page_id=page_id, include_metadata=include_metadata
        )

    async def create(
        self, parent_id: str, title: str, content: str, space_key: str = "engineer"
    ) -> str:
        """创建页面"""
        return await self._call_tool(
            "wiki_create", parent_id=parent_id, title=title, content=content, space_key=space_key
        )

    async def update(
        self, page_id: str, title: str | None = None, content: str | None = None
    ) -> str:
        """更新页面"""
        kwargs: dict[str, Any] = {"page_id": page_id}
        if title is not None:
            kwargs["title"] = title
        if content is not None:
            kwargs["content"] = content
        return await self._call_tool("wiki_update", **kwargs)

    async def list_children(self, page_id: str, recursive: bool = False) -> str:
        """列出子页面"""
        return await self._call_tool("wiki_list_children", page_id=page_id, recursive=recursive)

    async def delete(self, page_id: str, confirm: bool = False) -> str:
        """删除页面"""
        return await self._call_tool("wiki_delete", page_id=page_id, confirm=confirm)

    async def get_spaces(self) -> str:
        """获取空间列表"""
        return await self._call_tool("wiki_get_spaces")

    # ============== 生命周期管理 ==============

    async def close(self):
        """关闭客户端连接"""
        if self._client:
            if hasattr(self._client, "close"):
                await self._client.close()
            self._client = None
            self._tools = []
            self._tools_map = {}
            logger.info("MCP 客户端连接已关闭")

    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()

if __name__ == '__main__':
    WikiMCPClient()