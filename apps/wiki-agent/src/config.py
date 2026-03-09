"""
Wiki Agent Configuration
Wiki Agent 配置管理
"""

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class WikiConfig(BaseSettings):
    """Wiki 配置类"""

    # Wiki 设置
    wiki_base_url: str = Field(default="https://wiki.tuhu.cn", description="Wiki 基础 URL")
    wiki_space_key: str = Field(default="engineer", description="默认空间 Key")

    # Chrome DevTools Protocol 设置
    cdp_endpoint: str = Field(default="http://localhost:9222", description="Chrome CDP 端点")

    # MCP 服务器设置
    mcp_server_path: str = Field(default="src/wiki_mcp_server.py", description="MCP 服务器脚本路径")

    # MCP Transport 设置
    mcp_transport: str = Field(default="stdio", description="MCP transport 类型 (stdio 或 sse)")
    mcp_sse_host: str = Field(default="localhost", description="SSE transport 主机地址")
    mcp_sse_port: int = Field(default=3000, description="SSE transport 端口号")
    mcp_sse_url: str | None = Field(
        default=None, description="SSE transport 完整 URL (可选，优先级高于 host/port)"
    )

    # LLM 设置
    openai_api_key: str | None = Field(default=None, description="OpenAI API Key")
    openai_base_url: str | None = Field(default=None, description="OpenAI Base URL")
    openai_model: str = Field(default="gpt-4o", description="OpenAI 模型名称")

    # 日志设置
    log_level: str = Field(default="INFO", description="日志级别")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "env_prefix": "WIKI_",
        "extra": "ignore",
    }

    @property
    def effective_openai_api_key(self) -> str | None:
        """获取有效的 OpenAI API Key"""
        return self.openai_api_key or os.getenv("OPENAI_API_KEY")

    @property
    def effective_openai_base_url(self) -> str | None:
        """获取有效的 OpenAI Base URL"""
        return self.openai_base_url or os.getenv("OPENAI_BASE_URL")

    @property
    def effective_openai_model(self) -> str:
        """获取有效的 OpenAI 模型"""
        return os.getenv("OPENAI_MODEL") or self.openai_model


def get_config() -> WikiConfig:
    """获取配置实例"""
    return WikiConfig()


def create_env_template():
    """创建 .env 文件模板"""
    env_content = """# Wiki Agent 环境变量配置

# Wiki 设置
WIKI_BASE_URL=https://wiki.tuhu.cn
WIKI_SPACE_KEY=engineer

# Chrome DevTools Protocol 设置
# 需要先启动 Chrome 并开启远程调试:
# macOS: /Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome --remote-debugging-port=9222
# Linux: google-chrome --remote-debugging-port=9222
WIKI_CDP_ENDPOINT=http://localhost:9222

# MCP 服务器设置
WIKI_MCP_SERVER_PATH=src/wiki_mcp_server.py

# MCP Transport 设置
# 可选: stdio (默认) 或 sse
WIKI_MCP_TRANSPORT=stdio

# SSE Transport 设置 (仅当 WIKI_MCP_TRANSPORT=sse 时生效)
WIKI_MCP_SSE_HOST=127.0.0.1
WIKI_MCP_SSE_PORT=8000
# WIKI_MCP_SSE_URL=http://localhost:8000/sse

# FastMCP SSE 服务器设置 (服务端使用)
FASTMCP_HOST=127.0.0.1
FASTMCP_PORT=8000

# OpenAI 设置
OPENAI_API_KEY=your-api-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o

# 日志设置
WIKI_LOG_LEVEL=INFO
"""
    # 获取项目根目录 (config.py 的父目录的父目录)
    project_root = Path(__file__).parent.parent
    env_path = project_root / ".env"
    if not env_path.exists():
        env_path.write_text(env_content, encoding="utf-8")
        print(f"已创建环境变量模板: {env_path}")
    else:
        print(f"环境变量文件已存在: {env_path}")


if __name__ == "__main__":
    create_env_template()
