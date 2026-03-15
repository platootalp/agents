# CLI Agent Demo

一个基于 Rich 的交互式 CLI Agent，支持 OpenAI LLM、工具调用和持久化对话历史。

## 特性

- 🎨 **Rich UI**: 漂亮的终端界面，支持 Markdown 渲染、代码高亮、对话气泡
- 🔧 **工具调用**: 内置多种实用工具（天气、计算、搜索等）
- 💾 **持久化**: 自动保存对话历史到本地文件
- 🚀 **异步**: 基于 asyncio 的高性能实现

## 安装

```bash
# 进入项目目录
cd apps/cli-agent-demo

# 安装依赖
uv sync
```

## 配置

复制示例配置文件并设置你的 API Key:

```bash
cp .env.example .env
# 编辑 .env 文件，设置 OPENAI_API_KEY
```

## 使用

### 启动交互式对话

```bash
# 方式1: 使用 uv
uv run python -m cli_agent.main

# 方式2: 使用入口命令 (安装后)
cli-agent
```

### 加载历史会话

```bash
uv run python -m cli_agent.main --session 20240315_143022
```

### 可用命令

在对话中输入以下命令:

- `help` - 显示帮助信息
- `exit` / `quit` - 退出程序
- `clear` - 清屏
- `new` - 开始新会话
- `sessions` - 列出所有保存的会话
- `load <session_id>` - 加载指定会话
- `history` - 显示当前会话历史
- `tools` - 列出可用工具

## 项目结构

```
cli-agent-demo/
├── src/cli_agent/
│   ├── core/           # 核心模块
│   │   ├── provider.py # LLM Provider (OpenAI)
│   │   ├── memory.py   # 持久化存储
│   │   └── tools.py    # 工具系统
│   ├── ui/
│   │   └── console.py  # Rich UI 界面
│   ├── tools/
│   │   └── __init__.py # 默认工具实现
│   └── main.py         # 主入口
├── pyproject.toml
└── README.md
```

## 添加自定义工具

在 `tools/__init__.py` 中添加新工具:

```python
async def my_custom_tool(param: str) -> str:
    """Tool description."""
    return f"Result: {param}"

# 在 get_default_tools() 中添加
def get_default_tools() -> list[Tool]:
    return [
        # ... existing tools
        Tool.from_function(my_custom_tool, description="My custom tool"),
    ]
```

## 依赖

- `openai>=1.0.0` - OpenAI API 客户端
- `rich>=13.0.0` - 终端 UI 库
- `pydantic>=2.0.0` - 数据验证
- `python-dotenv>=1.0.0` - 环境变量
- `click>=8.0.0` - CLI 框架

## License

MIT
