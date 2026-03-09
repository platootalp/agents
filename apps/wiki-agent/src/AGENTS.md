# Wiki Agent Source Code

**Path**: `src/`  
**Scope**: Core implementation - MCP server, client, agent, CLI

## STRUCTURE

```
src/
├── core/
│   └── tool_executor.py    # OpenCode-style tool execution
├── cli.py                  # Typer CLI interface
├── config.py               # Pydantic settings management
├── wiki_mcp_server.py      # FastMCP server with Playwright tools
├── wiki_mcp_client.py      # MCP client (stdio/SSE transport)
├── wiki_agent.py           # LangChain ReAct agent implementation
└── __init__.py             # Package init (empty)
```

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Add CLI command | `cli.py` | Use `typer.Typer()`, wrap async with `asyncio.run()` |
| Add MCP tool | `wiki_mcp_server.py` | Use `@mcp.tool()` decorator |
| Add Agent tool | `wiki_agent.py` | Extend `BaseTool`, define `args_schema` with Pydantic |
| Modify config | `config.py` | Add fields to `WikiConfig` (BaseSettings) |
| Add tool executor | `core/tool_executor.py` | Add to `tools` dict in `execute()` method |

## CONVENTIONS

### Import Patterns
```python
# Internal imports - use 'src' prefix for consistency
from src.wiki_mcp_client import WikiMCPClient
from src.config import get_config

# Third-party - grouped by functionality
from mcp.server.fastmcp import FastMCP
from playwright.async_api import async_playwright
from langchain_core.tools import BaseTool
```

### MCP Tool Pattern
```python
@mcp.tool()
async def wiki_example(param: str) -> str:
    """Tool description for LLM"""
    page = await ensure_browser()
    try:
        # Implementation
        return json.dumps({"success": True, "data": result})
    except Exception as e:
        logger.error(f"Failed: {e}")
        return json.dumps({"success": False, "error": str(e)})
```

### LangChain Tool Pattern
```python
class WikiExampleTool(BaseTool):
    name: ClassVar[str] = "wiki_example"
    description: ClassVar[str] = "Description for LLM"
    args_schema: ClassVar[Type[BaseModel]] = WikiExampleInput
    client: WikiMCPClient = Field(default=None, exclude=True)

    def _run(self, param: str, run_manager=None) -> str:
        return asyncio.get_event_loop().run_until_complete(
            self.client.example(param)
        )

    async def _arun(self, param: str, run_manager=None) -> str:
        return await self.client.example(param)
```

### Async Pattern
- Always use `async def` for I/O operations
- Wrap sync calls in `asyncio.get_event_loop().run_until_complete()` for LangChain `_run`
- Use `asyncio.sleep()` after Playwright actions that trigger page loads

## NOTES

- **Browser State**: Global `_browser`, `_context`, `_page` in `wiki_mcp_server.py`
- **Error Format**: All MCP tools return JSON with `{"success": bool, "error": str}`
- **CDP Connection**: Connects to Chrome via `playwright.chromium.connect_over_cdp()`
- **Chinese Comments**: Docstrings primarily in Chinese for team consistency
