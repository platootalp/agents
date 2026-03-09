# Wiki Agent - Project Knowledge Base

**Project**: wiki-agent  
**Type**: Python MCP Agent for Wiki Operations  
**Stack**: Python 3.10+, FastMCP, Playwright, LangChain, FastAPI  

## OVERVIEW

Wiki automation agent using Playwright browser automation and MCP (Model Context Protocol) to perform file-system-like operations on Confluence Wiki: search, read, create, update, list.

## STRUCTURE

```
wiki-agent/
├── src/                    # Source code
│   ├── core/              # Tool executor utilities
│   ├── cli.py             # Typer CLI interface
│   ├── config.py          # Pydantic settings
│   ├── wiki_mcp_server.py # FastMCP server (main entry)
│   ├── wiki_mcp_client.py # MCP client (stdio/SSE)
│   └── wiki_agent.py      # LangChain ReAct agent
├── tests/                 # Pytest test suite
├── examples/              # Usage examples
├── docs/                  # Architecture docs
└── pyproject.toml         # UV package config
```

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Add CLI command | `src/cli.py` | Typer commands, use `asyncio.run()` |
| Add MCP tool | `src/wiki_mcp_server.py` | Decorate with `@mcp.tool()` |
| Add Agent tool | `src/wiki_agent.py` | Extend `BaseTool`, implement `_run`/`_arun` |
| Change config | `src/config.py` | Pydantic BaseSettings with env vars |
| Fix tool execution | `src/core/tool_executor.py` | OpenCode-style tool interface |
| Update deps | `pyproject.toml` | Use `uv` for package management |

## ENTRY POINTS

- **CLI**: `python -m src.cli` or `wiki-agent` (after install)
- **MCP Server**: `python src/wiki_mcp_server.py --transport sse`
- **Package**: `from src.wiki_mcp_client import WikiMCPClient`

## CONVENTIONS

### Configuration Files
- **Primary**: `pyproject.toml` (replaces setup.py)
- **Dependencies**: `requirements.txt`
- **Environment**: `.env` (generated from template)
- **No ESLint/Prettier**: Uses Ruff for Python linting
- **No TypeScript/JavaScript**: Pure Python codebase

### Code Style
- **Ruff**: Line length 100, Python 3.10+ target
- **Quotes**: Double quotes preferred
- **Typing**: Strict mypy, `disallow_untyped_defs=true`
- **Imports**: `known-first-party=["src"]` in ruff config
- **Indentation**: Spaces (not tabs)
- **Import Sorting**: Enabled via Ruff isort

### Type Checking (MyPy)
- **Strict Settings**: Enabled with comprehensive checks
- **Exceptions**: Ignore missing imports for external libraries (Playwright, MCP, LangChain)
- **Configuration**: `warn_return_any=true`, `disallow_untyped_defs=true`, etc.

### Testing
- **Framework**: pytest with `pytest-asyncio` mode auto
- **Test Discovery**: Files matching `test_*.py`, classes `Test*`, functions `test_*`
- **Coverage**: pytest-cov for test coverage
- **Mocking**: Playwright/browser in unit tests

### Async Patterns
- All I/O is async using `asyncio`
- MCP tools: `async def` with Playwright
- LangChain tools: implement both `_run` (sync wrapper) and `_arun` (async)

### Error Handling
- Return JSON with `{"success": bool, "error": str}` from MCP tools
- Use `loguru` for logging, never `print()`
- Wrap Playwright operations in try/except with timeouts

### Package Management
- **Primary**: uv (modern Python package manager)
- **Build System**: hatchling
- **Development**: `uv sync` for dependency management

### Environment Configuration
- **Prefix**: All environment variables use `WIKI_` prefix
- **Loading**: Uses `pydantic-settings` with `.env` file support
- **Variables**: See `src/config.py` for complete list

### Naming Conventions
- **Classes**: PascalCase (e.g., `WikiConfig`, `WikiAgent`)
- **Functions/Methods**: snake_case (e.g., `get_config`, `create_page`)
- **Variables**: snake_case (e.g., `page_id`, `wiki_base_url`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `DEFAULT_SPACE_KEY`)

### Documentation
- **Bilingual**: Code comments and docs in Chinese/English
- **Docstrings**: Google-style for public functions and classes
- **User-Facing**: Bilingual documentation for CLI and API

## COMMANDS

### Development Workflow

```bash
# Package Management (uv-based)
uv sync                    # Install all dependencies (main + dev)
uv sync --dev              # Install only dev dependencies
uv sync --all-extras       # Install with all extras

# Testing
uv run pytest              # Run all tests (async mode auto)
uv run pytest -v           # Verbose test output
uv run pytest tests/test_wiki_mcp_server.py::TestWikiMCPServer  # Run specific test class
uv run pytest --cov=src    # Run with coverage reporting

# Linting & Formatting
uv run ruff check src/     # Check code quality (110+ errors currently)
uv run ruff check src/ --fix  # Auto-fix fixable issues
uv run ruff format src/    # Format code (1 file needs formatting)
uv run ruff format src/ --check  # Check formatting without applying

# Type Checking
uv run mypy src/           # Type check (75 errors currently)
uv run mypy src/ --show-error-codes  # Show error codes for detailed info

# Building
uv build                   # Build package distributions
uv build --package wiki-agent  # Build specific package
```

### CLI Usage

```bash
# Environment Setup
python src/cli.py init                            # Create .env template

# Wiki Operations
python src/cli.py search "query" --limit 5        # Search wiki pages
python src/cli.py read <page_id>                  # Read page content
python src/cli.py create <parent_id> "title" "<html>content</html>"  # Create page
python src/cli.py update <page_id> --title "new" --content "<html>new</html>"  # Update page
python src/cli.py list-children <page_id> --recursive  # List child pages
python src/cli.py agent "natural language query"  # Agent mode (requires OpenAI API)

# MCP Server Operations
python src/wiki_mcp_server.py --transport sse     # SSE mode (port 8000)
python src/wiki_mcp_server.py --transport stdio   # Stdio mode
python src/wiki_mcp_server.py --help              # Show server options
```

### Legacy Script

```bash
# Original bash-based wiki automation (backward compatibility)
./wiki-auto.sh --parent "文档标题" --title "新文档" --content "<p>内容</p>"
./wiki-auto.sh --parent-id 12345678 --title "新文档" --file content.html
./wiki-auto.sh --method api --user username --password pass --parent "文档" --title "新文档"
```

### Browser Prerequisites

```bash
# Chrome must be running with remote debugging enabled
# macOS
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=9222

# Linux
google-chrome --remote-debugging-port=9222

# Verify Chrome connection
curl http://localhost:9222/json/version
```

### Python Environment

```bash
# Check Python versions
python3 --version          # System Python (3.14.2)
uv run python --version    # Project Python (3.11.9 via uv)

# Run Python scripts with project environment
uv run python examples/example_usage.py
uv run python -c "import sys; print(sys.path)"
```

## DEPENDENCIES

- **Core**: pydantic, pydantic-settings, loguru
- **MCP**: mcp>=1.0.0 (FastMCP)
- **Browser**: playwright>=1.40.0
- **Agent**: langchain>=1.0.0, langgraph>=0.0.50, langchain-openai
- **API**: fastapi, uvicorn, typer, rich

## ENVIRONMENT

Required environment variables (via `.env` file):
- `OPENAI_API_KEY` - For Agent mode
- `WIKI_BASE_URL` - Wiki base URL (default: https://wiki.tuhu.cn)
- `WIKI_SPACE_KEY` - Wiki space key (default: engineer)
- `WIKI_CDP_ENDPOINT` - Chrome CDP endpoint (default: http://localhost:9222)
- `WIKI_MCP_TRANSPORT` - stdio or sse
- `WIKI_MCP_SERVER_PATH` - MCP server script path (default: src/wiki_mcp_server.py)

**Current Configuration (from .env):**
- Uses Kimi-K2.5 model via custom OpenAI-compatible endpoint
- SSE transport configured for MCP
- Custom space key: `~lijunyi3`

Prerequisites:
- Chrome running with `--remote-debugging-port=9222`
- User logged into Wiki in Chrome
- Chrome DevTools Protocol accessible at configured endpoint

## ARCHITECTURE

Multi-layer design:
1. **CLI Layer** (`cli.py`): Typer commands, user interface
2. **Agent Layer** (`wiki_agent.py`): LangChain ReAct agent with tools
3. **Client Layer** (`wiki_mcp_client.py`): MCP client (stdio/SSE transport)
4. **Server Layer** (`wiki_mcp_server.py`): FastMCP server with Playwright tools
5. **Tool Layer** (`core/tool_executor.py`): OpenCode-style tool execution

See `docs/architecture.md` for planned multi-agent team architecture.

## CONFIGURATION DETAILS

### Ruff Linting Rules
- **Target Python**: 3.10+
- **Line Length**: 100 characters
- **Enabled Rules**: E (pycodestyle errors), F (Pyflakes), I (isort), N (pep8-naming), W (pycodestyle warnings), UP (pyupgrade), B (flake8-bugbear), C4 (flake8-comprehensions), SIM (flake8-simplify)
- **Formatting**: Double quotes, spaces for indentation, auto line endings

### MyPy Type Checking
- **Python Version**: 3.10
- **Strict Settings**: `warn_return_any=true`, `disallow_untyped_defs=true`, `disallow_incomplete_defs=true`, `check_untyped_defs=true`, `no_implicit_optional=true`
- **External Libraries**: Missing imports ignored for Playwright, MCP, LangChain packages

### Pytest Configuration
- **Async Mode**: auto
- **Test Discovery**: `test_*.py` files, `Test*` classes, `test_*` functions
- **Test Paths**: `tests` directory
- **Options**: Verbose output with short tracebacks

### Package Management
- **Build System**: hatchling
- **Package Structure**: `src` directory layout
- **Scripts**: `wiki-agent` (CLI), `wiki-mcp-server` (MCP server)

## NOTES

- **Chinese Language**: Code comments and docs primarily in Chinese
- **Browser Dependency**: Requires Chrome with remote debugging enabled
- **HTML Content**: Wiki create/update requires HTML format content
- **Legacy Script**: `wiki-auto.sh` maintained for backward compatibility

### Project-Specific Conventions Discovered
- **Configuration**: Uses modern Python packaging with `pyproject.toml` and `uv`
- **Linting**: Ruff replaces traditional Python linters (flake8, isort, pyupgrade)
- **Type Checking**: Strict MyPy configuration with specific exceptions
- **Environment**: Custom `WIKI_` prefix for environment variables
- **Testing**: Comprehensive pytest configuration with async support
- **Documentation**: Bilingual approach with Chinese/English content
- **Architecture**: Multi-layer design with MCP protocol integration

## ANTI-PATTERNS AND GOTCHAS

### Configuration Anti-Patterns
- **❌ DON'T**: Use `requirements.txt` as primary dependency management
- **✅ DO**: Use `pyproject.toml` with `uv` (modern Python standard)
- **❌ DON'T**: Mix environment variable prefixes inconsistently
- **✅ DO**: Use `WIKI_` prefix for all project-specific environment variables

### Import Anti-Patterns
- **❌ DON'T**: Use relative imports like `from .config import get_config`
- **✅ DO**: Use absolute imports with `src` prefix: `from src.config import get_config`
- **❌ DON'T**: Import third-party libraries without proper error handling
- **✅ DO**: Use MyPy ignore for external libraries with missing type stubs

### Async/Await Anti-Patterns
- **❌ DON'T**: Mix sync and async patterns inconsistently
- **✅ DO**: Follow established patterns:
  - MCP tools: `async def` with Playwright
  - LangChain tools: implement both `_run` (sync wrapper) and `_arun` (async)
- **❌ DON'T**: Use bare `asyncio.sleep()` without proper context
- **✅ DO**: Use `await asyncio.sleep()` after Playwright actions that trigger page loads

### Error Handling Anti-Patterns
- **❌ DON'T**: Use bare `except:` statements
- **✅ DO**: Use specific exception handling with `loguru` logging
- **❌ DON'T**: Return inconsistent error formats
- **✅ DO**: Always return JSON with `{"success": bool, "error": str}` from MCP tools

### Project-Specific Gotchas

#### Browser Dependency
- **Gotcha**: Requires Chrome running with `--remote-debugging-port=9222`
- **Risk**: Agent will fail if Chrome not properly configured
- **Mitigation**: Document prerequisite setup clearly in `.env` template

#### HTML Content Requirement
- **Gotcha**: Wiki create/update operations require HTML format content
- **Risk**: Plain text content will not render correctly
- **Mitigation**: Document format requirements and provide examples

#### Page ID vs Title Confusion
- **Gotcha**: Many operations require page_id, not page title
- **Risk**: Users may try to use titles directly
- **Mitigation**: Clear documentation and error messages in tool descriptions

#### Missing Quality Gates
- **Anti-Pattern**: No automated testing in CI pipeline
- **Risk**: Code quality depends on manual testing
- **Recommendation**: Add pytest automation and coverage reporting

#### Documentation Gaps
- **Anti-Pattern**: Bilingual documentation but no clear separation
- **Risk**: Mixed languages may confuse contributors
- **Recommendation**: Establish clear documentation language policy

### Missing Infrastructure
- **❌ No Git Repository**: Project is not initialized as git repo
- **❌ No CI/CD Pipeline**: Missing automated testing and deployment
- **❌ No Pre-commit Hooks**: Missing automated linting and formatting checks
- **Recommendation**: Initialize git repo and add CI/CD automation

## CURRENT CODE QUALITY STATUS

### Test Status
- **✅ Tests Passing**: All 10 tests pass successfully
- **✅ Test Coverage**: Basic unit tests for core functionality
- **⚠️ Test Coverage Gap**: Limited integration/e2e tests for browser operations

### Linting Status
- **⚠️ Ruff Issues**: 110 errors detected (95 fixable with `--fix`)
- **Common Issues**: Import sorting, unused imports, type annotation improvements
- **Formatting**: 1 file needs formatting (`src/core/tool_executor.py`)

### Type Checking Status
- **⚠️ MyPy Issues**: 75 errors across 6 files
- **Common Issues**: Missing type annotations, import resolution issues
- **External Libraries**: Missing type stubs for Playwright, MCP, LangChain (ignored)

### Non-Standard Patterns Discovered

#### Dual Python Environments
- **System Python**: 3.14.2 (not used for development)
- **Project Python**: 3.11.9 (via uv virtual environment)
- **Pattern**: Always use `uv run` prefix for project commands

#### Browser Dependency Pattern
- **Requirement**: Chrome must be running with remote debugging enabled
- **Pattern**: Browser automation via Playwright with CDP connection
- **Gotcha**: Agent fails if Chrome not properly configured

#### Legacy Script Maintenance
- **File**: `wiki-auto.sh` (469 lines of bash)
- **Purpose**: Backward compatibility with original wiki automation
- **Pattern**: Dual approach (bash script + Python agent) maintained

#### Modern Package Management
- **Primary**: `uv` (modern Python package manager)
- **Build System**: `hatchling` (no setup.py)
- **Pattern**: Modern Python packaging standards adopted

#### Bilingual Documentation
- **Pattern**: Code comments and docs in Chinese/English
- **Files**: README_WIKI_AGENT.md, AGENTS.md, source code comments
- **Benefit**: Supports both Chinese and English speaking developers
