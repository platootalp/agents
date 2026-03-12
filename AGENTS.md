# AI Agents Monorepo

**Type**: UV Workspace Monorepo  
**Stack**: Python 3.11+, UV, Ruff, MyPy, Pytest  
**Structure**: apps/ + packages/ + skills/

## OVERVIEW

A comprehensive monorepo containing AI agent applications, tools, and learning resources. Organized as a UV workspace with shared dev dependencies and cross-package dependencies.

## STRUCTURE

```
agents/
├── apps/                           # Applications
│   ├── engineer/                  # AI Engineer framework (71 Python files)
│   ├── wiki-agent/                # Wiki MCP agent (Playwright-based)
│   ├── langgraph-templates/       # LangGraph patterns (RAG, Memory, ReAct)
│   ├── coze-agent/                # Coze platform learning examples
│   ├── huggingface/               # HuggingFace integration
│   └── langchain-example/         # LangChain examples
├── packages/                       # Shared packages
│   └── rag-utils/                 # RAG utilities
├── skills/                         # Agent skills collection
│   ├── ai-product-manager/        # Product management skill
│   ├── github-stars-indexer/      # GitHub stars indexing
│   ├── java-ai-learning-planner/  # Learning path planner
│   └── skill-creator/             # Skill creation tools
└── docs/                          # Documentation
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Add new app | `apps/<new-app>/` | Create `pyproject.toml` with `[project]` |
| Add shared package | `packages/<name>/` | Reference in workspace members |
| Add skill | `skills/<name>/` | Follow SKILL.md + references/ pattern |
| Update workspace deps | `pyproject.toml` | Root dev dependencies |
| Run tests | `uv run pytest` | Discovers tests in apps/ and packages/ |
| Lint code | `uv run ruff check .` | Line length 100, Python 3.11+ |

## CONVENTIONS

### UV Workspace
- **Root**: Defines workspace members and shared dev deps
- **Apps**: Each has own `pyproject.toml`, set `package = false`
- **Cross-deps**: Use `{ workspace = true }` for internal packages

### Code Quality
- **Ruff**: Line length 100, target Python 3.11
- **MyPy**: Strict mode disabled for practicality
- **Testing**: pytest with asyncio support

### Project Structure
- **No nested workspaces**: `langgraph-templates/*` excluded from workspace
- **Skills pattern**: SKILL.md + references/ + scripts/ + assets/
- **Examples**: Each app has examples/ directory

## COMMANDS

```bash
# Install all dependencies
uv sync

# Run tests across workspace
uv run pytest

# Lint all code
uv run ruff check .

# Type check
uv run mypy .

# Run specific app
cd apps/framework && uv run python examples/tools_example.py
```

## NOTES

- **Python Version**: Requires 3.11+ (specified in root pyproject.toml)
- **Package Management**: UV only, no pip requirements files at root
- **Test Discovery**: Automatically finds tests in apps/ and packages/
- **Skill System**: Modular skills with progressive context loading
