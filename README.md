# AI Agents Monorepo

A comprehensive monorepo containing AI agent applications, tools, and learning resources.

## 🏗️ Project Structure

```
agents/
├── apps/                    # Applications and examples
│   ├── coze-agent/         # Coze platform agent examples (day-by-day tutorials)
│   ├── engineer/           # AI engineer tools and utilities
│   ├── huggingface/        # HuggingFace integration examples
│   ├── langgraph/          # LangGraph templates (RAG, Memory, ReAct)
│   ├── skills/             # Agent skills collection
│   └── examples/           # General AI framework examples
│
├── packages/               # Shared packages (to be developed)
│   └── (core utilities, shared types, etc.)
│
├── docs/                   # Documentation
├── pyproject.toml          # Root workspace configuration
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) - Python package manager

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd agents

# Install dependencies for all workspace members
uv sync

# Or install for a specific app
uv sync --package framework
```

### Running Applications

Each app in `apps/` has its own dependencies and can be run independently:

```bash
# Run framework app
cd apps/framework
uv run python -m src.main

# Run huggingface example
cd apps/huggingface
uv run python src/main.py
```

## 📦 Workspace Configuration

This monorepo uses [uv workspaces](https://docs.astral.sh/uv/concepts/workspaces/) for dependency management:

- **Root `pyproject.toml`**: Defines workspace members and shared dev dependencies
- **Each app**: Has its own `pyproject.toml` with specific dependencies
- **Shared packages**: Can be added to `packages/` for code reuse across apps

## 🛠️ Development

### Code Quality

```bash
# Lint all coder
uv run ruff check .

# Format coder
uv run ruff format .

# Type checking
uv run mypy .

# Run tests
uv run pytest
```

### Adding a New App

1. Create a new directory in `apps/`
2. Add a `pyproject.toml` with `[project]` metadata
3. Set `[tool.uv] package = false` for applications
4. Run `uv sync` to update the workspace

## 📚 Apps Overview

| App | Description |
|-----|-------------|
| `coze-agent` | Day-by-day Coze platform learning examples |
| `engineer` | AI-powered engineering tools and utilities |
| `huggingface` | HuggingFace Transformers integration examples |
| `langgraph-templates` | LangGraph agent patterns: RAG, Memory, ReAct |
| `skills` | Reusable agent skills and components |
| `examples` | General AI framework examples and snippets |

## 📄 License

See [LICENSE](LICENSE) for details.
