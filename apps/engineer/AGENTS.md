# Engineer - AI Engineer Framework

**Path**: `apps/engineer/`  
**Type**: Python AI Engineering Framework  
**Stack**: Python 3.11+, LangChain, Pydantic

## OVERVIEW

Comprehensive AI engineer toolkit with agent framework, memory systems, tool management, and LLM providers. Supports multiple agent patterns (ReAct, Reflection, PlanAndSolve) and memory types.

## STRUCTURE

```
engineer/
├── engineer/              # Main package
│   ├── core/             # Core framework
│   │   ├── agents/      # Agent implementations
│   │   ├── tools/       # Tool system
│   │   ├── memory/      # Memory management
│   │   ├── providers/   # LLM providers
│   │   └── pattern/     # Agent patterns
│   ├── rag/             # RAG implementation
│   ├── algorithm/       # Algorithms
│   └── data_structure/  # Data structures
├── examples/            # Usage examples
└── test/               # Test suite
```

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Add agent | `core/agents/` | Extend `BaseAgent` |
| Add tool | `core/tools/` | Extend `BaseTool` |
| Add memory | `core/memory/` | Extend `BaseMemory` |
| Add LLM provider | `core/providers/` | Implement provider interface |
| Add pattern | `core/pattern/` | ReAct, Reflection, etc. |
| Run examples | `examples/` | See `tools_example.py` |

## CORE CLASSES

### Agents
- `TravelAssistantAgent` - Travel assistant
- `CodeAgent` - Code generation
- `SQLAgent` - SQL operations
- `BaseAgent` - Base class

### Tools
- `BaseTool` - Tool base class
- `ToolManager` - Tool registry
- `ToolExecutor` - Tool execution

### Memory
- `BaseMemory` - Memory base
- `BufferMemory` - Short-term
- `VectorMemory` - Vector storage
- `EntityMemory` - Entity tracking

### LLM Providers
- `OpenAILLM` - OpenAI models
- `AnthropicLLM` - Claude models
- `QwenLLM` - Alibaba Qwen
- `OllamaLLM` - Local models

## CONVENTIONS

### Code Patterns
- Use Pydantic for all data models
- Async/await for I/O operations
- Abstract base classes for extensibility
- Factory pattern for model creation

### Agent Pattern
```python
class MyAgent(BaseAgent):
    def run(self, query: str) -> str:
        # Implementation
        pass
```

### Tool Pattern
```python
class MyTool(BaseTool):
    name = "my_tool"
    def execute(self, **kwargs) -> ToolResult:
        # Implementation
        pass
```

## COMMANDS

```bash
# Run examples
uv run python examples/tools_example.py
uv run python examples/llm_integration_example.py

# Run specific coder
uv run python framework/core/agents/coder.py

# Run tests
uv run pytest test/
```

## NOTES

- **No main.py**: Use examples/ for entry points
- **LLM Factory**: Use `chat_model_factory.py` for model selection
- **Environment**: Use `.env` for API keys
- **Extensible**: Easy to add new agents, tools, providers
