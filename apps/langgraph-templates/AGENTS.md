# LangGraph Templates

**Path**: `apps/langgraph-templates/`  
**Type**: LangGraph Pattern Collection  
**Stack**: Python 3.11+, LangGraph, LangChain

## OVERVIEW

Collection of LangGraph agent templates demonstrating common patterns: RAG retrieval, memory management, ReAct reasoning, and data enrichment. Each template is self-contained with its own configuration.

## STRUCTURE

```
langgraph-templates/
├── rag/                      # RAG Retrieval Bot
│   └── retrieval_graph/     # Main graph implementation
├── memory-agent/            # ReAct with Memory
│   └── memory_agent/       # Memory-enabled agent
├── reAct/                   # ReAct Agent
│   └── react_agent/        # Reasoning + Acting
└── data-agent/              # Data Enrichment
    └── enrichment_agent/   # Data processing
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| RAG pattern | `rag/retrieval_graph/` | Retrieval + generation |
| Memory pattern | `memory-agent/memory_agent/` | Stateful conversations |
| ReAct pattern | `reAct/react_agent/` | Tool use + reasoning |
| Data enrichment | `data-agent/enrichment_agent/` | Data processing |
| Add template | Create new dir | Follow existing structure |

## CORE FILES (All Templates)

Each template contains:
- `graph.py` - Main graph definition
- `state.py` - State management (TypedDict)
- `configuration.py` - Configuration class
- `prompts.py` - System prompts
- `utils.py` - Helper functions
- `tools.py` - Agent tools (except RAG)

## TEMPLATE PATTERNS

### RAG Template
```python
# graph.py
builder = StateGraph(State)
builder.add_node("retrieve", retrieve_documents)
builder.add_node("generate", generate_response)
```

### Memory Template
```python
# state.py
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

### ReAct Template
```python
# graph.py
builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode(tools))
```

## CONVENTIONS

### Configuration
- Use `configuration.py` for all settings
- Support both Anthropic and OpenAI
- Environment variables via `.env`

### State Management
- TypedDict for type safety
- Annotation for reducers
- Clear state fields

### Graph Structure
- Explicit node definitions
- Clear edge conditions
- Entry/exit points defined

## COMMANDS

```bash
# Run specific template
cd rag && uv run python -m retrieval_graph

# Run with LangGraph Studio
# Use langgraph.json in each template

# Tests
uv run pytest <template>/tests/
```

## NOTES

- **Self-contained**: Each template has own dependencies
- **LangGraph Studio**: Configured via `langgraph.json`
- **Model support**: Anthropic Claude, OpenAI GPT
- **Testing**: Unit + integration tests included
