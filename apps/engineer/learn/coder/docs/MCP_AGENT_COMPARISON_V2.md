# MCP Agent 三种实现对比

## 📊 代码量总览

| 实现方式 | 总行数 | 实际代码* | 相比原始节省 |
|---------|--------|----------|-------------|
| **原始 McpAgent** | 1,060 行 | 737 行 | - |
| **LangChain AgentExecutor** | 346 行 | 233 行 | **~68%** |
| **LangGraph StateGraph** | 584 行 | 351 行 | **~52%** |

*实际代码 = 去除注释、空行、文档字符串后的行数

---

## 🏗️ 架构对比

### 1. 原始 McpAgent (手动实现)

```
┌─────────────────────────────────────────────────────────────┐
│  McpAgent (手动实现一切)                                      │
├─────────────────────────────────────────────────────────────┤
│  ├─ 自定义 Model 类 (OpenAI 包装)                              │
│  ├─ 手动 MCP 连接管理 (stdio/http)                            │
│  ├─ 手动工具获取和转换                                        │
│  ├─ 自定义 ReAct 循环                                         │
│  ├─ 手动流式处理 (delta 累积)                                  │
│  ├─ 手动消息历史管理                                          │
│  └─ 工具执行和结果格式化                                      │
└─────────────────────────────────────────────────────────────┘
```

**特点：**
- ✅ 完全控制每个细节
- ✅ 无外部依赖（除 fastmcp/openai）
- ❌ 代码量大，维护成本高
- ❌ 需要自己处理边界情况

---

### 2. LangChain AgentExecutor (声明式)

```
┌─────────────────────────────────────────────────────────────┐
│  LangChainMcpAgent (声明式 Agent)                             │
├─────────────────────────────────────────────────────────────┤
│  ├─ ChatOpenAI (内置 LLM)                                    │
│  ├─ MCPToolkit (工具自动转换)                                  │
│  ├─ create_openai_tools_agent (自动工具调用)                   │
│  ├─ AgentExecutor (自动 ReAct 循环)                            │
│  └─ 内置流式支持                                              │
└─────────────────────────────────────────────────────────────┘
```

**代码核心 (约 30 行):**
```python
# 初始化
self.llm = ChatOpenAI(model=model)
toolkit = MCPToolkit(session)
tools = await toolkit.get_tools()

# 创建 Agent
agent = create_openai_tools_agent(self.llm, tools, prompt)
self.executor = AgentExecutor(agent=agent, tools=tools)

# 执行 (一行搞定)
result = await self.executor.ainvoke({"input": input})
```

**特点：**
- ✅ 代码最简洁
- ✅ 成熟稳定
- ✅ 自动处理复杂逻辑
- ❌ 黑盒，可控性较低
- ❌ 难以自定义中间步骤

---

### 3. LangGraph StateGraph (状态机)

```
┌─────────────────────────────────────────────────────────────┐
│  LangGraphMcpAgent (显式状态机)                               │
├─────────────────────────────────────────────────────────────┤
│  StateGraph (图结构)                                          │
│  │                                                            │
│  ├─ Node: agent (AgentNode) ──► 调用 LLM                      │
│  ├─ Node: tools (ToolNode)  ──► 执行 MCP 工具                  │
│  │                                                            │
│  └─ Edges:                                                   │
│      ├─ start → agent                                        │
│      ├─ agent → [条件路由] → tools / end                      │
│      └─ tools → agent (循环)                                  │
└─────────────────────────────────────────────────────────────┘
```

**代码核心 (约 60 行):**
```python
# 定义状态
@dataclass
class State:
    messages: Annotated[Sequence[AnyMessage], add_messages]

# 定义节点
async def agent_node(state: State):
    response = await llm.ainvoke(state.messages)
    return {"messages": [response]}

# 构建图
builder = StateGraph(State)
builder.add_node("coder", agent_node)
builder.add_node("tools", ToolNode(tools))
builder.add_edge("__start__", "coder")
builder.add_conditional_edges("coder", route_agent_output)
builder.add_edge("tools", "coder")

graph = builder.compile()
```

**特点：**
- ✅ 显式控制流，易于理解
- ✅ 可视化图结构
- ✅ 支持 Human-in-the-loop
- ✅ 可持久化状态
- ✅ 易于扩展复杂工作流
- ❌ 代码量比 AgentExecutor 多
- ❌ 学习曲线略陡

---

## 🔍 详细功能对比

| 功能 | 原始实现 | LangChain | LangGraph |
|------|---------|-----------|-----------|
| **代码行数** | 1,060 | 346 (-67%) | 584 (-45%) |
| **同步调用** | ✅ | ✅ | ✅ |
| **异步调用** | ✅ | ✅ | ✅ |
| **流式输出** | ✅ 手动 | ✅ 自动 | ✅ 自动 |
| **工具循环** | 手动实现 | AgentExecutor | StateGraph |
| **状态管理** | 手动 | 自动 | 显式 State |
| **人类介入** | ❌ | ❌ | ✅ interrupt |
| **持久化** | ❌ | ❌ | ✅ checkpoint |
| **可视化** | ❌ | ❌ | ✅ LangGraph Studio |
| **自定义路由** | 需修改代码 | 困难 | ✅ 条件边 |
| **LLM 切换** | 需改 Model 类 | ✅ 一行 | ✅ 一行 |
| **调试可见性** | 手动打印 | 中间步骤 | ✅ 完整 trace |

---

## 📈 性能与扩展性

### 简单任务 (单工具调用)

| 指标 | 原始 | LangChain | LangGraph |
|------|------|-----------|-----------|
| 初始化速度 | ⚡ 快 | 🐢 慢 (需加载工具) | 🐢 慢 (需编译图) |
| 执行速度 | ⚡ 快 | ⚡ 快 | ⚡ 快 |
| 内存占用 | 低 | 中 | 中 |
| 代码可读性 | 差 | 好 | 很好 |

### 复杂任务 (多步骤 + 人机交互)

| 指标 | 原始 | LangChain | LangGraph |
|------|------|-----------|-----------|
| 实现难度 | 😰 高 | 😐 中 | 😊 低 |
| 人机介入 | 🔧 手动实现 | ❌ 不支持 | ✅ 内置支持 |
| 流程分支 | 🔧 手动实现 | 😰 困难 | ✅ 条件边 |
| 状态恢复 | 🔧 手动实现 | ❌ 不支持 | ✅ 内置支持 |

---

## 🎯 选择建议

### 选择原始 McpAgent 当：
- 需要极致的性能优化
- 有特殊的传输层需求
- 需要完全控制每个字节
- 项目不允许额外依赖

### 选择 LangChain AgentExecutor 当：
- 追求最快开发速度
- 标准 ReAct 模式足够
- 不需要复杂工作流
- 团队熟悉 LangChain

### 选择 LangGraph StateGraph 当：
- 需要复杂的多步骤工作流
- 可能需要人机协作
- 需要状态持久化
- 希望可视化执行过程
- 计划扩展为更复杂的 Agent

---

## 📝 代码示例对比

### 简单查询: "What is 125 * 301?"

**原始实现:**
```python
agent = McpAgent(name="Agent", model=Model(), args=[server])
result = agent.invoke("What is 125 * 301?")  # 内部手动循环
```

**LangChain:**
```python
agent = LangChainMcpAgent(args=[server])
result = await agent.ainvoke("What is 125 * 301?")  # Executor 处理一切
```

**LangGraph:**
```python
agent = LangGraphMcpAgent(args=[server])
result = await agent.ainvoke("What is 125 * 301?")  # Graph 执行

# 或手动控制每一步
async for event in agent.astream("What is 125 * 301?"):
    print(event)  # 可见每个节点执行
```

---

## 🔮 未来扩展性

### LangGraph 的独特优势

```python
# 添加人机介入点 (只需一行)
graph = builder.compile(
    interrupt_before=["tools"],  # 工具执行前暂停，等待人类确认
)

# 添加持久化 (只需一行)
from langgraph.checkpoint import MemorySaver
graph = builder.compile(checkpointer=MemorySaver())

# 添加条件分支 (可视化路由)
builder.add_conditional_edges(
    "coder",
    lambda state: "tools" if should_call_tools(state) else "__end__"
)
```

---

## 📦 依赖对比

**原始:**
```toml
dependencies = ["fastmcp", "openai", "python-dotenv"]
```

**LangChain:**
```toml
dependencies = [
    "langchain", "langchain-openai", "langchain-mcp",
    "mcp", "python-dotenv"
]
```

**LangGraph:**
```toml
dependencies = [
    "langgraph", "langchain-openai",
    "mcp", "python-dotenv"
]
```

---

## ✅ 总结

| 维度 | 推荐方案 |
|------|---------|
| **快速原型** | LangChain AgentExecutor |
| **生产系统** | LangGraph StateGraph |
| **学习/教学** | 原始实现 → LangGraph |
| **复杂工作流** | LangGraph (唯一选择) |
| **性能敏感** | 原始实现 或 LangChain |

**我们的建议:**
- 如果项目刚起步 → 使用 **LangGraph** (更多可能性)
- 如果追求简单 → 使用 **LangChain AgentExecutor**
- 如果需要完全控制 → 保留 **原始实现** 作为参考
