# MCP Agent 实现方式最终对比

## 📊 代码量总览

| 实现方式 | 总行数 | 实际代码* | 相比原始节省 | 推荐使用 |
|---------|--------|----------|-------------|----------|
| **原始 McpAgent** | 1,060 行 | 737 行 | - | ⭐⭐ |
| **LangChain AgentExecutor** | 307 行 | 233 行 | **-68%** | ⭐⭐⭐ |
| **LangGraph 手动实现 (v1)** | 584 行 | 351 行 | **-52%** | ⭐⭐ |
| **LangGraph 官方适配器 (v2)** ⭐ | 446 行 | 277 行 | **-62%** | ⭐⭐⭐⭐⭐ |

*实际代码 = 去除注释、空行、文档字符串后的行数

---

## 🎯 最终推荐：LangGraph v2 (官方适配器)

### 核心优势

```python
# 使用 MultiServerMCPClient (官方包)
from langchain_mcp_adapters.client import MultiServerMCPClient

# 一行配置多个服务器
client = MultiServerMCPClient({
    "math": {"transport": "stdio", "command": "python", "args": ["math.py"]},
    "weather": {"transport": "http", "url": "http://localhost:8000/mcp"},
})

# 自动获取所有工具
tools = await client.get_tools()  # 搞定！
```

### 与手动实现的对比

| 功能 | v1 手动实现 | v2 官方适配器 |
|------|------------|---------------|
| **代码量** | 584 行 | 446 行 (-24%) |
| **工具转换** | 手动 MCPToolManager | 自动 |
| **连接管理** | 手动 stdio_client | 自动 |
| **多服务器** | 需自己实现 | ✅ 原生支持 |
| **传输方式** | 手动处理 | ✅ 自动支持 stdio/http/sse |
| **错误处理** | 手动 | ✅ 内置重试 |
| **拦截器** | ❌ | ✅ 认证、日志、限流 |
| **维护性** | 自己维护 | ✅ 官方维护 |

---

## 📝 四种实现的核心代码对比

### 1️⃣ 原始实现 (1,060 行)

```python
class McpAgent:
    def __init__(self, ...):
        # 手动处理多种传输方式
        if server_url:
            # HTTP transport with headers support
            transport = StreamableHttpTransport(url, headers)
            self._client = Client(transport)
        elif command and args:
            # Stdio transport
            transport = PythonStdioTransport(...)
            self._client = Client(transport)

    async def _list_tools(self):
        # 手动获取工具
        async with self._client as client:
            tools = await client.list_tools()
            return [手动转换格式 for tool in tools]

    async def _call_mcp_tool(self, name, args):
        # 手动调用工具
        async with self._client as client:
            result = await client.call_tool(name, args)
            return 手动解析结果

    def invoke(self, input):
        # 手动实现 ReAct 循环 (100+ 行)
        for step in range(max_steps):
            response = self.model.generate(...)
            if response.tool_calls:
                for tc in response.tool_calls:
                    result = self._execute_tool(tc)
            ...
```

### 2️⃣ LangChain AgentExecutor (307 行)

```python
class LangChainMcpAgent:
    async def _initialize(self):
        # MCP 连接
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                toolkit = MCPToolkit(session)
                tools = await toolkit.get_tools()

        # 一行创建 Agent
        agent = create_openai_tools_agent(llm, tools, prompt)
        self.executor = AgentExecutor(agent=agent, tools=tools)

    async def ainvoke(self, input):
        # 一行执行
        return await self.executor.ainvoke({"input": input})
```

### 3️⃣ LangGraph v1 手动实现 (584 行)

```python
class MCPToolManager:
    """手动管理 MCP 连接和工具转换"""
    async def initialize(self):
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                mcp_tools = await session.list_tools()
                # 手动转换每个工具
                self._tools = [
                    self._convert_mcp_tool(session, tool)
                    for tool in mcp_tools.tools
                ]

    def _convert_mcp_tool(self, session, mcp_tool):
        # 手动构建 StructuredTool
        async def tool_func(**kwargs):
            result = await session.call_tool(mcp_tool.name, kwargs)
            return 手动解析内容

        return StructuredTool.from_function(...)

# 构建 StateGraph
builder = StateGraph(State)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))
builder.add_conditional_edges("agent", route_agent_output)
```

### 4️⃣ LangGraph v2 官方适配器 ⭐ (446 行)

```python
class LangGraphMcpAgentV2:
    def __init__(self, servers):
        # 一行配置多服务器
        self._client = MultiServerMCPClient(servers)

    async def _initialize(self):
        # 一行获取所有工具
        self._tools = await self._client.get_tools()

        # 标准 LangGraph 构建
        builder = StateGraph(State)
        builder.add_node("agent", AgentNode())
        builder.add_node("tools", ToolNode(self._tools))
        builder.add_conditional_edges("agent", tools_condition)
        self._graph = builder.compile()
```

---

## 🔥 LangGraph v2 的高级特性

### 多服务器支持

```python
agent = LangGraphMcpAgentV2({
    "math": {
        "transport": "stdio",
        "command": "python",
        "args": ["math_server.py"],
    },
    "weather": {
        "transport": "http",
        "url": "http://localhost:8000/mcp",
        "headers": {"Authorization": "Bearer token"},
    },
    "search": {
        "transport": "sse",
        "url": "http://search.com/events",
    }
})
```

### 拦截器 (认证、日志、重试)

```python
from langchain_mcp_adapters.interceptors import MCPToolCallRequest

async def auth_interceptor(request: MCPToolCallRequest, handler):
    # 添加认证信息
    request.headers["Authorization"] = f"Bearer {api_key}"
    return await handler(request)

async def retry_interceptor(request, handler, max_retries=3):
    # 自动重试
    for attempt in range(max_retries):
        try:
            return await handler(request)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)

client = MultiServerMCPClient(
    {...},
    tool_interceptors=[auth_interceptor, retry_interceptor]
)
```

### 进度通知

```python
async def on_progress(progress: float, total: float, message: str, context):
    percent = (progress / total * 100) if total else 0
    print(f"[{context.server_name}] {percent:.1f}% - {message}")

client = MultiServerMCPClient(
    {...},
    callbacks=Callbacks(on_progress=on_progress)
)
```

---

## 📈 适用场景决策树

```
需要 MCP Agent?
│
├─ 完全控制底层 → 原始 McpAgent
│
├─ 快速开发，标准用例 → LangChain AgentExecutor
│
│
└─ 生产环境，需要扩展性 → LangGraph
    │
    ├─ 学习目的/特殊需求 → v1 手动实现
    │
    └─ 推荐做法 → v2 官方适配器 ⭐
        ├─ 多服务器支持
        ├─ 拦截器扩展
        ├─ 官方维护
        └─ 更好的生态集成
```

---

## 📦 依赖对比

### 原始实现
```toml
[dependencies]
fastmcp = "*"
openai = "*"
python-dotenv = "*"
```

### LangChain
```toml
[dependencies]
langchain = "*"
langchain-openai = "*"
langchain-mcp = "*"  # 第三方包
mcp = "*"
python-dotenv = "*"
```

### LangGraph v2 (推荐)
```toml
[dependencies]
langgraph = "*"
langchain-openai = "*"
langchain-mcp-adapters = "*"  # 官方包 ⭐
python-dotenv = "*"
```

---

## ✅ 最终建议

| 场景 | 推荐方案 | 理由 |
|------|---------|------|
| **快速原型** | LangChain AgentExecutor | 代码最少，快速验证 |
| **生产系统** | **LangGraph v2** ⭐ | 可扩展、可维护、官方支持 |
| **学习/教学** | 原始实现 → v2 | 先理解原理，再用标准方案 |
| **多 MCP 服务器** | **LangGraph v2** ⭐ | 原生多服务器支持 |
| **需要拦截器** | **LangGraph v2** ⭐ | 认证、日志、重试内置 |
| **复杂工作流** | **LangGraph v2** ⭐ | 状态机 + Human-in-the-loop |

---

## 🚀 快速开始 (LangGraph v2)

```python
# 1. 安装依赖
# uv add langgraph langchain-openai langchain-mcp-adapters

# 2. 创建 Agent
from apps.engineer.learn.agent.langgraph_mcp_agent_v2 import LangGraphMcpAgentV2

agent = LangGraphMcpAgentV2({
    "myserver": {
        "transport": "stdio",
        "command": "python",
        "args": ["server.py"],
    }
})

# 3. 运行
result = await agent.ainvoke("What is 125 * 301?")
```

---

## 📚 参考资源

- **官方文档**: https://docs.langchain.com/oss/python/langchain/mcp
- **GitHub**: https://github.com/langchain-ai/langchain-mcp-adapters
- **MCP 规范**: https://modelcontextprotocol.io/
- **LangGraph 文档**: https://langchain-ai.github.io/langgraph/

---

**结论**: 对于新项目，强烈建议使用 **LangGraph v2 (官方适配器)**，它在代码简洁性、功能丰富度和长期维护性之间取得了最佳平衡。
