# Agent开发知识

## 目录
- [Agent核心概念](#agent核心概念)
- [LangChain框架](#langchain框架)
- [Agent类型与模式](#agent类型与模式)
- [工具调用](#工具调用)
- [Multi-Agent系统](#multi-agent系统)
- [Agent记忆机制](#agent记忆机制)
- [Agent部署与监控](#agent部署与监控)
- [学习阶段规划](#学习阶段规划)
- [推荐资源](#推荐资源)

## Agent核心概念

### 什么是Agent
**Agent（智能体）**是一个能够：
- **感知环境**：接收用户输入、系统状态
- **推理决策**：理解任务、规划步骤
- **执行动作**：调用工具、生成文本
- **自我反思**：评估结果、调整策略

### Agent vs 传统应用
| 特性 | 传统应用 | AI Agent |
|------|---------|----------|
| 决策方式 | 硬编码逻辑 | 动态推理 |
| 灵活性 | 固定流程 | 自适应 |
| 复杂度处理 | 需预定义 | 能应对未知 |
| 人机交互 | 命令式 | 对话式 |

### Agent核心组件
```
┌─────────────────────────────────┐
│          用户请求               │
└────────────┬────────────────────┘
             ↓
┌─────────────────────────────────┐
│         Agent Core              │
│  ┌─────────┐  ┌────────────┐  │
│  │ LLM大脑  │←→│ Planner    │  │
│  └─────────┘  └────────────┘  │
│       ↓             ↓          │
│  ┌─────────┐  ┌────────────┐  │
│  │ Memory  │←→│ Tool调用器  │  │
│  └─────────┘  └────────────┘  │
└───────┬────────────────┬───────┘
        ↓                ↓
┌──────────┐      ┌───────────┐
│ 外部工具  │      │ 数据存储   │
└──────────┘      └───────────┘
```

## LangChain框架

### 核心概念
```python
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate

# 1. 初始化LLM
llm = ChatOpenAI(model="gpt-4")

# 2. 定义工具
def search_web(query: str) -> str:
    """搜索网络"""
    # 实际实现：调用搜索API
    return f"搜索结果：{query}"

tools = [
    Tool(
        name="Search",
        func=search_web,
        description="用于搜索网络信息"
    )
]

# 3. 创建Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的助手"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

# 4. 创建Agent
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

# 5. 运行
result = agent_executor.invoke({"input": "搜索Python的最新版本"})
print(result["output"])
```

### LangChain核心组件
1. **Models**：LLM、Chat Model、Embeddings
2. **Prompts**：PromptTemplate、ChatPromptTemplate
3. **Memory**：对话历史、长期记忆
4. **Chains**：链式调用多个组件
5. **Agents**：基于工具的自主决策
6. **Tools**：可调用的外部功能
7. **Retrievers**：信息检索

### LangChain vs Spring
| 概念 | LangChain | Spring |
|------|-----------|--------|
| 依赖注入 | 链式组装 | IOC容器 |
| 中间件 | Chain | Interceptor |
| 工具调用 | Tool | Service |
| 配置 | Python Config | @Configuration |

## Agent类型与模式

### 1. ReAct Agent（推理+行动）
```python
from langchain.agents import create_react_agent

# 思考-行动-观察循环
prompt = """
Answer the following questions as best you can.
You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""
```

### 2. Plan-and-Execute Agent
```python
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner

planner = load_chat_planner(llm)
executor = load_agent_executor(llm, tools, verbose=True)

agent = PlanAndExecute(planner=planner, executor=executor)
result = agent.invoke("帮我制定一个学习Python的计划")
```

### 3. 自定义Agent（适合复杂场景）
```python
from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, END
from operator import add

class AgentState(TypedDict):
    messages: Annotated[List[str], add]
    next_step: str

def research_step(state: AgentState):
    """研究步骤"""
    llm = ChatOpenAI()
    response = llm.invoke(state["messages"][-1])
    return {"messages": [response.content], "next_step": "analysis"}

def analysis_step(state: AgentState):
    """分析步骤"""
    # 分析研究结果
    return {"next_step": "report"}

def report_step(state: AgentState):
    """报告步骤"""
    # 生成最终报告
    return {"next_step": END}

# 构建图
workflow = StateGraph(AgentState)
workflow.add_node("research", research_step)
workflow.add_node("analysis", analysis_step)
workflow.add_node("report", report_step)
workflow.set_entry_point("research")

# 添加边
workflow.add_conditional_edges(
    "research",
    lambda x: x["next_step"],
    {"analysis": "analysis"}
)
workflow.add_edge("analysis", "report")
workflow.add_edge("report", END)

# 编译Agent
agent = workflow.compile()
```

## 工具调用

### 基础工具定义
```python
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(description="搜索查询")

def search_tool(query: str) -> str:
    """搜索工具"""
    # 实际实现：调用搜索API
    return f"关于'{query}'的搜索结果"

tool = StructuredTool.from_function(
    func=search_tool,
    name="search",
    description="搜索网络信息",
    args_schema=SearchInput
)
```

### 工具类型
1. **API工具**：调用外部服务
```python
def call_weather_api(location: str) -> str:
    response = requests.get(f"https://api.weather.com/{location}")
    return response.json()["weather"]
```

2. **数据库工具**：查询数据库
```python
from sqlalchemy import create_engine, text

def query_database(sql: str) -> str:
    engine = create_engine("sqlite:///data.db")
    with engine.connect() as conn:
        result = conn.execute(text(sql))
        return str(result.fetchall())
```

3. **文件操作工具**：读写文件
```python
def read_file(file_path: str) -> str:
    with open(file_path, "r") as f:
        return f.read()

def write_file(file_path: str, content: str) -> str:
    with open(file_path, "w") as f:
        f.write(content)
    return f"已写入{file_path}"
```

4. **计算工具**：数学计算
```python
import numpy as np

def calculate_stats(numbers: list) -> dict:
    return {
        "mean": np.mean(numbers),
        "std": np.std(numbers),
        "median": np.median(numbers)
    }
```

### 工具最佳实践
1. **清晰的描述**：让LLM知道何时使用
2. **参数验证**：使用Pydantic验证输入
3. **错误处理**：友好的错误信息
4. **超时控制**：避免长时间等待
5. **返回格式**：结构化JSON输出

## Multi-Agent系统

### 架构模式

#### 1. 协作模式（Cooperation）
```python
# 多个Agent协作完成复杂任务
researcher = Agent(
    role="研究员",
    goal="收集和分析信息"
)

writer = Agent(
    role="作家",
    goal="根据研究内容撰写文章"
)

reviewer = Agent(
    role="审稿人",
    goal="审核文章质量"
)

# 任务流
research_results = researcher.run("研究AI Agent的发展历史")
draft = writer.run(research_results)
final_article = reviewer.run(draft)
```

#### 2. 竞争模式（Competition）
```python
# 多个Agent产生多个方案，由仲裁者选择
agent_a = Agent(model="gpt-4")
agent_b = Agent(model="claude-3")
agent_c = Agent(model="gemini-pro")

arbitrator = Agent()

solutions = [
    agent_a.solve(problem),
    agent_b.solve(problem),
    agent_c.solve(problem)
]

best_solution = arbitrator.evaluate(solutions)
```

#### 3. 分工模式（Division）
```python
# 不同Agent负责不同领域
code_agent = Agent(specialty="编程")
design_agent = Agent(specialty="设计")
business_agent = Agent(specialty="业务")

team = Team([
    code_agent,
    design_agent,
    business_agent
])

result = team.collaborate("开发一个电商网站")
```

### LangGraph Multi-Agent
```python
from langgraph.graph import StateGraph, END
from typing import Annotated, List
from operator import add

class TeamState(TypedDict):
    messages: Annotated[List[str], add]
    next_agent: str

def agent_a(state: TeamState):
    """Agent A处理"""
    llm = ChatOpenAI()
    response = llm.invoke(state["messages"][-1])
    return {
        "messages": [f"Agent A: {response.content}"],
        "next_agent": "agent_b"
    }

def agent_b(state: TeamState):
    """Agent B处理"""
    llm = ChatOpenAI()
    response = llm.invoke(state["messages"][-1])
    return {
        "messages": [f"Agent B: {response.content}"],
        "next_agent": "agent_c"
    }

def agent_c(state: TeamState):
    """Agent C处理并总结"""
    llm = ChatOpenAI()
    summary = llm.invoke("\n".join(state["messages"]))
    return {
        "messages": [f"总结: {summary.content}"],
        "next_agent": END
    }

# 构建Multi-Agent工作流
workflow = StateGraph(TeamState)
workflow.add_node("agent_a", agent_a)
workflow.add_node("agent_b", agent_b)
workflow.add_node("agent_c", agent_c)

workflow.add_conditional_edges(
    "agent_a",
    lambda x: x["next_agent"],
    {"agent_b": "agent_b"}
)
workflow.add_conditional_edges(
    "agent_b",
    lambda x: x["next_agent"],
    {"agent_c": "agent_c"}
)
workflow.add_edge("agent_c", END)

multi_agent = workflow.compile()
```

### Multi-Agent vs 微服务
| 对比 | Multi-Agent | 微服务 |
|------|-------------|--------|
| 通信方式 | 自然语言对话 | HTTP/RPC |
| 灵活性 | 动态决策 | 固定接口 |
| 协调方式 | 自然协调 | API Gateway |
| 部署 | 运行时组合 | 部署时组合 |

## Agent记忆机制

### 记忆类型
1. **短期记忆**：对话上下文
```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 添加记忆
memory.chat_memory.add_user_message("你好")
memory.chat_memory.add_ai_message("你好！有什么可以帮你的？")

# 获取记忆
history = memory.load_memory_variables({})
```

2. **长期记忆**：向量存储
```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# 初始化向量存储
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(
    collection_name="agent_memories",
    embedding_function=embeddings
)

# 存储记忆
vectorstore.add_texts(["用户喜欢编程", "用户学习过Java"])

# 检索记忆
related_memories = vectorstore.similarity_search("用户的技术背景")
```

3. **总结记忆**：自动总结
```python
from langchain.memory import ConversationSummaryMemory

memory = ConversationSummaryMemory(
    llm=ChatOpenAI(),
    memory_key="summary"
)

# 自动总结对话
memory.save_context(
    {"input": "介绍一下Agent"},
    {"output": "Agent是能自主决策的AI系统..."}
)
```

### 记忆管理最佳实践
- **关键信息存储**：只存储重要信息
- **定期清理**：避免记忆溢出
- **上下文窗口管理**：控制输入长度
- **隐私保护**：敏感信息脱敏

## Agent部署与监控

### 部署方式

#### 1. FastAPI部署
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class AgentRequest(BaseModel):
    message: str
    conversation_id: str = None

# 全局Agent实例
agent = create_agent()

@app.post("/chat")
async def chat(request: AgentRequest):
    try:
        response = await agent.arun(request.message)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

#### 2. 流式响应
```python
from fastapi.responses import StreamingResponse

async def stream_response(message: str):
    async for chunk in agent.astream(message):
        yield chunk

@app.get("/stream")
async def stream_chat(message: str):
    return StreamingResponse(
        stream_response(message),
        media_type="text/plain"
    )
```

### 监控指标
1. **性能指标**
   - 响应时间
   - Token消耗
   - API调用次数
   - 错误率

2. **质量指标**
   - 用户满意度
   - 任务完成率
   - 意图识别准确率

3. **安全指标**
   - 敏感信息泄露
   - 恶意攻击检测
   - 内容合规性

### 日志记录
```python
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent")

def log_interaction(user_input: str, agent_response: str, metadata: dict):
    logger.info({
        "timestamp": datetime.now().isoformat(),
        "user_input": user_input,
        "agent_response": agent_response,
        "tokens_used": metadata.get("tokens", 0),
        "latency_ms": metadata.get("latency", 0)
    })
```

## 学习阶段规划

### 阶段1：Agent基础（3-4周）
- **目标**：理解Agent概念，能使用LangChain创建简单Agent
- **内容**：
  - Agent核心概念和架构（1周）
  - LangChain基础使用（1-2周）
  - 工具调用基础（1周）
- **实践任务**：
  - 创建一个简单问答Agent
  - 实现带工具的查询Agent
- **Java经验复用**：
  - 面向对象设计思维
  - 模块化开发经验
  - API设计经验

### 阶段2：高级Agent（4-6周）
- **目标**：掌握复杂Agent模式，能处理多步骤任务
- **内容**：
  - ReAct、Plan-Execute模式（1-2周）
  - 记忆机制（1-2周）
  - 自定义Agent开发（1-2周）
- **实践任务**：
  - 构建带记忆的个人助理Agent
  - 实现多步骤任务规划Agent
- **Java经验复用**：
  - 状态机设计
  - 异步处理经验
  - 数据持久化知识

### 阶段3：Multi-Agent（4-6周）
- **目标**：掌握多Agent协作，能设计复杂系统架构
- **内容**：
  - Multi-Agent模式（2周）
  - LangGraph使用（2周）
  - 协作与竞争机制（2周）
- **实践任务**：
  - 构建代码审查Multi-Agent系统
  - 实现团队协作模拟系统
- **Java经验复用**：
  - 微服务架构经验
  - 分布式系统知识
  - 消息队列经验

### 阶段4：生产化（3-4周）
- **目标**：掌握Agent部署和监控，能构建生产级应用
- **内容**：
  - FastAPI部署（1周）
  - 监控和日志（1周）
  - 性能优化（1-2周）
- **实践任务**：
  - 部署一个完整的Agent应用
  - 实现监控和告警系统
- **Java经验复用**：
  - 部署运维经验
  - 监控体系建设
  - 性能调优技巧

## 推荐资源

### 官方文档
- LangChain文档：https://python.langchain.com/
- LangGraph文档：https://langchain-ai.github.io/langgraph/
- OpenAI文档：https://platform.openai.com/docs

### 开源项目
- AutoGPT：自主Agent
- BabyAGI：任务规划Agent
- MetaGPT：多角色协作
- CrewAI：多Agent框架

### 学习资源
- LangChain官方教程
- 《Building AI Applications with LangChain》
- Andrew Ng《AI for Everyone》

### 实践平台
- Coze：低代码Agent平台
- Dify：开源Agent平台
- Flowise：可视化Agent构建
