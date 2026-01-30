# LLM+Agent基础知识+Agent开发框架 12天精简学习计划（2小时/天）

## 学习计划概述

**学习目标**：高效掌握LLM基础知识和Agent开发框架的核心技能

**学习模式**：一天理论知识学习 + 一天实践操作

**总时长**：12天（6个学习周期）

**每日投入**：2小时（理论1小时 + 实践1小时）

**适用人群**：

- 有一定编程基础的开发者
- 希望快速掌握LLM和Agent核心技能的工程师

---

## 第一阶段：LLM基础入门（第1-6天）

### Day 1：LLM核心概念与API调用（理论）

#### 学习目标

- 理解LLM核心原理和关键参数
- 掌握API调用基础方法

#### 核心内容（1小时）

**1. LLM基础概念（30分钟）**

- Transformer架构简介
- Token机制和上下文窗口
- 关键参数：
    - Temperature（0-1：控制随机性）
    - Max Tokens（输出长度限制）
    - Top P（核采样，推荐0.9）
- 主流平台对比：
    - OpenAI：GPT-4（能力强、成本高）
    - 智谱AI：GLM-4（中文好、性价比高）
    - Claude：长文本、安全性好

**2. API调用基础（30分钟）**

```python
# OpenAI API调用示例
from openai import OpenAI

client = OpenAI(api_key="your-api-key")

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "你是一个有帮助的助手"},
        {"role": "user", "content": "解释什么是大语言模型"}
    ],
    temperature=0.7
)

print(response.choices[0].message.content)
```

**Java经验迁移**：

- HTTP客户端调用（如Java的HttpClient/OkHttp）
- 异常处理和重试机制
- API Key管理（类似Java的配置管理）

#### 学习资源

- OpenAI官方文档：https://platform.openai.com/docs
- 智谱AI文档：https://open.bigmodel.cn/dev/api

---

### Day 2：API调用与流式输出实践（实践）

#### 实践目标

- 成功调用LLM API
- 实现流式输出功能

#### 实践任务（1小时）

**任务1：环境搭建与简单调用（30分钟）**

```python
# 完整的聊天机器人代码
from openai import OpenAI
import os


class SimpleChatbot:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.history = []

    def chat(self, user_message: str) -> str:
        self.history.append({"role": "user", "content": user_message})

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=self.history,
            temperature=0.7
        )

        assistant_message = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": assistant_message})
        return assistant_message


# 测试
bot = SimpleChatbot()
print(bot.chat("你好，请介绍一下你自己"))
```

**任务2：实现流式输出（30分钟）**

```python
def stream_chat(prompt: str):
    for chunk in client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            stream=True
    ):
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()


# 测试流式输出
stream_chat("请写一首关于AI的诗")
```

#### 交付成果

- [ ] 可调用的聊天机器人代码
- [ ] 流式输出功能实现

---

### Day 3：提示工程与RAG原理（理论）

#### 学习目标

- 掌握提示工程的核心技巧
- 理解RAG的工作原理

#### 核心内容（1小时）

**1. 提示工程技巧（30分钟）**

**角色设定：**

```
你是一个有10年经验的Java架构师，精通微服务架构。
请帮我设计一个电商系统的技术方案。
```

**Few-shot学习（少样本学习）：**

```
任务：将自然语言转换为SQL查询

示例1：
输入：查询所有年龄大于30的用户
输出：SELECT * FROM users WHERE age > 30

示例2：
输入：查询销售额最高的5个产品
输出：SELECT * FROM products ORDER BY sales DESC LIMIT 5

输入：查询上个月注册的用户
输出：
```

**思维链（Chain of Thought）：**

```
问题：一个商店的苹果每个3元，橙子每个5元，小明买了3个苹果和2个橙子，一共花了多少钱？

请一步步思考：
1. 计算苹果的总价
2. 计算橙子的总价
3. 计算总价
```

**结构化输出：**

```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{
        "role": "system",
        "content": "请以JSON格式输出：{\"summary\": \"总结\", \"key_points\": [\"要点1\", \"要点2\"]}"
    }, {
        "role": "user",
        "content": "分析这段文本"
    }]
)
```

**2. RAG基本原理（30分钟）**

**RAG架构流程：**

```
用户问题
    ↓
向量检索 → 相关文档片段
    ↓                ↓
    └────→ LLM生成 ←──┘
          ↓
      回答用户
```

**核心组件：**

- 向量数据库：FAISS、Chroma、Pinecone
- Embeddings：将文本转换为向量表示
- 文档分块：合理切分（500-1000 tokens）

**RAG vs 直接调用LLM：**

- RAG：基于特定文档，回答更准确、可追溯来源
- 直接调用：基于训练数据，适合通用问答

#### 学习资源

- 《Prompt Engineering Guide》：https://www.promptingguide.ai/
- LangChain RAG教程

---

### Day 4：RAG系统基础实践（实践）

#### 实践目标

- 搭建一个简单的RAG系统
- 掌握文档加载和向量检索

#### 实践任务（1小时）

**任务1：搭建基础RAG系统（40分钟）**

```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader

# 1. 加载文档
loader = TextLoader("docs/sample.txt")
documents = loader.load()

# 2. 创建向量索引
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

# 3. 创建RAG链
llm = ChatOpenAI(model="gpt-4")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

# 4. 查询
query = "文档中提到了哪些主要内容？"
result = qa_chain.invoke({"query": query})
print(f"回答：{result['result']}")
print(f"来源：{len(result['source_documents'])} 个文档片段")
```

**任务2：测试和优化（20分钟）**

- 测试不同文档的查询效果
- 调整检索数量（k值）
- 尝试不同的文档分块策略

#### 交付成果

- [ ] 可运行的RAG问答系统
- [ ] 支持文档加载和检索

---

### Day 5：Agent核心概念与LangChain基础（理论）

#### 学习目标

- 理解Agent的定义和核心组件
- 掌握LangChain框架的基本概念

#### 核心内容（1小时）

**1. Agent核心概念（30分钟）**

**什么是Agent？**

- Agent是能够自主感知、推理、执行、反思的AI系统
- 与传统应用的区别：
    - 传统应用：固定流程、硬编码逻辑
    - Agent：动态决策、自适应、能应对未知

**Agent核心组件：**

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

**2. LangChain框架（30分钟）**

**核心组件：**

- Models：LLM、Chat Model、Embeddings
- Prompts：PromptTemplate、ChatPromptTemplate
- Memory：对话历史、长期记忆
- Chains：链式调用多个组件
- Agents：基于工具的自主决策
- Tools：可调用的外部功能
- Retrievers：信息检索

**LangChain vs Spring对比：**
| 概念 | LangChain | Spring |
|------|-----------|--------|
| 依赖注入 | 链式组装 | IOC容器 |
| 中间件 | Chain | Interceptor |
| 工具调用 | Tool | Service |
| 配置 | Python Config | @Configuration |

**Java经验迁移**：

- 面向对象设计思维（类、继承、多态）
- 模块化开发经验（组件化、解耦）
- API设计经验（RESTful、接口定义）

#### 学习资源

- LangChain官方文档：https://python.langchain.com/
- LangChain入门教程

---

### Day 6：简单Agent应用实践（实践）

#### 实践目标

- 创建一个带工具的Agent
- 实现Agent的基本功能

#### 实践任务（1小时）

**任务1：创建基础Agent（60分钟）**

```python
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate


# 1. 定义工具
def calculator(expression: str) -> str:
    """计算数学表达式"""
    try:
        result = eval(expression, {"__builtins__": {}}, {
            "add": lambda a, b: a + b,
            "sub": lambda a, b: a - b,
            "mul": lambda a, b: a * b,
            "div": lambda a, b: a / b
        })
        return str(result)
    except Exception as e:
        return f"计算错误: {str(e)}"


def search(query: str) -> str:
    """网络搜索（模拟）"""
    return f"关于'{query}'的搜索结果：\n1. 官方文档\n2. 教程\n3. 社区讨论"


tools = [
    Tool(name="calculator", func=calculator, description="执行数学计算"),
    Tool(name="search", func=search, description="搜索网络信息")
]

# 2. 创建Agent
llm = ChatOpenAI(model="gpt-4")
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的助理，可以使用工具"),
    ("user", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

agent = create_openai_functions_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 3. 测试
print(executor.invoke({"input": "计算 123 * 456"}))
print(executor.invoke({"input": "搜索Python的最新版本"}))
```

#### 交付成果

- [ ] 可运行的Agent应用
- [ ] 至少实现2个工具
- [ ] Agent能自主选择工具执行任务

---

## 第二阶段：Agent开发框架深入（第7-12天）

### Day 7：Multi-Agent系统与LangGraph（理论）

#### 学习目标

- 理解Multi-Agent系统的协作模式
- 掌握LangGraph的基本概念

#### 核心内容（1小时）

**1. Multi-Agent协作模式（30分钟）**

**协作模式：**

```
研究Agent → 内容 → 写作Agent → 初稿 → 审核Agent → 最终报告
```

- 多个Agent按顺序协作完成复杂任务

**竞争模式：**

```
问题 → Agent A ──┐
           Agent B ─→ 仲裁者 → 最佳方案
           Agent C ──┘
```

- 多个Agent产生方案，由仲裁者选择最优

**分工模式：**

```
          任务
            ↓
    ┌───────┴───────┐
    ↓               ↓
代码Agent        设计Agent        业务Agent
```

- 不同Agent负责不同领域

**Multi-Agent vs 微服务：**
| 对比 | Multi-Agent | 微服务 |
|------|-------------|--------|
| 通信方式 | 自然语言对话 | HTTP/RPC |
| 灵活性 | 动态决策 | 固定接口 |
| 协调方式 | 自然协调 | API Gateway |

**2. LangGraph框架（30分钟）**

**核心概念：**

- StateGraph：状态图，定义Agent工作流
- Node（节点）：Agent的处理逻辑
- Edge（边）：节点间的流转
- State（状态）：在Agent间传递的数据

**基本结构：**

```python
from langgraph.graph import StateGraph, END

# 1. 定义状态
from typing import TypedDict, List, Annotated
from operator import add


class TeamState(TypedDict):
    messages: Annotated[List[str], add]


# 2. 定义节点
def agent_a(state):
    # Agent A的处理逻辑
    return {"messages": ["Agent A 处理结果"]}


def agent_b(state):
    # Agent B的处理逻辑
    return {"messages": ["Agent B 处理结果"]}


# 3. 构建图
workflow = StateGraph(TeamState)
workflow.add_node("agent_a", agent_a)
workflow.add_node("agent_b", agent_b)

# 4. 添加边
workflow.add_edge("agent_a", "agent_b")
workflow.add_edge("agent_b", END)

# 5. 编译
app = workflow.compile()
```

**Java经验迁移：**

- 状态机设计（类似Spring State Machine）
- 工作流编排（类似Activiti/Camunda）
- 分布式协作（类似微服务架构）

#### 学习资源

- LangGraph文档：https://langchain-ai.github.io/langgraph/
- Multi-Agent系统论文

---

### Day 8：Multi-Agent系统实践（实践）

#### 实践目标

- 使用LangGraph构建简单的Multi-Agent系统
- 实现Agent间的协作

#### 实践任务（1小时）

**任务：构建代码审查Multi-Agent系统（60分钟）**

```python
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from typing import TypedDict, List, Annotated
from operator import add


# 1. 定义状态
class ReviewState(TypedDict):
    code: str
    analysis: str = ""
    security: str = ""
    final_report: str = ""


# 2. 定义Agent
llm = ChatOpenAI(model="gpt-4")


def analyzer_agent(state):
    """代码分析Agent"""
    prompt = f"分析以下代码的结构和逻辑：\n{state['code']}"
    response = llm.invoke(prompt)
    return {"analysis": response.content}


def security_agent(state):
    """安全检查Agent"""
    prompt = f"检查以下代码的安全问题：\n{state['code']}"
    response = llm.invoke(prompt)
    return {"security": response.content}


def reporter_agent(state):
    """报告生成Agent"""
    summary = f"""
    代码分析：
    {state['analysis']}

    安全检查：
    {state['security']}
    """
    response = llm.invoke(f"生成审查报告：\n{summary}")
    return {"final_report": response.content}


# 3. 构建工作流
workflow = StateGraph(ReviewState)
workflow.add_node("analyze", analyzer_agent)
workflow.add_node("security_check", security_agent)
workflow.add_node("report", reporter_agent)

# 添加边（串行执行）
workflow.add_edge("analyze", "security_check")
workflow.add_edge("security_check", "report")
workflow.add_edge("report", END)

# 编译
app = workflow.compile()

# 4. 执行
sample_code = """
def process_user_input(user_input: str):
    conn = sqlite3.connect('database.db')
    query = f"SELECT * FROM users WHERE name = '{user_input}'"
    conn.execute(query)
"""

result = app.invoke({"code": sample_code})
print(result["final_report"])
```

#### 交付成果

- [ ] 可运行的Multi-Agent系统
- [ ] 至少包含2个协作的Agent
- [ ] 实现状态流转和数据共享

---

### Day 9：AI应用架构设计（理论）

#### 学习目标

- 理解AI应用架构的特点
- 掌握主流架构模式

#### 核心内容（1小时）

**1. 传统架构 vs AI架构（20分钟）**

| 维度   | 传统应用  | AI应用    |
|------|-------|---------|
| 核心逻辑 | 硬编码规则 | 概率推理    |
| 数据处理 | 结构化为主 | 非结构化为主  |
| 延迟要求 | 毫秒级   | 秒级（可接受） |
| 可预测性 | 确定性   | 不确定性    |

**AI架构新挑战：**

- 不可预测性：模型输出不确定性
- 高延迟：LLM推理耗时较长
- 成本控制：Token消耗成本高
- 数据隐私：敏感数据处理

**2. AI应用架构模式（40分钟）**

**模式1：简单RAG架构**

```
用户请求 → API网关 → 向量检索服务 → LLM推理服务 → 返回结果
                       ↓
                    向量数据库
```

适用场景：文档问答、知识库检索

**模式2：Agent架构**

```
用户请求 → Agent编排层 → 工具调度层 → LLM层
                         ↓
                    记忆层
```

适用场景：复杂任务、多步骤流程

**模式3：Multi-Agent架构**

```
用户请求 → 协调器层
             ↓
    ┌────────┴────────┐
    ↓                 ↓
Agent A           Agent B
    ↓                 ↓
    └──────┬──────────┘
           ↓
    共享工具和记忆层
```

适用场景：复杂协作、分工明确

**Java经验迁移：**

- 分层架构思想（Controller → Service → DAO）
- API网关模式（类似Zuul/Gateway）
- 缓存策略（类似Redis使用）
- 监控告警体系（类似Prometheus/Grafana）

#### 学习资源

- 《Building LLM Applications for Production》
- Azure AI Architecture

---

### Day 10：架构设计实践（实践）

#### 实践目标

- 设计一个完整的AI应用架构
- 实现核心组件

#### 实践任务（1小时）

**任务1：设计架构（30分钟）**

选择以下场景之一，设计架构：

1. **企业知识库问答系统**
2. **智能客服系统**
3. **代码审查系统**

**架构设计要点：**

```
┌─────────────────────────────────┐
│         API Gateway             │
└────────────┬────────────────────┘
             ↓
┌─────────────────────────────────┐
│      FastAPI应用 (多实例)        │
│  ┌──────────┐  ┌──────────┐    │
│  │  RAG服务  │  │缓存服务  │    │
│  └──────────┘  └──────────┘    │
└──────┬──────────────────────────┘
       ↓
┌─────────────────────────────────┐
│  PostgreSQL  │  Redis  │  向量库 │
└─────────────────────────────────┘
```

**任务2：实现核心组件（30分钟）**

选择以下组件之一实现：

```python
# 选项1：带缓存的RAG服务
from functools import lru_cache


class CachedRAG:
    def __init__(self):
        self.vectorstore = None
        self.llm = ChatOpenAI()

    @lru_cache(maxsize=100)
    def query(self, question: str) -> str:
        # 检查缓存
        # 执行RAG查询
        pass


# 选项2：FastAPI接口
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()


class QueryRequest(BaseModel):
    question: str


@app.post("/query")
async def query(request: QueryRequest):
    try:
        # 实现查询逻辑
        result = rag.query(request.question)
        return {"answer": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

#### 交付成果

- [ ] 完整的架构设计图
- [ ] 至少实现一个核心组件

---

### Day 11：监控、日志与优化（理论）

#### 学习目标

- 理解AI应用的监控指标
- 掌握日志记录方法
- 了解优化策略

#### 核心内容（1小时）

**1. 监控指标（30分钟）**

**性能指标：**

- 响应时间：平均响应时间、P95/P99延迟
- Token消耗：每次请求的Token使用量
- API调用次数：请求频率、并发量
- 错误率：失败请求比例

**质量指标：**

- 用户满意度：用户评分、反馈
- 任务完成率：成功完成任务的比例
- 意图识别准确率：用户意图识别的正确率

**成本指标：**

- API成本：每月API调用费用
- Token成本：按Token计费
- 资源成本：服务器、数据库成本

**2. 日志记录（20分钟）**

**日志格式：**

```python
import logging
from datetime import datetime
import json


class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage()
        }
        return json.dumps(log_data)


logger = logging.getLogger("agent")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)
```

**关键日志：**

- 用户输入和Agent响应
- Token使用量
- 响应时间
- 错误信息

**3. 优化策略（10分钟）**

**成本优化：**

- 使用缓存减少重复调用
- 选择合适的模型（便宜模型+少量高精度模型）
- 缩短上下文长度

**性能优化：**

- 异步处理
- 流式输出
- 批量请求

**Java经验迁移：**

- 日志框架（类似Log4j/SLF4J）
- 监控体系（类似Prometheus/Grafana）
- 性能优化（类似异步处理、缓存）

#### 学习资源

- Prometheus文档：https://prometheus.io/docs/
- Grafana文档：https://grafana.com/docs/

---

### Day 12：监控与优化实践（实践）

#### 实践目标

- 实现基础的监控和日志功能
- 进行简单的性能优化

#### 实践任务（1小时）

**任务1：实现日志系统（30分钟）**

```python
import logging
from datetime import datetime
import json


class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage()
        }
        return json.dumps(log_data)


def setup_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    return logger


# 使用
logger = setup_logger("agent")
logger.info("Agent started")
logger.info("User input: hello")
```

**任务2：实现简单监控（20分钟）**

```python
from functools import lru_cache
from time import time
import json


class SimpleMonitor:
    def __init__(self):
        self.metrics = {
            "query_count": 0,
            "total_tokens": 0,
            "errors": 0
        }

    def log_query(self, tokens: int, error: bool = False):
        self.metrics["query_count"] += 1
        self.metrics["total_tokens"] += tokens
        if error:
            self.metrics["errors"] += 1

    def get_stats(self):
        return {
            "avg_tokens": self.metrics["total_tokens"] / self.metrics["query_count"] if self.metrics[
                                                                                            "query_count"] > 0 else 0,
            "error_rate": self.metrics["errors"] / self.metrics["query_count"] if self.metrics["query_count"] > 0 else 0
        }


# 使用
monitor = SimpleMonitor()
monitor.log_query(tokens=100)
print(json.dumps(monitor.get_stats(), indent=2))
```

**任务3：实现缓存优化（10分钟）**

```python
from functools import lru_cache
import hashlib


def get_cache_key(prompt: str) -> str:
    return hashlib.md5(prompt.encode()).hexdigest()


class CacheManager:
    def __init__(self):
        self.cache = {}

    def get(self, key: str):
        return self.cache.get(key)

    def set(self, key: str, value: str):
        self.cache[key] = value


# 使用
cache = CacheManager()
cache_key = get_cache_key("用户问题")

if cached := cache.get(cache_key):
    print("Cache hit:", cached)
else:
    # 执行API调用
    response = "API响应"
    cache.set(cache_key, response)
```

#### 交付成果

- [ ] JSON格式日志系统
- [ ] 基础监控指标统计
- [ ] 简单的缓存优化

---

## 学习资源清单

### 核心文档

1. OpenAI API文档：https://platform.openai.com/docs
2. LangChain文档：https://python.langchain.com/
3. LangGraph文档：https://langchain-ai.github.io/langgraph/
4. 智谱AI文档：https://open.bigmodel.cn/dev/api

### 在线教程

1. LangChain入门教程
2. FastAPI官方教程
3. Prompt Engineering Guide：https://www.promptingguide.ai/

### 书籍推荐

1. 《Building LLM Applications for Production》
2. 《Prompt Engineering for Developers》

### 开源项目

1. LangChain：https://github.com/langchain-ai/langchain
2. LangGraph：https://github.com/langchain-ai/langgraph
3. Dify（开源Agent平台）：https://github.com/langgenius/dify

---

## 学习建议

1. **高效利用2小时**
    - 理论学习：快速浏览核心概念，不要钻牛角尖
    - 实践操作：专注于代码实现，先跑通再优化

2. **善用Java经验**
    - 对比学习，理解Python与Java的差异
    - 发挥架构设计和工程实践经验

3. **循序渐进**
    - 每天完成当前任务，不要急于超前
    - 遇到问题及时查阅文档

4. **记录笔记**
    - 每天总结关键知识点
    - 记录遇到的问题和解决方案

5. **灵活调整**
    - 根据自己的学习速度调整进度
    - 重点内容可以多花时间，简单内容可以快速通过

---

## 常见问题 FAQ

### Q1：2小时时间不够完成实践任务怎么办？

A：优先完成核心功能，代码优化和测试可以后续补充。重点是理解原理，而不是写出完美代码。

### Q2：需要购买API额度吗？

A：建议准备$20-30的API额度，可以通过使用免费模型（如智谱GLM-3-turbo）降低成本。

### Q3：Python基础薄弱能跟上吗？

A：建议在学习Day 1前，花2-3天补习Python基础：基础语法、函数、类、模块导入。

### Q4：遇到技术难题如何解决？

A：优先查阅官方文档，然后在技术社区（GitHub、Stack Overflow）搜索相关问题。

### Q5：学完后能做什么？

A：能够独立开发简单的LLM应用和Agent系统，理解核心原理，具备进一步深入学习的能力。

---

## 结语

这份12天精简学习计划专为每天2小时的学习时间设计，聚焦LLM和Agent的核心知识点，通过理论与实践交替的方式，帮助你高效掌握关键技能。

关键在于：

- 专注核心概念，不要迷失在细节中
- 动手实践，代码跑通比完美更重要
- 持续学习，12天只是一个起点

祝学习顺利！
