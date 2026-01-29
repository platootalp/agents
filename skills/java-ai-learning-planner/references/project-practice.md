# 项目实践指南

## 目录
- [项目阶段划分](#项目阶段划分)
- [Hello World项目](#hello-world项目)
- [RAG问答系统](#rag问答系统)
- [简单Agent应用](#简单agent应用)
- [多Agent系统](#多agent系统)
- [生产级应用](#生产级应用)
- [最佳实践](#最佳实践)
- [常见问题](#常见问题)
- [学习资源](#学习资源)

## 项目阶段划分

### 渐进式学习路径
```
阶段1（第1-2周）：Hello World
    ↓
阶段2（第3-6周）：RAG问答系统
    ↓
阶段3（第7-12周）：简单Agent应用
    ↓
阶段4（第13-20周）：多Agent系统
    ↓
阶段5（第21-28周）：生产级应用
```

### 各阶段目标
| 阶段 | 项目 | 技能点 | 交付物 |
|------|------|--------|--------|
| 1 | Hello World | Python基础、API调用 | 聊天机器人 |
| 2 | RAG问答 | 向量数据库、文档处理 | 知识库问答 |
| 3 | 简单Agent | Agent框架、工具调用 | 个人助理 |
| 4 | 多Agent | 协作机制、编排 | 团队协作系统 |
| 5 | 生产级 | 部署、监控、优化 | 完整应用 |

## Hello World项目

### 项目目标
创建一个简单的聊天机器人，使用大模型API进行对话

### 技术栈
- Python 3.9+
- OpenAI API 或 智谱AI API
- FastAPI（可选，用于Web界面）

### 项目结构
```
hello-chatbot/
├── main.py          # 主程序
├── config.py        # 配置文件
├── requirements.txt # 依赖
└── README.md        # 说明文档
```

### 实现代码

#### config.py
```python
import os

# API配置
API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME = "gpt-3.5-turbo"

# 对话配置
MAX_HISTORY = 10  # 保留最近10轮对话
TEMPERATURE = 0.7
```

#### main.py
```python
from openai import OpenAI
from config import API_KEY, MODEL_NAME, MAX_HISTORY, TEMPERATURE

class SimpleChatbot:
    """简单聊天机器人"""

    def __init__(self):
        self.client = OpenAI(api_key=API_KEY)
        self.history = []

    def add_message(self, role: str, content: str):
        """添加消息到历史"""
        self.history.append({"role": role, "content": content})

        # 限制历史长度
        if len(self.history) > MAX_HISTORY * 2:
            self.history = self.history[-MAX_HISTORY * 2:]

    def chat(self, user_message: str) -> str:
        """聊天"""
        # 添加用户消息
        self.add_message("user", user_message)

        # 调用API
        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=self.history,
            temperature=TEMPERATURE
        )

        # 获取回复
        assistant_message = response.choices[0].message.content

        # 添加到历史
        self.add_message("assistant", assistant_message)

        return assistant_message

    def clear_history(self):
        """清空历史"""
        self.history = []

# 命令行交互
def main():
    bot = SimpleChatbot()
    print("你好！我是AI助手，输入'exit'退出")

    while True:
        user_input = input("\n你: ")

        if user_input.lower() == "exit":
            print("再见！")
            break

        if user_input.lower() == "clear":
            bot.clear_history()
            print("对话历史已清空")
            continue

        response = bot.chat(user_input)
        print(f"助手: {response}")

if __name__ == "__main__":
    main()
```

### 功能扩展

#### 1. 添加角色设定
```python
class RoleChatbot(SimpleChatbot):
    """带角色的聊天机器人"""

    def __init__(self, system_prompt: str):
        super().__init__()
        self.system_prompt = system_prompt

    def chat(self, user_message: str) -> str:
        # 在历史消息前插入系统提示
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_message})

        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE
        )

        return response.choices[0].message.content

# 使用示例
bot = RoleChatbot("你是一个经验丰富的Java架构师，擅长微服务设计")
```

#### 2. 添加流式输出
```python
def stream_chat(self, user_message: str):
    """流式聊天"""
    self.add_message("user", user_message)

    stream = self.client.chat.completions.create(
        model=MODEL_NAME,
        messages=self.history,
        stream=True
    )

    full_response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            full_response += content

    print()
    self.add_message("assistant", full_response)
    return full_response
```

### 学习要点
- 理解大模型API的基本调用方式
- 掌握对话历史的管理
- 学习参数调节（temperature、max_tokens）
- 理解流式输出的优势

## RAG问答系统

### 项目目标
构建一个基于文档的问答系统，能够根据文档内容回答用户问题

### 技术栈
- LangChain
- OpenAI Embeddings
- FAISS/Chroma（向量数据库）
- FastAPI（Web API）

### 项目结构
```
rag-qa-system/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI应用
│   ├── rag.py           # RAG核心逻辑
│   ├── documents.py     # 文档处理
│   └── api.py           # API路由
├── data/
│   └── documents/       # 待处理的文档
├── requirements.txt
└── README.md
```

### 核心实现

#### 1. 文档处理 (documents.py)
```python
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_documents(directory: str):
    """加载文档"""
    loader = DirectoryLoader(
        directory,
        glob="**/*.txt",
        loader_cls=TextLoader
    )
    return loader.load()

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """分割文档"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return splitter.split_documents(documents)
```

#### 2. RAG核心 (rag.py)
```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import os

class RAGSystem:
    """RAG问答系统"""

    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-4")
        self.vectorstore = None

    def build_index(self, documents):
        """构建向量索引"""
        print(f"正在构建索引，文档数量: {len(documents)}")
        self.vectorstore = FAISS.from_documents(
            documents,
            self.embeddings
        )
        print("索引构建完成")

    def save_index(self, path: str):
        """保存索引"""
        if self.vectorstore:
            self.vectorstore.save_local(path)

    def load_index(self, path: str):
        """加载索引"""
        self.vectorstore = FAISS.load_local(
            path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

    def query(self, question: str, top_k: int = 3) -> dict:
        """查询"""
        if not self.vectorstore:
            raise ValueError("向量索引未初始化")

        # 创建QA链
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": top_k}
            ),
            return_source_documents=True
        )

        # 执行查询
        result = qa_chain.invoke({"query": question})

        return {
            "answer": result["result"],
            "sources": [
                {
                    "content": doc.page_content[:200],
                    "metadata": doc.metadata
                }
                for doc in result["source_documents"]
            ]
        }
```

#### 3. FastAPI接口 (api.py)
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="RAG问答系统")

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

class QueryResponse(BaseModel):
    answer: str
    sources: list

# 全局RAG实例
rag_system = RAGSystem()

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """查询接口"""
    try:
        result = rag_system.query(request.question, request.top_k)
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 使用示例
```python
from documents import load_documents, split_documents
from rag import RAGSystem

# 1. 加载文档
documents = load_documents("./data/documents")
print(f"加载了 {len(documents)} 个文档")

# 2. 分割文档
chunks = split_documents(documents)
print(f"分割为 {len(chunks)} 个文档块")

# 3. 构建索引
rag = RAGSystem()
rag.build_index(chunks)

# 4. 保存索引（可选）
rag.save_index("./data/index")

# 5. 查询
result = rag.query("Spring Boot中如何配置数据库？")
print(f"回答: {result['answer']}")
print(f"来源: {len(result['sources'])} 个文档片段")
```

### 功能优化

#### 1. 添加缓存
```python
import hashlib
from functools import lru_cache

class CachedRAGSystem(RAGSystem):
    """带缓存的RAG系统"""

    @lru_cache(maxsize=100)
    def _get_cache_key(self, question: str) -> str:
        return hashlib.md5(question.encode()).hexdigest()

    def query(self, question: str, top_k: int = 3) -> dict:
        cache_key = self._get_cache_key(question)
        # 检查缓存...
        return super().query(question, top_k)
```

#### 2. 多轮对话
```python
from langchain.chains import ConversationalRetrievalChain

class ConversationalRAG(RAGSystem):
    """支持多轮对话的RAG"""

    def create_chat_chain(self):
        """创建对话链"""
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(),
            return_source_documents=True
        )

    def chat(self, question: str, chat_history: list = []):
        """多轮对话"""
        chain = self.create_chat_chain()
        result = chain({
            "question": question,
            "chat_history": chat_history
        })
        return result
```

### 学习要点
- 理解向量数据库的基本概念
- 掌握文档分块策略
- 学习RAG的工作原理
- 实践API设计和缓存优化

## 简单Agent应用

### 项目目标
构建一个带工具调用的Agent，能够搜索信息、查询数据

### 技术栈
- LangChain Agents
- OpenAI Functions
- 自定义工具
- Streamlit（简单UI）

### 项目结构
```
personal-assistant/
├── app/
│   ├── __init__.py
│   ├── agent.py         # Agent核心
│   ├── tools/           # 工具定义
│   │   ├── __init__.py
│   │   ├── search.py
│   │   ├── calculator.py
│   │   └── weather.py
│   └── ui.py            # Streamlit界面
├── requirements.txt
└── README.md
```

### 核心实现

#### 1. 工具定义 (tools/calculator.py)
```python
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
import operator

class CalculatorInput(BaseModel):
    """计算器输入"""
    expression: str = Field(description="数学表达式，如 '2 + 3 * 4'")

def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        # 安全的计算（仅支持基础运算）
        result = eval(expression, {"__builtins__": {}}, {
            "add": operator.add,
            "sub": operator.sub,
            "mul": operator.mul,
            "div": operator.truediv
        })
        return str(result)
    except Exception as e:
        return f"计算错误: {str(e)}"

calculator_tool = StructuredTool.from_function(
    func=calculate,
    name="calculator",
    description="执行数学计算，支持加减乘除",
    args_schema=CalculatorInput
)
```

#### 2. 搜索工具 (tools/search.py)
```python
import requests
from langchain.tools import Tool
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    """搜索输入"""
    query: str = Field(description="搜索关键词")

def web_search(query: str) -> str:
    """网络搜索（示例使用免费API）"""
    # 实际项目中可使用Google Search API、Bing API等
    # 这里返回模拟结果
    return f"关于'{query}'的搜索结果：\n1. 官方文档\n2. 教程\n3. 社区讨论"

search_tool = StructuredTool.from_function(
    func=web_search,
    name="search",
    description="搜索网络信息",
    args_schema=SearchInput
)
```

#### 3. Agent核心 (agent.py)
```python
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from tools.calculator import calculator_tool
from tools.search import search_tool

class PersonalAssistant:
    """个人助理Agent"""

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4")
        self.tools = [calculator_tool, search_tool]
        self.agent = None
        self.executor = None

    def create_agent(self):
        """创建Agent"""
        # 定义Prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个有帮助的个人助理。
            你可以使用以下工具：
            - calculator: 数学计算
            - search: 网络搜索

            请根据用户需求选择合适的工具。"""),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        # 创建Agent
        self.agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )

        # 创建执行器
        self.executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )

    def chat(self, message: str) -> str:
        """聊天"""
        if not self.executor:
            self.create_agent()

        result = self.executor.invoke({"input": message})
        return result["output"]
```

#### 4. Streamlit UI (ui.py)
```python
import streamlit as st
from agent import PersonalAssistant

st.title("🤖 AI个人助理")

# 初始化Agent
if "agent" not in st.session_state:
    st.session_state.agent = PersonalAssistant()

# 对话历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 用户输入
if user_input := st.chat_input("请输入您的需求..."):
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Agent响应
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            response = st.session_state.agent.chat(user_input)
            st.markdown(response)

    # 保存响应
    st.session_state.messages.append({"role": "assistant", "content": response})
```

### 运行方式
```bash
# 安装依赖
pip install streamlit langchain openai

# 运行Streamlit
streamlit run app/ui.py
```

### 扩展功能

#### 1. 添加记忆
```python
from langchain.memory import ConversationBufferMemory

class MemoryAgent(PersonalAssistant):
    """带记忆的Agent"""

    def __init__(self):
        super().__init__()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

    def create_agent(self):
        # 在prompt中添加历史
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个有帮助的助理，记得对话历史"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        self.agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )

        self.executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,  # 添加记忆
            verbose=True
        )
```

#### 2. 添加更多工具
```python
# 天气工具
def get_weather(city: str) -> str:
    """获取天气信息"""
    # 调用天气API
    return f"{city}今天天气晴，温度25°C"

weather_tool = Tool(
    name="weather",
    func=get_weather,
    description="查询城市天气"
)

# 添加到tools列表
self.tools.append(weather_tool)
```

### 学习要点
- 理解Agent的工作原理
- 掌握工具定义和使用
- 学习Agent的Prompt设计
- 实践记忆机制和状态管理

## 多Agent系统

### 项目目标
构建一个代码审查系统，包含多个Agent协作完成不同任务

### 技术栈
- LangGraph（Multi-Agent框架）
- OpenAI GPT-4
- 多Agent协作模式

### 项目架构
```
代码审查系统
├── Analyzer Agent     # 分析代码结构和逻辑
├── Security Agent     # 检查安全问题
├── Style Agent        # 检查代码风格
├── Optimizer Agent    # 提供优化建议
└── Reporter Agent     # 生成审查报告
```

### 项目结构
```
code-review-system/
├── app/
│   ├── __init__.py
│   ├── agents.py          # Agent定义
│   ├── workflow.py        # 工作流编排
│   └── main.py            # 主程序
├── tests/
│   └── test_review.py
├── requirements.txt
└── README.md
```

### 核心实现

#### 1. Agent定义 (agents.py)
```python
from langchain_openai import ChatOpenAI
from typing import Dict

class BaseAgent:
    """Agent基类"""

    def __init__(self, name: str, role: str, system_prompt: str):
        self.name = name
        self.role = role
        self.llm = ChatOpenAI(model="gpt-4")
        self.system_prompt = system_prompt

    def run(self, input_data: str) -> str:
        """运行Agent"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": input_data}
        ]
        response = self.llm.invoke(messages)
        return response.content

class AnalyzerAgent(BaseAgent):
    """代码分析Agent"""

    def __init__(self):
        super().__init__(
            name="analyzer",
            role="代码分析",
            system_prompt="""你是一个代码分析专家。
            任务：分析代码的结构、逻辑、复杂度。
            输出：JSON格式，包含：
            - structure: 代码结构分析
            - complexity: 复杂度评估
            - issues: 发现的问题列表"""
        )

class SecurityAgent(BaseAgent):
    """安全检查Agent"""

    def __init__(self):
        super().__init__(
            name="security",
            role="安全检查",
            system_prompt="""你是一个安全专家。
            任务：检查代码中的安全漏洞、SQL注入、XSS等。
            输出：JSON格式，包含：
            - vulnerabilities: 漏洞列表
            - severity: 严重程度
            - recommendations: 修复建议"""
        )

class StyleAgent(BaseAgent):
    """代码风格Agent"""

    def __init__(self):
        super().__init__(
            name="style",
            role="风格检查",
            system_prompt="""你是一个代码风格专家。
            任务：检查代码是否符合规范、命名是否合理、注释是否充分。
            输出：JSON格式，包含：
            - style_issues: 风格问题
            - naming: 命名问题
            - comments: 注释建议"""
        )

class ReporterAgent(BaseAgent):
    """报告生成Agent"""

    def __init__(self):
        super().__init__(
            name="reporter",
            role="报告生成",
            system_prompt="""你是一个报告生成专家。
            任务：汇总各Agent的分析结果，生成完整的审查报告。
            输出：Markdown格式的审查报告"""
        )
```

#### 2. 工作流编排 (workflow.py)
```python
from typing import TypedDict, List, Annotated
from operator import add
from langgraph.graph import StateGraph, END
from agents import AnalyzerAgent, SecurityAgent, StyleAgent, ReporterAgent

class ReviewState(TypedDict):
    """审查状态"""
    code: str
    analysis: str = ""
    security: str = ""
    style: str = ""
    final_report: str = ""

class CodeReviewWorkflow:
    """代码审查工作流"""

    def __init__(self):
        # 初始化Agent
        self.analyzer = AnalyzerAgent()
        self.security = SecurityAgent()
        self.style = StyleAgent()
        self.reporter = ReporterAgent()

        # 构建工作流
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """构建工作流图"""
        workflow = StateGraph(ReviewState)

        # 添加节点
        workflow.add_node("analyze", self._analyze_node)
        workflow.add_node("security_check", self._security_node)
        workflow.add_node("style_check", self._style_node)
        workflow.add_node("report", self._report_node)

        # 定义边
        workflow.set_entry_point("analyze")
        workflow.add_edge("analyze", "security_check")
        workflow.add_edge("security_check", "style_check")
        workflow.add_edge("style_check", "report")
        workflow.add_edge("report", END)

        return workflow.compile()

    def _analyze_node(self, state: ReviewState):
        """分析节点"""
        result = self.analyzer.run(state["code"])
        return {"analysis": result}

    def _security_node(self, state: ReviewState):
        """安全检查节点"""
        result = self.security.run(state["code"])
        return {"security": result}

    def _style_node(self, state: ReviewState):
        """风格检查节点"""
        result = self.style.run(state["code"])
        return {"style": result}

    def _report_node(self, state: ReviewState):
        """报告节点"""
        # 汇总所有结果
        summary = f"""
        代码分析：
        {state['analysis']}

        安全检查：
        {state['security']}

        风格检查：
        {state['style']}
        """

        result = self.reporter.run(summary)
        return {"final_report": result}

    def execute(self, code: str) -> str:
        """执行审查"""
        initial_state = {"code": code}
        final_state = self.workflow.invoke(initial_state)
        return final_state["final_report"]
```

#### 3. 主程序 (main.py)
```python
from workflow import CodeReviewWorkflow

def review_code(code: str) -> str:
    """审查代码"""
    workflow = CodeReviewWorkflow()
    report = workflow.execute(code)
    return report

# 示例使用
if __name__ == "__main__":
    sample_code = """
    def process_user_input(user_input: str):
        import sqlite3
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        # SQL注入风险
        query = f"SELECT * FROM users WHERE name = '{user_input}'"
        cursor.execute(query)
        return cursor.fetchall()
    """

    report = review_code(sample_code)
    print(report)
```

### 高级功能

#### 1. 并行执行
```python
from langgraph.graph import StateGraph

# 并行节点
workflow.add_node("parallel_security", self._security_node)
workflow.add_node("parallel_style", self._style_node)

# 并行边
workflow.add_conditional_edges(
    "analyze",
    lambda x: x["analysis"],
    {
        "continue": ["parallel_security", "parallel_style"]
    }
)

# 汇聚节点
workflow.add_edge("parallel_security", "report")
workflow.add_edge("parallel_style", "report")
```

#### 2. 条件分支
```python
def _conditional_router(state: ReviewState):
    """条件路由"""
    # 如果发现严重安全问题，直接报告
    if "critical" in state["security"].lower():
        return "urgent_report"

    # 否则继续完整流程
    return "full_review"

workflow.add_conditional_edges(
    "security_check",
    _conditional_router,
    {
        "urgent_report": "report",
        "full_review": "style_check"
    }
)
```

### 学习要点
- 理解Multi-Agent协作模式
- 掌握LangGraph工作流设计
- 学习状态管理和数据流转
- 实践并行执行和条件分支

## 生产级应用

### 项目目标
构建一个生产级的企业知识库问答系统

### 技术栈
- FastAPI（API服务）
- PostgreSQL（元数据存储）
- Redis（缓存）
- Docker（容器化）
- Prometheus + Grafana（监控）

### 项目架构
```
┌─────────────┐
│   负载均衡   │
└──────┬──────┘
       ↓
┌─────────────────────────────────┐
│         API Gateway             │
└──────┬──────────────────────────┘
       ↓
┌─────────────────────────────────┐
│      FastAPI应用 (多实例)        │
│  ┌──────────┐  ┌──────────┐    │
│  │  RAG服务  │  │缓存服务  │    │
│  └──────────┘  └──────────┘    │
└──────┬──────────────────────────┘
       ↓
┌─────────────────────────────────┐
│       PostgreSQL               │
│       Redis                    │
│       向量数据库               │
└─────────────────────────────────┘
```

### 项目结构
```
enterprise-qa/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── core.py
│   │   ├── retriever.py
│   │   └── generator.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py
│   │   └── schemas.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── database.py
│   └── utils/
│       ├── __init__.py
│       ├── cache.py
│       └── logger.py
├── tests/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

### 核心实现

#### 1. 配置管理 (config.py)
```python
import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """配置管理"""

    # API配置
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # OpenAI配置
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4"

    # 数据库配置
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "enterprise_qa"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = ""

    # Redis配置
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0

    # 向量数据库配置
    VECTOR_DB_PATH: str = "./data/vector_db"

    # 缓存配置
    CACHE_TTL: int = 3600

    class Config:
        env_file = ".env"

settings = Settings()
```

#### 2. 数据库模型 (models/database.py)
```python
from sqlalchemy import Column, Integer, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Document(Base):
    """文档表"""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255))
    content = Column(Text)
    category = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class QueryLog(Base):
    """查询日志表"""
    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, index=True)
    question = Column(Text)
    answer = Column(Text)
    tokens_used = Column(Integer)
    latency_ms = Column(Integer)
    cached = Column(String(10))  # "true"/"false"
    created_at = Column(DateTime, default=datetime.utcnow)
```

#### 3. RAG核心 (rag/core.py)
```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from app.utils.cache import CacheManager
from app.utils.logger import Logger
import time

class ProductionRAG:
    """生产级RAG"""

    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model=settings.OPENAI_MODEL)
        self.vectorstore = None
        self.cache = CacheManager()
        self.logger = Logger(__name__)

    def load_vectorstore(self):
        """加载向量数据库"""
        try:
            self.vectorstore = FAISS.load_local(
                settings.VECTOR_DB_PATH,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            self.logger.info("向量数据库加载成功")
        except Exception as e:
            self.logger.error(f"向量数据库加载失败: {str(e)}")
            raise

    def query(self, question: str) -> dict:
        """查询"""
        start_time = time.time()

        # 1. 检查缓存
        cached = self.cache.get(f"qa:{question}")
        if cached:
            self.logger.info(f"缓存命中: {question[:50]}...")
            return {
                "answer": cached,
                "cached": True,
                "tokens_used": 0,
                "latency_ms": int((time.time() - start_time) * 1000)
            }

        # 2. 执行RAG
        try:
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
                return_source_documents=True
            )

            result = qa_chain.invoke({"query": question})

            # 3. 缓存结果
            answer = result["result"]
            self.cache.set(f"qa:{question}", answer, settings.CACHE_TTL)

            latency_ms = int((time.time() - start_time) * 1000)

            return {
                "answer": answer,
                "cached": False,
                "sources": [doc.metadata for doc in result["source_documents"]],
                "tokens_used": 0,  # 需要从response中获取
                "latency_ms": latency_ms
            }

        except Exception as e:
            self.logger.error(f"查询失败: {str(e)}")
            raise
```

#### 4. API路由 (api/routes.py)
```python
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import Optional

from app.rag.core import ProductionRAG
from app.models.database import QueryLog, get_db
from app.api.schemas import QueryRequest, QueryResponse

router = APIRouter()

# 全局RAG实例
rag = ProductionRAG()
rag.load_vectorstore()

@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    db: Session = Depends(get_db)
):
    """查询接口"""
    try:
        # 执行查询
        result = rag.query(request.question)

        # 记录日志
        log = QueryLog(
            question=request.question,
            answer=result["answer"],
            tokens_used=result["tokens_used"],
            latency_ms=result["latency_ms"],
            cached="true" if result["cached"] else "false"
        )
        db.add(log)
        db.commit()

        return QueryResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "healthy",
        "vectorstore": "loaded" if rag.vectorstore else "not loaded"
    }
```

#### 5. Docker配置 (Dockerfile)
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### 6. Docker Compose (docker-compose.yml)
```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: enterprise_qa
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      POSTGRES_HOST: postgres
      REDIS_HOST: redis
    depends_on:
      - postgres
      - redis
    volumes:
      - ./data/vector_db:/app/data/vector_db

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    depends_on:
      - prometheus

volumes:
  postgres_data:
```

### 部署流程
```bash
# 1. 构建镜像
docker-compose build

# 2. 启动服务
docker-compose up -d

# 3. 初始化数据库
docker-compose exec api python -m app.init_db

# 4. 构建向量索引
docker-compose exec api python -m app.build_index

# 5. 测试
curl http://localhost:8000/health
```

### 监控和日志

#### 1. Prometheus监控指标
```python
from prometheus_client import Counter, Histogram

# 指标定义
QUERY_COUNT = Counter('qa_query_count', 'Total queries')
QUERY_LATENCY = Histogram('qa_query_latency_seconds', 'Query latency')
CACHE_HIT_RATE = Counter('qa_cache_hits', 'Cache hits')
CACHE_MISS_RATE = Counter('qa_cache_misses', 'Cache misses')

# 在query函数中使用
@QUERY_LATENCY.time()
def query_with_metrics(question: str):
    QUERY_COUNT.inc()

    result = rag.query(question)

    if result["cached"]:
        CACHE_HIT_RATE.inc()
    else:
        CACHE_MISS_RATE.inc()

    return result
```

#### 2. 日志配置 (utils/logger.py)
```python
import logging
from datetime import datetime
import json

class JSONFormatter(logging.Formatter):
    """JSON格式日志"""

    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage()
        }
        return json.dumps(log_data)

def setup_logger(name: str):
    """设置日志"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)

    return logger
```

## 最佳实践

### 1. 代码组织
- 模块化设计，职责清晰
- 使用Type Hints提高可读性
- 单元测试覆盖率>80%

### 2. 性能优化
- 合理使用缓存
- 批量处理请求
- 异步IO操作

### 3. 错误处理
- 完善的异常捕获
- 友好的错误提示
- 详细的日志记录

### 4. 安全性
- 敏感信息脱敏
- API限流
- 输入验证

## 常见问题

### Q1: 如何选择向量数据库？
根据数据量和需求选择：
- 小规模（<1万文档）：FAISS（本地）
- 中规模（1-10万）：Chroma、Qdrant
- 大规模（>10万）：Pinecone、Milvus

### Q2: 如何降低API成本？
- 使用缓存减少重复调用
- 选择合适的模型（便宜模型+少量高精度模型）
- 批量处理请求

### Q3: 如何提高响应速度？
- 流式输出
- 预计算和缓存
- 使用更快的模型

## 学习资源

### 官方文档
- LangChain文档：https://python.langchain.com/
- FastAPI文档：https://fastapi.tiangolo.com/
- Docker文档：https://docs.docker.com/

### 开源项目
- LangChain仓库：https://github.com/langchain-ai/langchain
- LlamaIndex：https://github.com/run-llama/llama_index
- Dify：https://github.com/langgenius/dify

### 学习平台
- Hugging Face：模型和datasets
- Kaggle：数据科学竞赛
- Coursera：在线课程
