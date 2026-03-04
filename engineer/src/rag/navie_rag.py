import os
import bs4
from typing import TypedDict, List, Annotated
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_qdrant import Qdrant
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ..util import get_qwen_model, get_embedding_model

# 加载环境变量
load_dotenv()

# 设置 USER_AGENT
if not os.getenv("USER_AGENT"):
    os.environ["USER_AGENT"] = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )


# ============ 1. 定义状态 ============
class RagState(TypedDict):
    """RAG 工作流状态"""

    question: str  # 用户问题
    retrieved_docs: List[str]  # 检索到的文档片段
    answer: str  # LLM 生成的答案


# ============ 2. 构建向量知识库 ============
def build_vectorstore(doc_path: str = "./knowledge.txt") -> VectorStore:
    """加载web文档"""
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()

    # 文本分割
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    # 创建向量库
    embeddings = get_embedding_model()
    vectorstore = Qdrant.from_documents(
        splits,
        embeddings,
        location=":memory:",
        collection_name="langgraph_knowledge",
    )
    return vectorstore


# 初始化向量库（全局单例）
vectorstore = build_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# ============ 3. 定义节点函数 ============
def retrieve_node(state: RagState) -> RagState:
    """检索节点：根据问题检索相关文档"""
    docs = retriever.invoke(state["question"])
    retrieved_texts = [doc.page_content for doc in docs]
    print(f"🔍 检索到 {len(retrieved_texts)} 个相关片段")
    for i, text in enumerate(retrieved_texts, 1):
        print(f"  [{i}] {text.strip()}")
    return {
        "question": state["question"],
        "retrieved_docs": retrieved_texts,
        "answer": "",  # 重置答案
    }


def generate_node(state: RagState) -> RagState:
    """生成节点：基于问题和检索结果生成答案"""
    # 构建 Prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一个专业助手，请基于以下检索到的上下文回答问题。"
                "\n上下文：{context}"
                "\n如果上下文无法回答问题，请说明你不知道。",
            ),
            ("human", "{question}"),
        ]
    )

    # 构建 RAG Chain
    llm = get_qwen_model()
    rag_chain = (
        {
            "context": lambda x: "\n\n".join(x["retrieved_docs"]),
            "question": lambda x: x["question"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # 生成答案
    answer = rag_chain.invoke(state)
    print(f"✅ 生成答案：{answer[:100]}...")
    return {
        "question": state["question"],
        "retrieved_docs": state["retrieved_docs"],
        "answer": answer,
    }


# ============ 4. 构建 LangGraph 工作流 ============
def create_rag_graph() -> StateGraph:
    """创建朴素 RAG 工作流图"""
    workflow = StateGraph(RagState)

    # 添加节点
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    # 定义执行流：START → retrieve → generate → END
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


# ============ 5. 使用示例 ============
if __name__ == "__main__":
    # 编译图
    rag_app = create_rag_graph()

    # 示例问题
    questions = [
        "LangGraph 是什么？",
        "LangGraph 支持哪些特性？",
        "如何用 LangGraph 构建 Agent？",
    ]

    for q in questions:
        print(f"\n{'=' * 50}")
        print(f"❓ 问题: {q}")
        print(f"{'=' * 50}")

        # 执行 RAG 流程
        result = rag_app.invoke({"question": q, "retrieved_docs": [], "answer": ""})
