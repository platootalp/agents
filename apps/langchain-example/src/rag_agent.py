"""RAG Agent with LangChain - Retrieval Augmented Generation example.

This script demonstrates:
1. Web document loading
2. Text splitting and indexing
3. RAG Agent with tool calling
4. RAG Chain for direct question-answering
"""

import os
from typing import List, Tuple

from bs4 import SoupStrainer
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.tools import tool

from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI

from rag_utils import (
    EMBEDDING_CONFIG,
    SPLITTER_CONFIG,
    create_embeddings,
    split_documents,
    create_vector_store,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

load_dotenv()

# OpenAI-compatible LLM configuration
LLM_CONFIG = {
    "model": os.environ.get("LLM_MODEL", "gpt-3.5-turbo"),
    "base_url": os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
    "api_key": os.environ.get("OPENAI_API_KEY", "not-needed"),
    "temperature": 0.7,
}

# Target URL for RAG (Lilian Weng's blog post on LLM agents)
TARGET_URL = "https://lilianweng.github.io/posts/2023-06-23-agent/"


# =============================================================================
# SECTION 1: INDEXING
# =============================================================================


def load_web_document(url: str) -> List[Document]:
    """Load web document using WebBaseLoader with BeautifulSoup parsing.

    Only extracts post title, headers, and content from the HTML.
    """
    bs4_strainer = SoupStrainer(class_=("post-title", "post-header", "post-content"))
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs={"parse_only": bs4_strainer},
    )
    docs = loader.load()
    print(f"Loaded document from {url}")
    print(f"Total characters: {len(docs[0].page_content)}")
    print(f"\nPreview:\n{docs[0].page_content[:500]}\n")
    return docs


# =============================================================================
# SECTION 2: RAG AGENT
# =============================================================================


def create_retrieve_tool(vector_store: InMemoryVectorStore):
    """Create a retrieval tool for the RAG agent.

    The tool retrieves relevant documents and returns both
    serialized content and the retrieved documents.
    """

    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str) -> Tuple[str, List[Document]]:
        """Retrieve information to help answer a query."""
        retrieved_docs = vector_store.similarity_search(query, k=2)
        serialized = "\n\n".join(
            f"Source: {doc.metadata}\nContent: {doc.page_content}" for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    return retrieve_context


def create_llm():
    """Initialize chat model with OpenAI-compatible API."""
    llm = ChatOpenAI(**LLM_CONFIG)
    print(f"LLM model: {LLM_CONFIG['model']}")
    return llm


def create_rag_agent(vector_store: InMemoryVectorStore):
    """Create a RAG agent with retrieval tool.

    Note: This uses a simplified agent pattern. For production use,
    consider using langgraph for more complex agent workflows.
    """
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    llm = create_llm()
    retrieve_tool = create_retrieve_tool(vector_store)
    tools = [retrieve_tool]

    # Create prompt with system message and agent scratchpad
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that can retrieve information from a blog post "
                "about LLM Powered Autonomous Agents. Use the retrieve_context tool to get "
                "relevant information to answer user queries. Always cite your sources.",
            ),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # Create agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    return agent_executor


def run_rag_agent(agent, query: str):
    """Run the RAG agent with a query."""
    print(f"\n{'=' * 60}")
    print(f"RAG AGENT QUERY: {query}")
    print(f"{'=' * 60}\n")

    result = agent.invoke({"input": query})
    print(f"\n{'=' * 60}")
    print("AGENT RESPONSE:")
    print(f"{'=' * 60}")
    print(result["output"])
    return result


# =============================================================================
# SECTION 3: RAG CHAIN (Alternative to Agent)
# =============================================================================


def create_rag_chain(vector_store: InMemoryVectorStore):
    """Create a simple RAG chain without agent overhead.

    This is faster and more efficient for simple Q&A tasks.
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser

    llm = create_llm()
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    # Create prompt template
    template = """Answer the question based only on the following context:

{context}

Question: {question}

Answer:"""

    prompt = ChatPromptTemplate.from_template(template)

    # Helper function to format retrieved documents
    def format_docs(docs: List[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    # Build RAG chain
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def run_rag_chain(chain, query: str):
    """Run the RAG chain with a query."""
    print(f"\n{'=' * 60}")
    print(f"RAG CHAIN QUERY: {query}")
    print(f"{'=' * 60}\n")

    result = chain.invoke(query)
    print(f"{'=' * 60}")
    print("CHAIN RESPONSE:")
    print(f"{'=' * 60}")
    print(result)
    return result


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main():
    """Execute the RAG pipeline."""
    print("=" * 60)
    print("RAG AGENT PIPELINE")
    print("=" * 60)

    # Step 1: Load web document
    print("\n[1] Loading web document...")
    docs = load_web_document(TARGET_URL)

    # Step 2: Split documents
    print("[2] Splitting documents...")
    splits = split_documents(docs)

    # Step 3: Create embeddings and vector store
    print("[3] Creating vector store...")
    embeddings = create_embeddings()
    vector_store = create_vector_store(embeddings, splits)

    # Step 4: Test RAG Chain (simpler approach)
    print("[4] Testing RAG Chain...")
    rag_chain = create_rag_chain(vector_store)

    queries = [
        "What is task decomposition?",
        "What are the main components of an autonomous agent?",
    ]

    for query in queries:
        run_rag_chain(rag_chain, query)
        print()

    # Step 5: Test RAG Agent (more flexible, tool-based approach)
    print("[5] Testing RAG Agent...")
    rag_agent = create_rag_agent(vector_store)

    for query in queries:
        run_rag_agent(rag_agent, query)
        print()

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
