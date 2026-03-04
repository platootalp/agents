"""Semantic Search with LangChain - A complete RAG pipeline example.

This script demonstrates:
1. Document loading (PDF)
2. Text splitting
3. Embeddings generation
4. Vector store indexing
5. Similarity search and retrieval
"""

import os
from typing import List

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.runnables import chain
from langchain_core.vectorstores import InMemoryVectorStore

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# =============================================================================
# CONFIGURATION
# =============================================================================

load_dotenv()

# OpenAI-compatible embedding service configuration
# Supports: vLLM, Ollama, LM Studio, Xinference, etc.
EMBEDDING_CONFIG = {
    "model": os.environ.get("EMBEDDING_MODEL", "text-embedding-v1"),
    "base_url": os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
    "api_key": os.environ.get("OPENAI_API_KEY", "not-needed"),
}

# PDF file path
PDF_PATH = "./data/nke-10k-2023.pdf"

# Text splitting configuration
SPLITTER_CONFIG = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "add_start_index": True,
}


# =============================================================================
# SECTION 1: DOCUMENT LOADING
# =============================================================================

def load_pdf(file_path: str):
    """Load PDF document into LangChain Document objects."""
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages from PDF")
    print(f"\nFirst page preview:\n{docs[0].page_content[:200]}\n")
    print(f"Metadata: {docs[0].metadata}\n")
    return docs


def create_sample_documents() -> List[Document]:
    """Create sample documents for testing."""
    return [
        Document(
            page_content="Dogs are great companions, known for their loyalty and friendliness.",
            metadata={"source": "mammal-pets-doc"},
        ),
        Document(
            page_content="Cats are independent pets that often enjoy their own space.",
            metadata={"source": "mammal-pets-doc"},
        ),
    ]


# =============================================================================
# SECTION 2: TEXT SPLITTING
# =============================================================================

def split_documents(docs: List[Document]) -> List[Document]:
    """Split documents into smaller chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(**SPLITTER_CONFIG)
    all_splits = text_splitter.split_documents(docs)
    print(f"Split into {len(all_splits)} chunks\n")
    return all_splits


# =============================================================================
# SECTION 3: EMBEDDINGS
# =============================================================================

def create_embeddings():
    """Initialize embedding model with OpenAI-compatible API."""
    embeddings = OpenAIEmbeddings(**EMBEDDING_CONFIG)
    print(f"Embeddings model: {EMBEDDING_CONFIG['model']}")
    print(f"Base URL: {EMBEDDING_CONFIG['base_url']}\n")
    return embeddings


def test_embeddings(embeddings, splits: List[Document]):
    """Test embedding generation."""
    vector_1 = embeddings.embed_query(splits[0].page_content)
    vector_2 = embeddings.embed_query(splits[1].page_content)

    assert len(vector_1) == len(vector_2)
    print(f"Generated vectors of length {len(vector_1)}")
    print(f"Sample vector values: {vector_1[:10]}\n")


# =============================================================================
# SECTION 4: VECTOR STORE
# =============================================================================

def create_vector_store(embeddings, splits: List[Document]) -> InMemoryVectorStore:
    """Create vector store and index documents."""
    vector_store = InMemoryVectorStore(embeddings)
    ids = vector_store.add_documents(documents=splits)
    print(f"Indexed {len(ids)} documents into vector store\n")
    return vector_store


def search_vector_store(vector_store: InMemoryVectorStore, query: str):
    """Perform similarity search."""
    results = vector_store.similarity_search(query)
    print(f"Query: {query}")
    print(f"Result: {results[0].page_content[:300]}...\n")
    return results


def search_with_scores(vector_store: InMemoryVectorStore, query: str):
    """Perform similarity search with scores."""
    results = vector_store.similarity_search_with_score(query)
    doc, score = results[0]
    print(f"Query: {query}")
    print(f"Score: {score}")
    print(f"Result: {doc.page_content[:300]}...\n")
    return results


# =============================================================================
# SECTION 5: RETRIEVERS
# =============================================================================

def create_custom_retriever(vector_store: InMemoryVectorStore):
    """Create a custom retriever using @chain decorator."""
    @chain
    def retriever(query: str) -> List[Document]:
        return vector_store.similarity_search(query, k=1)

    return retriever


def create_vector_retriever(vector_store: InMemoryVectorStore):
    """Create a VectorStoreRetriever."""
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 1},
    )


def test_retrievers(vector_store: InMemoryVectorStore):
    """Test both retriever types."""
    queries = [
        "How many distribution centers does Nike have in the US?",
        "When was Nike incorporated?",
    ]

    # Test custom retriever
    custom_retriever = create_custom_retriever(vector_store)
    print("=== Custom Retriever Results ===")
    custom_results = custom_retriever.batch(queries)
    for i, (query, result) in enumerate(zip(queries, custom_results)):
        print(f"\nQuery {i+1}: {query}")
        print(f"Result: {result[0].page_content[:200]}...")

    # Test VectorStoreRetriever
    vector_retriever = create_vector_retriever(vector_store)
    print("\n\n=== VectorStoreRetriever Results ===")
    vector_results = vector_retriever.batch(queries)
    for i, (query, result) in enumerate(zip(queries, vector_results)):
        print(f"\nQuery {i+1}: {query}")
        print(f"Result: {result[0].page_content[:200]}...")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Execute the semantic search pipeline."""
    print("=" * 60)
    print("SEMANTIC SEARCH PIPELINE")
    print("=" * 60)

    # Step 1: Load documents
    print("\n[1] Loading PDF document...")
    docs = load_pdf(PDF_PATH)

    # Step 2: Split documents
    print("[2] Splitting documents...")
    splits = split_documents(docs)

    # Step 3: Initialize embeddings
    print("[3] Initializing embeddings...")
    embeddings = create_embeddings()
    test_embeddings(embeddings, splits)

    # Step 4: Create vector store
    print("[4] Creating vector store...")
    vector_store = create_vector_store(embeddings, splits)

    # Step 5: Test searches
    print("[5] Testing similarity search...")
    search_vector_store(
        vector_store,
        "How many distribution centers does Nike have in the US?"
    )
    search_with_scores(
        vector_store,
        "What was Nike's revenue in 2023?"
    )

    # Step 6: Test retrievers
    print("[6] Testing retrievers...")
    test_retrievers(vector_store)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
