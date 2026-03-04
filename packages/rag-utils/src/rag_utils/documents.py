"""Document processing utilities for RAG applications."""

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import SPLITTER_CONFIG


def split_documents(
    docs: list[Document],
    config: dict | None = None,
) -> list[Document]:
    """Split documents into chunks using recursive character text splitter.

    Args:
        docs: List of documents to split.
        config: Optional configuration for the text splitter. If not provided,
                uses SPLITTER_CONFIG.

    Returns:
        A list of document chunks.

    Example:
        >>> docs = [Document(page_content="Long text...", metadata={"source": "doc.pdf"})]
        >>> chunks = split_documents(docs)
        >>> chunks = split_documents(docs, {"chunk_size": 500, "chunk_overlap": 50})
    """
    cfg = config or SPLITTER_CONFIG
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg["chunk_size"],
        chunk_overlap=cfg["chunk_overlap"],
        add_start_index=True,
    )
    return text_splitter.split_documents(docs)


def create_vector_store(
    embeddings: OpenAIEmbeddings,
    splits: list[Document] | None = None,
) -> InMemoryVectorStore:
    """Create an in-memory vector store with optional initial documents.

    Args:
        embeddings: An embeddings client for creating vector representations.
        splits: Optional list of document chunks to add to the store.

    Returns:
        An InMemoryVectorStore instance ready for similarity search.

    Example:
        >>> from rag_utils.embeddings import create_embeddings
        >>> embeddings = create_embeddings()
        >>> vector_store = create_vector_store(embeddings)
        >>> # Or with initial documents
        >>> vector_store = create_vector_store(embeddings, document_chunks)
    """
    vector_store = InMemoryVectorStore(embeddings)
    if splits:
        vector_store.add_documents(documents=splits)
    return vector_store
