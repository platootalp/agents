"""Embedding utilities for RAG applications."""

from langchain_openai import OpenAIEmbeddings

from .config import EMBEDDING_CONFIG


def create_embeddings(config: dict | None = None) -> OpenAIEmbeddings:
    """Create an embeddings client with the given configuration.

    Args:
        config: Optional configuration dictionary. If not provided, uses EMBEDDING_CONFIG.

    Returns:
        An OpenAIEmbeddings instance configured for OpenAI-compatible APIs.

    Example:
        >>> embeddings = create_embeddings()
        >>> embeddings = create_embeddings({
        ...     "model": "text-embedding-v2",
        ...     "base_url": "http://localhost:8000/v1",
        ...     "api_key": "sk-xxx"
        ... })
    """
    cfg = config or EMBEDDING_CONFIG
    return OpenAIEmbeddings(
        model=cfg["model"],
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
    )
