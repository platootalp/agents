"""Shared configuration for RAG applications."""

import os

from dotenv import load_dotenv

load_dotenv()


# OpenAI-compatible LLM configuration
# Supports: OpenAI, vLLM, Ollama, LM Studio, Xinference, etc.
LLM_CONFIG = {
    "model": os.environ.get("LLM_MODEL", "gpt-3.5-turbo"),
    "base_url": os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
    "api_key": os.environ.get("OPENAI_API_KEY", "not-needed"),
    "temperature": float(os.environ.get("LLM_TEMPERATURE", "0.7")),
}

# OpenAI-compatible embedding service configuration
EMBEDDING_CONFIG = {
    "model": os.environ.get("EMBEDDING_MODEL", "text-embedding-v1"),
    "base_url": os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
    "api_key": os.environ.get("OPENAI_API_KEY", "not-needed"),
}

# Text splitting configuration
SPLITTER_CONFIG = {
    "chunk_size": int(os.environ.get("CHUNK_SIZE", "1000")),
    "chunk_overlap": int(os.environ.get("CHUNK_OVERLAP", "200")),
    "add_start_index": True,
}
