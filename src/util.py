import os
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings

# Load environment variables from .env file
load_dotenv()


def get_tavily_client() -> TavilyClient:
    """
    获取 Tavily 客户端实例
    
    Returns:
        TavilyClient: 配置好的 Tavily 客户端
    """
    # Check if the TAVILY_API_KEY is loaded correctly
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        raise ValueError("TAVILY_API_KEY is not set in environment variables or .env file.")

    return TavilyClient(api_key=tavily_api_key)


def get_qwen_model() -> ChatOpenAI:
    """
    获取 Qwen 模型实例
    
    Returns:
        ChatOpenAI: 配置好的 Qwen 模型
    """
    # Check if the DASHSCOPE_API_KEY is loaded correctly
    dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
    if not dashscope_api_key:
        raise ValueError("DASHSCOPE_API_KEY is not set in environment variables or .env file.")

    # Get model configuration from environment variables
    model_name = os.getenv("DASHSCOPE_API_MODEL", "qwen3-max-preview")
    base_url = os.getenv("DASHSCOPE_API_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

    # Initialize Qwen model using DASHSCOPE_API_KEY
    return ChatOpenAI(
        model=model_name,
        api_key=dashscope_api_key,
        base_url=base_url
    )


def get_embedding_model() -> DashScopeEmbeddings:
    # Check if the DASHSCOPE_API_KEY is loaded correctly
    dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
    if not dashscope_api_key:
        raise ValueError("DASHSCOPE_API_KEY is not set in environment variables or .env file.")

    embedding_name = os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v3")
    return DashScopeEmbeddings(model=embedding_name, dashscope_api_key=dashscope_api_key)
