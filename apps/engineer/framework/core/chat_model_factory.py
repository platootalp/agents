"""
聊天模型工厂函数
对齐 LangChain 的模型创建模式，提供便捷的模型实例化接口
"""

# genAI_main_start
import os
from typing import Optional, Type, Dict, Any
from .language_models import (
    BaseChatModel,
    ChatModelConfig,
    ModelType,
    AIMessage
)


# ==================== 模型提供商映射 ====================

def get_chat_model_class(model_type: ModelType) -> Type[BaseChatModel]:
    """获取聊天模型类
    
    根据模型类型返回对应的模型类
    延迟导入以避免循环依赖
    
    Args:
        model_type: 模型类型枚举
    
    Returns:
        Type[BaseChatModel]: 模型类
    
    Raises:
        ValueError: 不支持的模型类型
    """
    # 延迟导入避免循环依赖
    from .providers import (
        OpenAILLM,
        AnthropicLLM,
        QwenLLM,
        HuggingFaceLLM,
        OllamaLLM
    )
    
    provider_map: Dict[ModelType, Type[BaseChatModel]] = {
        ModelType.OPENAI: OpenAILLM,
        ModelType.ANTHROPIC: AnthropicLLM,
        ModelType.QWEN: QwenLLM,
        ModelType.HUGGINGFACE: HuggingFaceLLM,
        ModelType.OLLAMA: OllamaLLM,
    }
    
    model_class = provider_map.get(model_type)
    if not model_class:
        raise ValueError(
            f"不支持的模型类型: {model_type}. "
            f"支持的类型: {list(provider_map.keys())}"
        )
    
    return model_class


# ==================== 模型类型推断 ====================

def infer_model_type(model_name: str) -> ModelType:
    """根据模型名称推断模型类型
    
    基于模型名称的模式匹配自动推断模型类型
    
    推断规则：
    - gpt-* -> OpenAI
    - claude-* -> Anthropic
    - qwen-* -> Qwen
    - */* (包含斜杠) -> HuggingFace
    - 其他 -> Ollama
    
    Args:
        model_name: 模型名称
    
    Returns:
        ModelType: 推断出的模型类型
    
    Examples:
        >>> infer_model_type("gpt-4")
        ModelType.OPENAI
        >>> infer_model_type("claude-3-opus")
        ModelType.ANTHROPIC
        >>> infer_model_type("meta-llama/Llama-2-7b")
        ModelType.HUGGINGFACE
    """
    model_name_lower = model_name.lower()
    
    if model_name_lower.startswith("gpt"):
        return ModelType.OPENAI
    elif model_name_lower.startswith("claude"):
        return ModelType.ANTHROPIC
    elif "qwen" in model_name_lower:
        return ModelType.QWEN
    elif "/" in model_name_lower:
        # HuggingFace 模型通常包含 org/model 格式
        return ModelType.HUGGINGFACE
    else:
        # 默认为 Ollama（本地模型）
        return ModelType.OLLAMA


# ==================== 主要工厂函数 ====================

def init_chat_model(
    model: str,
    *,
    model_provider: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    **kwargs: Any
) -> BaseChatModel:
    """初始化聊天模型 (对齐 LangChain 的 init_chat_model 函数)
    
    提供统一的模型初始化接口，支持自动推断模型类型
    
    Args:
        model: 模型名称（必需）
        model_provider: 模型提供商（可选，未指定时自动推断）
        api_key: API 密钥（可选，未指定时从环境变量读取）
        base_url: API 基础 URL（可选）
        temperature: 温度参数
        max_tokens: 最大 token 数
        **kwargs: 其他配置参数
    
    Returns:
        BaseChatModel: 聊天模型实例
    
    Raises:
        ValueError: 模型名称为空或不支持的提供商
    
    Examples:
        >>> # 自动推断类型
        >>> model = init_chat_model("gpt-4")
        
        >>> # 指定提供商
        >>> model = init_chat_model(
        ...     "gpt-4",
        ...     model_provider="openai",
        ...     api_key="sk-..."
        ... )
        
        >>> # 使用配置参数
        >>> model = init_chat_model(
        ...     "claude-3-opus",
        ...     temperature=0.5,
        ...     max_tokens=4000
        ... )
    """
    if not model:
        raise ValueError("模型名称不能为空")
    
    # 确定模型类型
    if model_provider:
        try:
            model_type = ModelType(model_provider.lower())
        except ValueError:
            raise ValueError(
                f"不支持的模型提供商: {model_provider}. "
                f"支持的提供商: {[t.value for t in ModelType]}"
            )
    else:
        model_type = infer_model_type(model)
    
    # 根据模型类型设置默认配置
    config_params = {
        "model_name": model,
        "model_type": model_type,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    
    # 处理 API 密钥和 URL
    if model_type == ModelType.OPENAI:
        config_params["api_key"] = api_key or os.getenv("OPENAI_API_KEY")
        config_params["base_url"] = base_url or os.getenv("OPENAI_BASE_URL")
    
    elif model_type == ModelType.ANTHROPIC:
        config_params["api_key"] = api_key or os.getenv("ANTHROPIC_API_KEY")
        config_params["base_url"] = base_url or os.getenv("ANTHROPIC_BASE_URL")
    
    elif model_type == ModelType.QWEN:
        config_params["api_key"] = api_key or os.getenv("DASHSCOPE_API_KEY")
        config_params["base_url"] = base_url or os.getenv(
            "DASHSCOPE_API_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    
    elif model_type == ModelType.HUGGINGFACE:
        config_params["api_key"] = api_key or os.getenv("HUGGINGFACE_API_KEY")
        config_params["base_url"] = base_url or os.getenv("HUGGINGFACE_API_URL")
    
    elif model_type == ModelType.OLLAMA:
        config_params["base_url"] = base_url or os.getenv(
            "OLLAMA_BASE_URL",
            "http://localhost:11434"
        )
    
    # 合并额外参数
    config_params.update(kwargs)
    
    # 创建配置对象
    config = ChatModelConfig(**config_params)
    
    # 获取模型类并实例化
    model_class = get_chat_model_class(model_type)
    return model_class(config)




# ==================== 便捷函数 ====================

def quick_invoke(
    text: str,
    model: str = "gpt-3.5-turbo",
    **kwargs: Any
) -> str:
    """快速调用模型（对齐 LangChain 风格）
    
    适合快速测试和简单场景
    
    Args:
        text: 输入文本
        model: 模型名称
        **kwargs: 其他参数
    
    Returns:
        str: 模型输出文本
    
    Examples:
        >>> response = quick_invoke("你好", model="gpt-4")
        >>> print(response)
    """
    chat_model = init_chat_model(model, **kwargs)
    ai_message = chat_model.invoke(text)
    return ai_message.content


def quick_batch(
    texts: list[str],
    model: str = "gpt-3.5-turbo",
    **kwargs: Any
) -> list[str]:
    """快速批量调用（对齐 LangChain 风格）
    
    Args:
        texts: 输入文本列表
        model: 模型名称
        **kwargs: 其他参数
    
    Returns:
        list[str]: 模型输出文本列表
    
    Examples:
        >>> responses = quick_batch(["你好", "再见"], model="gpt-4")
        >>> for resp in responses:
        ...     print(resp)
    """
    chat_model = init_chat_model(model, **kwargs)
    ai_messages = chat_model.batch(texts)
    return [msg.content for msg in ai_messages]


# ==================== 类型特定的工厂函数 ====================

def ChatOpenAI(
    model_name: str = "gpt-3.5-turbo",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs: Any
) -> BaseChatModel:
    """创建 OpenAI 聊天模型 (对齐 LangChain 的 ChatOpenAI)
    
    Args:
        model_name: 模型名称
        api_key: API 密钥
        base_url: API 基础 URL
        **kwargs: 其他参数
    
    Returns:
        BaseChatModel: OpenAI 聊天模型实例
    
    Examples:
        >>> model = ChatOpenAI(model_name="gpt-4")
        >>> response = model.invoke("你好")
    """
    return init_chat_model(
        model=model_name,
        model_provider="openai",
        api_key=api_key,
        base_url=base_url,
        **kwargs
    )


def ChatAnthropic(
    model_name: str = "claude-3-opus-20240229",
    api_key: Optional[str] = None,
    **kwargs: Any
) -> BaseChatModel:
    """创建 Anthropic 聊天模型 (对齐 LangChain 的 ChatAnthropic)
    
    Args:
        model_name: 模型名称
        api_key: API 密钥
        **kwargs: 其他参数
    
    Returns:
        BaseChatModel: Anthropic 聊天模型实例
    """
    return init_chat_model(
        model=model_name,
        model_provider="anthropic",
        api_key=api_key,
        **kwargs
    )


def ChatOllama(
    model_name: str = "llama2",
    base_url: Optional[str] = None,
    **kwargs: Any
) -> BaseChatModel:
    """创建 Ollama 聊天模型 (对齐 LangChain 的 ChatOllama)
    
    Args:
        model_name: 模型名称
        base_url: Ollama 服务地址
        **kwargs: 其他参数
    
    Returns:
        BaseChatModel: Ollama 聊天模型实例
    """
    return init_chat_model(
        model=model_name,
        model_provider="ollama",
        base_url=base_url,
        **kwargs
    )


# genAI_main_end
