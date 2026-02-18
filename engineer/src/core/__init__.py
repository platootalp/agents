"""
LLM 核心模块
提供统一的语言模型调用接口，支持多供应商模型集成和本地模型部署
对齐 LangChain 的命名规范和架构设计
"""

# genAI_main_start
# ==================== 消息类（统一模块）====================

from .messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    FunctionMessage,
    ToolMessage,
    Message,
    messages_to_dict,
    messages_from_dict,
    get_buffer_string,
)

# ==================== 语言模型基础类 ====================

from .language_models import (
    # 基础类
    BaseChatModel,
    
    # 配置和结果类
    ChatModelConfig,
    ChatResult,
    LLMResult,
    ModelMetrics,
    ModelType,
)
# genAI_main_end

# ==================== 聊天模型工厂 ====================

from .chat_model_factory import (
    # 主要工厂函数（对齐 LangChain）
    init_chat_model,
    
    # 类型特定工厂函数（对齐 LangChain）
    ChatOpenAI,
    ChatAnthropic,
    ChatOllama,
    
    # 便捷函数
    quick_invoke,
    quick_batch,
    
    # 辅助函数
    infer_model_type,
    get_chat_model_class,
)

# ==================== 提供商实现 ====================

from .providers import (
    OpenAILLM,
    AnthropicLLM,
    QwenLLM,
    HuggingFaceLLM,
    OllamaLLM
)

# ==================== 模型信息 ====================

from .model_info import (
    ModelInfoProvider,
    ModelInfo,
    list_models,
    print_models
)


# ==================== 导出列表 ====================

# genAI_main_start
__all__ = [
    # 消息类
    "BaseMessage",
    "HumanMessage",
    "AIMessage",
    "SystemMessage",
    "FunctionMessage",
    "ToolMessage",
    "Message",
    "messages_to_dict",
    "messages_from_dict",
    "get_buffer_string",
    
    # 基础类
    "BaseChatModel",
    
    # 配置和结果类
    "ChatModelConfig",
    "ChatResult",
    "LLMResult",
    "ModelMetrics",
    "ModelType",
    
    # 工厂函数（对齐 LangChain）
    "init_chat_model",
    "ChatOpenAI",
    "ChatAnthropic",
    "ChatOllama",
    
    # 便捷函数
    "quick_invoke",
    "quick_batch",
    "infer_model_type",
    "get_chat_model_class",
    
    # 提供商类
    "OpenAILLM",
    "AnthropicLLM",
    "QwenLLM",
    "HuggingFaceLLM",
    "OllamaLLM",
    
    # 模型信息
    "ModelInfoProvider",
    "ModelInfo",
    "list_models",
    "print_models",
]

__version__ = "3.0.0"  # 移除向后兼容，升级到 3.0.0
# genAI_main_end
