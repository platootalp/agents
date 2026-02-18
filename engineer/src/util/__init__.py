from core import create_model

def get_qwen_model(**kwargs):
    """获取 Qwen 模型实例
    
    Args:
        **kwargs: 额外的模型配置参数
        
    Returns:
        QwenLLM: Qwen 模型实例
    """
    return create_model("qwen-turbo", **kwargs)

__all__ = [
    "get_qwen_model"
]
