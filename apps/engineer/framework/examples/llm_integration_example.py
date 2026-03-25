"""
LLM集成使用示例
展示如何使用LLM集成模块进行模型调用、检测和管理
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv()

from core import (
    ModelConfig, Message, ModelType,
    OpenAILLM, AnthropicLLM, QwenLLM, HuggingFaceLLM, OllamaLLM,
    create_model, infer_model_type, quick_chat, quick_chat_stream,
    ModelInfoProvider, ModelInfo, list_models, print_models
)


def example_1_quick_chat():
    """示例1: 快速聊天接口"""
    print("=" * 70)
    print("示例1: 快速聊天接口")
    print("=" * 70)
    
    try:
        response = quick_chat("你好，请介绍一下自己", model_name="qwen-plus")
        print(f"响应: {response.content}")
        print(f"延迟: {response.latency:.2f}秒")
        print(f"Token使用: {response.usage}")
    except Exception as e:
        print(f"错误: {e}")
    
    print()


def example_2_create_specific_model():
    """示例2: 创建特定类型的模型"""
    print("=" * 70)
    print("示例2: 创建特定类型的模型")
    print("=" * 70)
    
    try:
        print("创建OpenAI模型...")
        openai_model = create_model(
            model_name="gpt-3.5-turbo"
        )
        print(f"OpenAI模型创建成功: {openai_model.config.model_name}")
    except Exception as e:
        print(f"OpenAI模型创建失败: {e}")
    
    try:
        print("\n创建通义千问模型...")
        qwen_model = create_model(
            model_name="qwen-plus"
        )
        print(f"通义千问模型创建成功: {qwen_model.config.model_name}")
    except Exception as e:
        print(f"通义千问模型创建失败: {e}")
    
    print()


def example_3_chat_with_messages():
    """示例3: 使用消息列表进行多轮对话"""
    print("=" * 70)
    print("示例3: 使用消息列表进行多轮对话")
    print("=" * 70)
    
    try:
        model = create_model(model_name="qwen-plus")
        
        messages = [
            Message(role="system", content="你是一个有帮助的AI助手"),
            Message(role="user", content="什么是Python？"),
            Message(role="assistant", content="Python是一种高级编程语言"),
            Message(role="user", content="它有哪些特点？")
        ]
        
        response = model.chat(messages)
        print(f"响应: {response.content}")
        print(f"延迟: {response.latency:.2f}秒")
    except Exception as e:
        print(f"错误: {e}")
    
    print()


def example_4_stream_chat():
    """示例4: 流式聊天"""
    print("=" * 70)
    print("示例4: 流式聊天")
    print("=" * 70)
    
    try:
        print("流式响应: ", end="", flush=True)
        for chunk in quick_chat_stream("请用三句话介绍一下人工智能", model_name="qwen-plus"):
            print(chunk, end="", flush=True)
        print("\n")
    except Exception as e:
        print(f"错误: {e}")
    
    print()


def example_5_model_detection():
    """示例5: 查看可用模型"""
    print("=" * 70)
    print("示例5: 查看可用模型")
    print("=" * 70)
    
    print("列出所有可用模型...")
    all_models = list_models()
    
    total = 0
    for provider, models in all_models.items():
        if models:
            print(f"\n{provider.upper()} ({len(models)} 个模型):")
            for i, model in enumerate(models[:5], 1):
                print(f"  {i}. {model.model_name}")
            if len(models) > 5:
                print(f"  ... 还有 {len(models) - 5} 个模型")
            total += len(models)
    
    if total == 0:
        print("未检测到可用模型")
        print("\n提示：")
        print("- 配置API密钥：设置 OPENAI_API_KEY, ANTHROPIC_API_KEY, DASHSCOPE_API_KEY")
        print("- 启动Ollama：运行 'ollama serve'")
        print("- 下载Hugging Face模型：使用 transformers 库下载")
    else:
        print(f"\n总计: {total} 个可用模型")
    
    print()


def example_6_auto_create_model():
    """示例6: 显式创建指定模型"""
    print("=" * 70)
    print("示例6: 显式创建指定模型")
    print("=" * 70)
    
    print("创建指定的模型...")
    try:
        model = create_model(model_name="qwen-plus")
        print(f"创建的模型: {model.config.model_name}")
        print(f"模型类型: {model.config.model_type.value}")
        
        response = model.complete("你好")
        print(f"测试响应: {response.content[:50]}...")
    except Exception as e:
        print(f"错误: {e}")
    
    print()


def example_7_model_metrics():
    """示例7: 查看模型性能指标"""
    print("=" * 70)
    print("示例7: 查看模型性能指标")
    print("=" * 70)
    
    try:
        model = create_model(model_name="qwen-plus")
        
        print("进行多次调用以收集指标...")
        for i in range(3):
            model.complete(f"测试消息 {i+1}")
        
        metrics = model.metrics
        print(f"模型名称: {metrics.model_name}")
        print(f"平均延迟: {metrics.avg_latency:.2f}秒")
        print(f"平均速度: {metrics.avg_tokens_per_second:.2f} tokens/秒")
        print(f"成功率: {metrics.success_rate * 100:.1f}%")
        print(f"总调用次数: {metrics.total_calls}")
        print(f"错误次数: {metrics.error_count}")
    except Exception as e:
        print(f"错误: {e}")
    
    print()


def example_8_custom_config():
    """示例8: 使用自定义配置创建模型"""
    print("=" * 70)
    print("示例8: 使用自定义配置创建模型")
    print("=" * 70)
    
    try:
        model = create_model(
            model_name="qwen-plus",
            temperature=0.7,
            max_tokens=500,
            top_p=0.9,
            timeout=30
        )
        print(f"模型创建成功: {model.config.model_name}")
        print(f"温度: {model.config.temperature}")
        print(f"最大tokens: {model.config.max_tokens}")
        
        response = model.complete("你好")
        print(f"响应: {response.content[:50]}...")
    except Exception as e:
        print(f"错误: {e}")
    
    print()


def example_9_local_models():
    """示例9: 使用本地模型（Hugging Face和Ollama）"""
    print("=" * 70)
    print("示例9: 使用本地模型")
    print("=" * 70)
    
    try:
        print("尝试创建Hugging Face本地模型...")
        hf_model = create_model(
            model_name="gpt2"
        )
        print(f"Hugging Face模型创建成功: {hf_model.config.model_name}")
        
        response = hf_model.complete("Hello, how are you?")
        print(f"响应: {response.content[:100]}...")
    except Exception as e:
        print(f"Hugging Face模型创建失败: {e}")
    
    try:
        print("\n尝试创建Ollama本地模型...")
        ollama_model = create_model(
            model_name="llama2"
        )
        print(f"Ollama模型创建成功: {ollama_model.config.model_name}")
        
        response = ollama_model.complete("Hello")
        print(f"响应: {response.content[:100]}...")
    except Exception as e:
        print(f"Ollama模型创建失败: {e}")
    
    print()


def example_10_infer_model_type():
    """示例10: 模型类型推断"""
    print("=" * 70)
    print("示例10: 模型类型推断")
    print("=" * 70)
    
    test_models = [
        "gpt-4",
        "gpt-3.5-turbo",
        "claude-3-opus",
        "qwen-plus",
        "qwen-turbo",
        "llama2",
        "mistralai/Mistral-7B-Instruct-v0.2"
    ]
    
    for model_name in test_models:
        model_type = infer_model_type(model_name)
        print(f"{model_name:50} -> {model_type.value}")
    
    print()


def main():
    """运行所有示例"""
    print("\n")
    print("*" * 70)
    print("LLM集成功能使用示例")
    print("*" * 70)
    print("\n")
    
    examples = [
        example_1_quick_chat,
        example_3_chat_with_messages,
        example_4_stream_chat,
        example_5_model_detection,
        example_6_auto_create_model,
        example_7_model_metrics,
        example_8_custom_config,
        example_10_infer_model_type,
        # example_2_create_specific_model,
        # example_9_local_models,
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"示例执行失败: {e}")
            print()
    
    print("*" * 70)
    print("所有示例执行完毕")
    print("*" * 70)


if __name__ == "__main__":
    main()
