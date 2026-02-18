"""
LLM信息查询和验证
提供LLM信息查询功能，不自动选择LLM
"""

import os
import requests
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# genAI_main_start
from .language_models import ChatModelConfig, ModelType
# genAI_main_end


@dataclass
class ModelInfo:
    """LLM信息"""
    model_name: str
    model_type: ModelType
    provider: str
    available: bool
    description: str = ""


class ModelInfoProvider:
    """LLM信息提供者"""

    @staticmethod
    def get_openai_models(api_key: Optional[str] = None) -> List[ModelInfo]:
        """
        获取OpenAI可用LLM列表
        :param api_key: API密钥，不指定则从环境变量读取
        :return: LLM信息列表
        """
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            return []

        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

        try:
            response = requests.get(
                f"{base_url}/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=5
            )
            if response.status_code == 200:
                models = response.json().get("data", [])
                return [
                        ModelInfo(
                            model_name=m["id"],
                            model_type=ModelType.OPENAI,
                            provider="OpenAI",
                            available=True,
                            description=f"OpenAI LLM: {m['id']}"
                        )
                        for m in models if m["id"].startswith("gpt")
                    ]
        except Exception:
            pass

        return []

    @staticmethod
    def get_anthropic_models(api_key: Optional[str] = None) -> List[ModelInfo]:
        """
        获取Anthropic可用LLM列表
        :param api_key: API密钥，不指定则从环境变量读取
        :return: LLM信息列表
        """
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return []

        try:
            from anthropic import Anthropic
            client = Anthropic(api_key=api_key)
            models = client.models.list()

            return [
                ModelInfo(
                    model_name=m.id,
                    model_type=ModelType.ANTHROPIC,
                    provider="Anthropic",
                    available=True,
                    description=f"Anthropic LLM: {m.id}"
                )
                for m in models.data
            ]
        except Exception:
            pass

        return []

    @staticmethod
    def get_qwen_models(api_key: Optional[str] = None) -> List[ModelInfo]:
        """
        获取通义千问可用LLM列表
        :param api_key: API密钥，不指定则从环境变量读取
        :return: LLM信息列表
        """
        api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            return []

        base_url = os.getenv("DASHSCOPE_API_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            response = requests.get(
                f"{base_url}/models",
                headers=headers,
                timeout=5
            )
            if response.status_code == 200:
                models = response.json().get("data", [])

                text_models = []
                for m in models:
                    model_id = m["id"].lower()
                    if "qwen" in model_id and "image" not in model_id and "vision" not in model_id and "vl" not in model_id and "tts" not in model_id and "asr" not in model_id and "mt" not in model_id and "character" not in model_id and "omni" not in model_id and "livetranslate" not in model_id and "s2s" not in model_id:
                        text_models.append(m["id"])

                return [
                    ModelInfo(
                        model_name=model_name,
                        model_type=ModelType.QWEN,
                        provider="通义千问",
                        available=True,
                        description=f"通义千问LLM: {model_name}"
                    )
                    for model_name in text_models
                ]
        except Exception:
            pass

        return []

    @staticmethod
    def get_ollama_models(base_url: Optional[str] = None) -> List[ModelInfo]:
        """
        获取Ollama本地LLM列表
        :param base_url: Ollama服务地址，不指定则从环境变量读取
        :return: LLM信息列表
        """
        base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        try:
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                result = response.json()
                models = result.get("models", [])

                return [
                    ModelInfo(
                        model_name=m["name"],
                        model_type=ModelType.OLLAMA,
                        provider="Ollama",
                        available=True,
                        description=f"Ollama本地LLM: {m['name']}"
                    )
                    for m in models
                ]
        except Exception:
            pass

        return []

    @staticmethod
    def get_huggingface_models() -> List[ModelInfo]:
        """
        获取Hugging Face本地LLM列表
        :return: LLM信息列表
        """
        try:
            from transformers import AutoTokenizer
            import torch
        except ImportError:
            return []

        device = "cuda" if torch.cuda.is_available() else "cpu"
        common_models = [
            "Qwen/Qwen2.5-7B-Instruct",
            "THUDM/chatglm3-6b",
            "microsoft/DialoGPT-medium",
            "gpt2",
            "facebook/opt-1.3b"
        ]

        available_models = []
        for model_name in common_models:
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    local_files_only=True
                )
                available_models.append(
                    ModelInfo(
                        model_name=model_name,
                        model_type=ModelType.HUGGINGFACE,
                        provider="Hugging Face",
                        available=True,
                        description=f"Hugging Face本地LLM，设备: {device}"
                    )
                )
            except Exception:
                continue

        return available_models

    @staticmethod
    def list_all_models() -> Dict[str, List[ModelInfo]]:
        """
        列出所有可用的LLM
        :return: 按提供商分组的LLM信息字典
        """
        return {
            "openai": ModelInfoProvider.get_openai_models(),
            "anthropic": ModelInfoProvider.get_anthropic_models(),
            "qwen": ModelInfoProvider.get_qwen_models(),
            "ollama": ModelInfoProvider.get_ollama_models(),
            "huggingface": ModelInfoProvider.get_huggingface_models()
        }

    @staticmethod
    def print_available_models():
        """打印所有可用LLM信息"""
        print("\n" + "=" * 70)
        print("可用的LLM")
        print("=" * 70)

        all_models = ModelInfoProvider.list_all_models()

        total = 0
        for provider, models in all_models.items():
            if models:
                print(f"\n{provider.upper()} ({len(models)} 个LLM):")
                for i, model in enumerate(models[:5], 1):
                    print(f"  {i}. {model.model_name}")
                if len(models) > 5:
                    print(f"  ... 还有 {len(models) - 5} 个LLM")
                total += len(models)

        if total == 0:
            print("\n未检测到可用的LLM")
            print("\n提示：")
            print("- 配置API密钥：设置 OPENAI_API_KEY, ANTHROPIC_API_KEY, DASHSCOPE_API_KEY")
            print("- 启动Ollama：运行 'ollama serve'")
            print("- 下载Hugging Face LLM：使用 transformers 库下载")
        else:
            print(f"\n总计: {total} 个可用LLM")

        print("=" * 70 + "\n")


def list_models() -> Dict[str, List[ModelInfo]]:
    """
    列出所有可用的LLM
    :return: 按提供商分组的LLM信息字典
    """
    return ModelInfoProvider.list_all_models()


def print_models():
    """打印所有可用LLM信息"""
    ModelInfoProvider.print_available_models()
