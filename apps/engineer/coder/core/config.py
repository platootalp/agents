import os
from typing import Optional

from pydantic import BaseModel


class ProviderConfig(BaseModel):
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    timeout: int = 60

    @classmethod
    def from_env(cls, provider: str = "openai", **overrides) -> "ProviderConfig":
        """从环境变量加载配置

        Args:
            provider: provider 名称，如 "openai", "anthropic"
            **overrides: 覆盖环境变量的参数

        Returns:
            ProviderConfig 实例

        Example:
            config = ProviderConfig.from_env("openai")
            config = ProviderConfig.from_env("openai", model="gpt-4o-mini")
        """
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass

        prefix = provider.upper()

        env_vars = {
            "api_key": os.getenv(f"{prefix}_API_KEY"),
            "base_url": os.getenv(f"{prefix}_BASE_URL"),
            "model": os.getenv(f"{prefix}_MODEL", "gpt-4o"),
            "temperature": os.getenv(f"{prefix}_TEMPERATURE"),
            "max_tokens": os.getenv(f"{prefix}_MAX_TOKENS"),
            "timeout": os.getenv(f"{prefix}_TIMEOUT", "60"),
        }

        if env_vars["temperature"] is not None:
            env_vars["temperature"] = float(env_vars["temperature"])
        else:
            env_vars["temperature"] = 0.7

        if env_vars["max_tokens"] is not None:
            env_vars["max_tokens"] = int(env_vars["max_tokens"])
        else:
            env_vars["max_tokens"] = None

        env_vars["timeout"] = int(env_vars["timeout"])

        config_dict = {k: v for k, v in env_vars.items() if v is not None or k == "max_tokens"}
        config_dict.update(overrides)

        return cls(**config_dict)
