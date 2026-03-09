"""Shared configuration utilities for AI applications."""

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelConfig:
    """Configuration for an LLM model.

    Attributes:
        model_name: Name of the model (e.g., "gpt-4", "claude-3-sonnet")
        provider: Provider name (e.g., "openai", "anthropic")
        api_key: API key for the provider (loaded from env if not provided)
        base_url: Optional base URL for the API
        temperature: Sampling temperature (0-2)
        max_tokens: Maximum tokens to generate
        timeout: Request timeout in seconds
        extra_params: Additional provider-specific parameters
    """

    model_name: str
    provider: str
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.7
    max_tokens: int | None = None
    timeout: float = 60.0
    extra_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Load API key from environment if not provided."""
        if self.api_key is None:
            env_var = f"{self.provider.upper()}_API_KEY"
            self.api_key = os.getenv(env_var)

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for LLM clients."""
        result = {
            "model": self.model_name,
            "temperature": self.temperature,
            "timeout": self.timeout,
        }
        if self.max_tokens:
            result["max_tokens"] = self.max_tokens
        result.update(self.extra_params)
        return result


def load_config_from_env(prefix: str = "LLM") -> dict[str, Any]:
    """Load configuration from environment variables.

    Args:
        prefix: Prefix for environment variables (e.g., "OPENAI", "ANTHROPIC")

    Returns:
        Dictionary of configuration values
    """
    config = {}

    # Load common parameters
    if model := os.getenv(f"{prefix}_MODEL"):
        config["model_name"] = model
    if temp := os.getenv(f"{prefix}_TEMPERATURE"):
        config["temperature"] = float(temp)
    if max_tokens := os.getenv(f"{prefix}_MAX_TOKENS"):
        config["max_tokens"] = int(max_tokens)
    if timeout := os.getenv(f"{prefix}_TIMEOUT"):
        config["timeout"] = float(timeout)
    if base_url := os.getenv(f"{prefix}_BASE_URL"):
        config["base_url"] = base_url

    return config
