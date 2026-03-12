import os
from typing import List, Optional, Dict, Any

from openai import OpenAI


class Model:
    def __init__(self,
                 name: Optional[str] = None,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        """Initialize model with OpenAI API.

        Args:
            name: Model name (e.g., "gpt-4o", "gpt-3.5-turbo")
            api_key: OpenAI API key. If not provided, loads from OPENAI_API_KEY env var.
            base_url: Custom base URL. If not provided, loads from OPENAI_BASE_URL env var.
        """
        # Load from environment variables if not provided
        self.name = name or os.getenv("MODEL_NAME")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")

        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. "
                "Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )

        client_kwargs: Dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        self.client = OpenAI(**client_kwargs)

    def generate(
            self,
            messages: List[Dict[str, str]],
            temperature: float = 0.0,
            tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Any:
        """Generate a response from the model.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (0.0 = deterministic)
            tools: Optional list of tool definitions for function calling

        Returns:
            The completion response object
        """
        kwargs: Dict[str, Any] = {
            "model": self.name,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        return self.client.chat.completions.create(**kwargs)
