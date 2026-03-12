from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Dict, Any

from apps.engineer.learn.agent.core.model import Model


class BaseAgent(ABC):
    def __init__(
            self,
            name: str,
            description: str = "",
            model: Optional[Model] = None,
            max_steps: int = 5
    ):
        self.name = name
        self.description = description
        self.message_history: List[Dict[str, str]] = []
        self.model = model
        self.max_steps = max_steps

    @abstractmethod
    def invoke(self, input: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def stream(self, input: str) -> str:
        raise NotImplementedError
