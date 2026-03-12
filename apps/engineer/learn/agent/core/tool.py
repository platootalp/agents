from dataclasses import dataclass, field
from typing import Callable, Optional, Dict, Any


@dataclass
class Tool:
    name: str
    description: str
    func: Optional[Callable[[str], str]] = None
    parameters: Dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The input for the tool"}},
            "required": ["query"],
        }
    )
