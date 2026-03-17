from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4
from pydantic import BaseModel, Field

from .message import Message


class Session(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    messages: List[Message] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def add_message(self, message: Message) -> None:
        self.messages.append(message)
        self.updated_at = datetime.now()

    def get_messages(self, limit: Optional[int] = None) -> List[Message]:
        if limit:
            return self.messages[-limit:]
        return self.messages.copy()

    def to_openai_list(self) -> List[Dict[str, Any]]:
        return [m.to_openai_dict() for m in self.messages]

    def clear(self) -> None:
        self.messages.clear()
        self.updated_at = datetime.now()

    def estimate_tokens(self) -> int:
        total = sum(len(m.content) for m in self.messages)
        return int(total / 4)


class SessionManager:
    def __init__(self):
        self._sessions: Dict[str, Session] = {}

    def create(self, session_id: Optional[str] = None) -> Session:
        sid = session_id or str(uuid4())
        session = Session(session_id=sid)
        self._sessions[sid] = session
        return session

    def get(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def get_or_create(self, session_id: Optional[str] = None) -> Session:
        if session_id:
            session = self.get(session_id)
            if session:
                return session
        return self.create(session_id)

    def delete(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def add_message(self, session_id: str, message: Message) -> Optional[Session]:
        session = self.get_or_create(session_id)
        session.add_message(message)
        return session

    def list_sessions(self) -> List[str]:
        return list(self._sessions.keys())
