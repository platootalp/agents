"""Persistent memory system for conversation history."""

import json
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from .provider import Message


class ConversationSession(BaseModel):
    """A saved conversation session."""

    session_id: str = Field(..., description="Unique session identifier")
    title: str = Field("", description="Session title/summary")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    messages: list[Message] = Field(default_factory=list)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class MemoryManager:
    """Manages persistent conversation memory."""

    def __init__(self, storage_dir: str | None = None):
        if storage_dir:
            self.storage_dir = Path(storage_dir)
        else:
            # Default: ~/.cli-agent/history
            home = Path.home()
            self.storage_dir = home / ".cli-agent" / "history"

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._current_session: ConversationSession | None = None

    def _get_session_path(self, session_id: str) -> Path:
        """Get file path for a session."""
        return self.storage_dir / f"{session_id}.json"

    def create_session(self, title: str = "") -> ConversationSession:
        """Create a new conversation session."""
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        session = ConversationSession(
            session_id=session_id,
            title=title or f"Session {session_id}",
        )
        self._current_session = session
        self.save_session(session)
        return session

    def save_session(self, session: ConversationSession) -> None:
        """Save a session to disk."""
        session.updated_at = datetime.now()
        path = self._get_session_path(session.session_id)

        # Convert to JSON-serializable format
        data = {
            "session_id": session.session_id,
            "title": session.title,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "tool_calls": m.tool_calls,
                    "tool_call_id": m.tool_call_id,
                    "name": m.name,
                }
                for m in session.messages
            ],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_session(self, session_id: str) -> ConversationSession | None:
        """Load a session from disk."""
        path = self._get_session_path(session_id)
        if not path.exists():
            return None

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        session = ConversationSession(
            session_id=data["session_id"],
            title=data["title"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            messages=[
                Message(
                    role=m["role"],
                    content=m.get("content"),
                    tool_calls=m.get("tool_calls"),
                    tool_call_id=m.get("tool_call_id"),
                    name=m.get("name"),
                )
                for m in data["messages"]
            ],
        )
        self._current_session = session
        return session

    def list_sessions(self) -> list[ConversationSession]:
        """List all saved sessions."""
        sessions = []
        for path in self.storage_dir.glob("*.json"):
            try:
                session_id = path.stem
                session = self.load_session(session_id)
                if session:
                    sessions.append(session)
            except Exception:
                continue

        # Sort by updated_at descending
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        path = self._get_session_path(session_id)
        if path.exists():
            path.unlink()
            if self._current_session and self._current_session.session_id == session_id:
                self._current_session = None
            return True
        return False

    def get_current_session(self) -> ConversationSession | None:
        """Get the current active session."""
        return self._current_session

    def set_current_session(self, session: ConversationSession) -> None:
        """Set the current session."""
        self._current_session = session

    def add_message(self, message: Message) -> None:
        """Add a message to the current session and save."""
        if self._current_session:
            self._current_session.messages.append(message)
            self.save_session(self._current_session)

    def clear_current_session(self) -> None:
        """Clear the current session without deleting from disk."""
        self._current_session = None

    def get_messages(self) -> list[Message]:
        """Get all messages from current session."""
        if self._current_session:
            return self._current_session.messages.copy()
        return []
