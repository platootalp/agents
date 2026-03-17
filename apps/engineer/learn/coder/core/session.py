"""Session management module for coder agents.

Provides session persistence, message history management, and conversation state tracking.
"""

import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Message(BaseModel):
    """A single message in a conversation."""

    model_config = ConfigDict(frozen=True)

    role: str = Field(..., description="Message role: 'user', 'assistant', 'system', or 'tool'")
    content: str = Field(..., description="Message content")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional message metadata"
    )
    timestamp: datetime = Field(default_factory=datetime.now, description="Message creation time")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Tool calls if any"
    )
    tool_call_id: Optional[str] = Field(
        default=None, description="ID of tool call this message responds to"
    )

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        allowed = {"user", "assistant", "system", "tool"}
        if v not in allowed:
            raise ValueError(f"role must be one of {allowed}, got '{v}'")
        return v


class Session(BaseModel):
    """A conversation session with message history and metadata.

    Attributes:
        session_id: Unique identifier for the session
        workspace: Workspace or project context for the session
        user_id: Optional user identifier
        messages: List of conversation messages
        created_at: Session creation timestamp
        updated_at: Last update timestamp
        metadata: Arbitrary session metadata
        title: Optional session title for display
        tags: List of tags for categorization
    """

    model_config = ConfigDict(validate_assignment=True)

    session_id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique session identifier"
    )
    workspace: str = Field(default="default", description="Workspace context")
    user_id: Optional[str] = Field(default=None, description="Associated user ID")
    messages: List[Message] = Field(default_factory=list, description="Conversation messages")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Session metadata")
    title: Optional[str] = Field(default=None, description="Session title")
    tags: List[str] = Field(default_factory=list, description="Categorization tags")

    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        tool_call_id: Optional[str] = None,
    ) -> "Message":
        """Add a message to the session.

        Args:
            role: Message role (user/assistant/system/tool)
            content: Message content
            metadata: Optional message metadata
            tool_calls: Optional tool calls
            tool_call_id: Optional tool call ID

        Returns:
            The created Message instance
        """
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {},
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
        )
        self.messages.append(message)
        self.updated_at = datetime.now()
        return message

    def get_messages(self, limit: Optional[int] = None, offset: int = 0) -> List["Message"]:
        """Get messages with optional pagination.

        Args:
            limit: Maximum number of messages to return
            offset: Number of messages to skip from the end

        Returns:
            List of messages (most recent first if offset/limit used)
        """
        msgs = self.messages
        if offset:
            msgs = msgs[:-offset] if offset < len(msgs) else []
        if limit:
            msgs = msgs[-limit:] if limit < len(msgs) else msgs
        return msgs

    def get_last_message(self) -> Optional["Message"]:
        """Get the most recent message."""
        return self.messages[-1] if self.messages else None

    def clear_messages(self) -> None:
        """Clear all messages from the session."""
        self.messages.clear()
        self.updated_at = datetime.now()

    def to_dict_list(self) -> List[Dict[str, str]]:
        """Convert messages to simple dict list format for LLM APIs."""
        return [{"role": msg.role, "content": msg.content} for msg in self.messages]

    def get_message_count(self) -> int:
        """Get total number of messages."""
        return len(self.messages)

    def get_token_estimate(self) -> int:
        """Estimate token count (rough approximation)."""
        return int(sum(len(msg.content.split()) for msg in self.messages) * 1.3)

    def get_duration(self) -> float:
        """Get session duration in seconds."""
        return (self.updated_at - self.created_at).total_seconds()

    def update_title(self, title: str) -> None:
        """Update session title."""
        self.title = title
        self.updated_at = datetime.now()

    def add_tag(self, tag: str) -> None:
        """Add a tag to the session."""
        if tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = datetime.now()

    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the session."""
        if tag in self.tags:
            self.tags.remove(tag)
            self.updated_at = datetime.now()


class SessionStore(ABC):
    """Abstract base class for session persistence."""

    @abstractmethod
    def save(self, session: Session) -> None:
        """Save a session."""
        raise NotImplementedError

    @abstractmethod
    def load(self, session_id: str) -> Optional[Session]:
        """Load a session by ID. Returns None if not found."""
        raise NotImplementedError

    @abstractmethod
    def delete(self, session_id: str) -> bool:
        """Delete a session. Returns True if deleted."""
        raise NotImplementedError

    @abstractmethod
    def list_sessions(
        self,
        workspace: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Session]:
        """List sessions with optional filtering."""
        raise NotImplementedError


class InMemorySessionStore(SessionStore):
    """In-memory session store for testing and development."""

    def __init__(self):
        self._sessions: Dict[str, Session] = {}

    def save(self, session: Session) -> None:
        """Save session to memory."""
        self._sessions[session.session_id] = session

    def load(self, session_id: str) -> Optional[Session]:
        """Load session from memory."""
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> bool:
        """Delete session from memory."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def list_sessions(
        self,
        workspace: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Session]:
        """List sessions from memory with filtering."""
        sessions = list(self._sessions.values())

        if workspace:
            sessions = [s for s in sessions if s.workspace == workspace]
        if user_id:
            sessions = [s for s in sessions if s.user_id == user_id]

        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions[offset : offset + limit]

    def clear(self) -> None:
        """Clear all sessions."""
        self._sessions.clear()


class FileSystemSessionStore(SessionStore):
    """File-based session store with JSON persistence.

    Stores each session as a separate JSON file in the specified directory.
    Files are named: {session_id}.json
    """

    def __init__(self, storage_dir: str = ".coder/sessions"):
        """Initialize file system store.

        Args:
            storage_dir: Directory to store session JSON files (relative to cwd)
        """
        self.storage_dir = Path.cwd() / storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _get_file_path(self, session_id: str) -> Path:
        """Get file path for a session."""
        return self.storage_dir / f"{session_id}.json"

    def _serialize_datetime(self, obj: Any) -> Any:
        """Serialize datetime objects for JSON."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def _deserialize_datetime(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize ISO format datetime strings."""
        datetime_fields = {"timestamp", "created_at", "updated_at"}
        for key, value in data.items():
            if key in datetime_fields and isinstance(value, str):
                try:
                    data[key] = datetime.fromisoformat(value)
                except ValueError:
                    pass  # Keep as string if parsing fails
        return data

    def save(self, session: Session) -> None:
        """Save session to JSON file."""
        file_path = self._get_file_path(session.session_id)
        data = session.model_dump()

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=self._serialize_datetime)

    def load(self, session_id: str) -> Optional[Session]:
        """Load session from JSON file."""
        file_path = self._get_file_path(session_id)

        if not file_path.exists():
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Deserialize datetime fields
            data = self._deserialize_datetime(data)
            if "messages" in data:
                data["messages"] = [self._deserialize_datetime(msg) for msg in data["messages"]]

            return Session.model_validate(data)
        except (json.JSONDecodeError, FileNotFoundError, PermissionError):
            return None

    def delete(self, session_id: str) -> bool:
        """Delete session file."""
        file_path = self._get_file_path(session_id)

        try:
            if file_path.exists():
                file_path.unlink()
                return True
        except (PermissionError, OSError):
            pass
        return False

    def list_sessions(
        self,
        workspace: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Session]:
        """List sessions from file system with filtering."""
        sessions: List[Session] = []

        # Iterate over all JSON files in storage directory
        for file_path in sorted(self.storage_dir.glob("*.json")):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Quick filter check before full deserialization
                if workspace and data.get("workspace") != workspace:
                    continue
                if user_id and data.get("user_id") != user_id:
                    continue

                # Deserialize and add to list
                data = self._deserialize_datetime(data)
                if "messages" in data:
                    data["messages"] = [self._deserialize_datetime(msg) for msg in data["messages"]]

                session = Session.model_validate(data)
                sessions.append(session)
            except (json.JSONDecodeError, FileNotFoundError, PermissionError):
                continue  # Skip corrupted or inaccessible files

        # Sort by updated_at descending
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions[offset : offset + limit]

    def clear(self) -> None:
        """Delete all session files."""
        for file_path in self.storage_dir.glob("*.json"):
            try:
                file_path.unlink()
            except (PermissionError, OSError):
                pass

    def get_storage_size(self) -> int:
        """Get total storage size in bytes."""
        return sum(f.stat().st_size for f in self.storage_dir.glob("*.json"))

    def get_session_count(self) -> int:
        """Get total number of session files."""
        return len(list(self.storage_dir.glob("*.json")))


class SessionManager:
    """High-level session management interface.

    Provides convenient methods for creating, retrieving, and managing sessions.
    """

    def __init__(self, store: Optional[SessionStore] = None):
        """Initialize with optional custom store.

        Args:
            store: SessionStore implementation. Defaults to InMemorySessionStore.
        """
        self.store = store or FileSystemSessionStore()

    def create_session(
        self,
        workspace: str = "default",
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Session:
        """Create a new session.

        Args:
            workspace: Workspace context
            user_id: Optional user ID
            session_id: Optional custom session ID (auto-generated if None)
            title: Optional session title
            metadata: Optional initial metadata

        Returns:
            New Session instance
        """
        kwargs = {
            "workspace": workspace,
            "user_id": user_id,
            "title": title,
            "metadata": metadata or {},
        }
        if session_id is not None:
            kwargs["session_id"] = session_id

        session = Session(**kwargs)
        self.store.save(session)
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID."""
        return self.store.load(session_id)

    def save_session(self, session: Session) -> None:
        """Save session changes."""
        self.store.save(session)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        return self.store.delete(session_id)

    def list_sessions(
        self,
        workspace: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Session]:
        """List sessions with filtering."""
        return self.store.list_sessions(workspace=workspace, user_id=user_id, limit=limit)

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Message]:
        """Add message to existing session.

        Returns:
            Message if session found, None otherwise
        """
        session = self.get_session(session_id)
        if session:
            message = session.add_message(role, content, metadata)
            self.save_session(session)
            return message
        return None

    def get_or_create_session(
        self,
        session_id: Optional[str] = None,
        workspace: str = "default",
        user_id: Optional[str] = None,
    ) -> Session:
        """Get existing session or create new one.

        Args:
            session_id: Session ID to look up (used as new session ID if not found)
            workspace: Workspace for new session
            user_id: User ID for new session

        Returns:
            Existing or new Session
        """
        if session_id:
            session = self.get_session(session_id)
            if session:
                return session
            return self.create_session(workspace=workspace, user_id=user_id, session_id=session_id)
        return self.create_session(workspace=workspace, user_id=user_id)


# Singleton manager instance for convenience
_default_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get default session manager singleton."""
    global _default_manager
    if _default_manager is None:
        _default_manager = SessionManager()
    return _default_manager


def set_session_manager(manager: SessionManager) -> None:
    """Set default session manager."""
    global _default_manager
    _default_manager = manager
