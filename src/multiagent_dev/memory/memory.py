"""Simple in-memory storage for messages and notes."""

from __future__ import annotations

from multiagent_dev.agents.base import AgentMessage


class MemoryService:
    """Stores conversation history and notes for a session."""

    def __init__(self) -> None:
        """Initialize empty memory stores."""

        self._conversations: dict[str, list[AgentMessage]] = {}
        self._notes: dict[str, str] = {}

    def append_message(self, session_id: str, message: AgentMessage) -> None:
        """Append a message to a session history.

        Args:
            session_id: Identifier for the conversation/session.
            message: Message to append.
        """

        self._conversations.setdefault(session_id, []).append(message)

    def get_messages(self, session_id: str) -> list[AgentMessage]:
        """Retrieve the messages stored for a session."""

        return list(self._conversations.get(session_id, []))

    def save_note(self, key: str, text: str) -> None:
        """Save a note keyed by a unique identifier."""

        self._notes[key] = text

    def get_note(self, key: str) -> str | None:
        """Return a stored note if present."""

        return self._notes.get(key)
