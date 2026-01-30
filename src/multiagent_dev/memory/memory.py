"""Simple in-memory storage for messages and notes."""

from __future__ import annotations

from multiagent_dev.agents.base import AgentMessage


class MemoryService:
    """Stores short-term session memory and long-term project notes."""

    def __init__(self) -> None:
        """Initialize empty memory stores."""

        self._conversations: dict[str, list[AgentMessage]] = {}
        self._session_notes: dict[str, dict[str, str]] = {}
        self._project_notes: dict[str, str] = {}

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

    def save_session_note(self, session_id: str, key: str, text: str) -> None:
        """Save a note scoped to a session.

        Args:
            session_id: Identifier for the conversation/session.
            key: Unique key for the note.
            text: Note content to store.
        """

        self._session_notes.setdefault(session_id, {})[key] = text

    def get_session_note(self, session_id: str, key: str) -> str | None:
        """Return a session-scoped note if present."""

        return self._session_notes.get(session_id, {}).get(key)

    def save_project_note(self, key: str, text: str) -> None:
        """Save a long-term project note keyed by an identifier."""

        self._project_notes[key] = text

    def get_project_note(self, key: str) -> str | None:
        """Return a long-term project note if present."""

        return self._project_notes.get(key)

    def save_note(self, key: str, text: str) -> None:
        """Save a legacy note keyed by a unique identifier."""

        self.save_project_note(key, text)

    def get_note(self, key: str) -> str | None:
        """Return a stored legacy note if present."""

        return self.get_project_note(key)
