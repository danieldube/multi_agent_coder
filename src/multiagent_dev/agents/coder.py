"""Coding agent implementation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from multiagent_dev.agents.base import Agent, AgentMessage


class CodingAgentError(RuntimeError):
    """Raised when the coding agent cannot apply updates."""


@dataclass(frozen=True)
class FileUpdate:
    """Represents a file update produced by the coding agent.

    Attributes:
        path: File path relative to the workspace root.
        content: Full file content to write.
    """

    path: Path
    content: str


class CodingAgent(Agent):
    """Agent responsible for applying code changes."""

    def handle_message(self, message: AgentMessage) -> list[AgentMessage]:
        """Generate code changes and apply them to the workspace.

        Args:
            message: Incoming instruction message.

        Returns:
            Messages for downstream agents indicating completion.
        """

        prompt = self._build_prompt(message.content)
        response = self._llm_client.complete_chat(prompt)
        updates = self._parse_updates(response)
        modified_files: list[str] = []

        for update in updates:
            if self._workspace.file_exists(update.path):
                previous = self._workspace.read_text(update.path)
                self._memory.save_note(self._snapshot_key(update.path), previous)
            else:
                self._memory.save_note(self._snapshot_key(update.path), "")
            self._workspace.write_text(update.path, update.content)
            modified_files.append(str(update.path))

        summary = "Updated files: " + ", ".join(modified_files)
        metadata = {"files": modified_files}
        return [
            AgentMessage(
                sender=self.agent_id,
                recipient="reviewer",
                content=summary,
                metadata=metadata,
            ),
            AgentMessage(
                sender=self.agent_id,
                recipient="planner",
                content=summary,
                metadata=metadata,
            ),
        ]

    def _build_prompt(self, instruction: str) -> list[dict[str, str]]:
        """Build the prompt for requesting code changes.

        Args:
            instruction: Instruction from the planner or reviewer.

        Returns:
            Chat messages for the LLM.
        """

        files = ", ".join(str(path) for path in self._workspace.list_files("*.py"))
        system_prompt = (
            "You are a coding agent. Respond only with file updates using the "
            "format:\nFILE: path\nCODE:\n<full file content>"
        )
        user_prompt = (
            f"Instruction: {instruction}\n"
            f"Existing Python files: {files}\n"
            "Provide full file contents for any files you modify."
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _parse_updates(self, response: str) -> list[FileUpdate]:
        """Parse the LLM response into file updates.

        Args:
            response: Raw LLM output.

        Returns:
            List of file updates.

        Raises:
            CodingAgentError: If no file updates are found.
        """

        updates: list[FileUpdate] = []
        current_path: Path | None = None
        buffer: list[str] = []

        def flush() -> None:
            if current_path is None:
                return
            content = "\n".join(buffer).rstrip("\n") + "\n"
            updates.append(FileUpdate(path=current_path, content=content))

        for line in response.splitlines():
            if line.startswith("FILE:"):
                flush()
                path_str = line.removeprefix("FILE:").strip()
                current_path = Path(path_str)
                buffer = []
                continue
            if line.startswith("CODE:"):
                continue
            buffer.append(line)

        flush()

        if not updates:
            raise CodingAgentError("No file updates found in LLM response")
        return updates

    def _snapshot_key(self, path: Path) -> str:
        """Generate the memory key for a file snapshot.

        Args:
            path: File path relative to the workspace.

        Returns:
            Key used to store file snapshots.
        """

        return f"file_snapshot:{path}"
