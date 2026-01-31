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

        session_id = message.metadata.get("task_id", "default")
        for update in updates:
            exists = self.use_tool("file_exists", {"path": str(update.path)})
            if not exists.success or not isinstance(exists.output, dict):
                raise CodingAgentError(
                    f"Failed to check file existence for {update.path}: {exists.error}"
                )
            if exists.output.get("exists"):
                previous = self._read_file(update.path)
            else:
                previous = ""
            self._memory.save_session_note(
                session_id,
                self._snapshot_key(update.path),
                previous,
            )
            write_result = self.use_tool(
                "write_file",
                {"path": str(update.path), "content": update.content},
            )
            if not write_result.success:
                raise CodingAgentError(
                    f"Failed to write file {update.path}: {write_result.error}"
                )
            modified_files.append(str(update.path))
            self._retrieval.index_text(str(update.path), update.content)

        self.log_event(
            "agent.code_applied",
            {"task_id": session_id, "files": list(modified_files)},
        )
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

        files = ", ".join(self._list_workspace_files())
        system_prompt = (
            "You are a coding agent. Respond only with file updates using the "
            "format:\nFILE: path\nCODE:\n<full file content>"
        )
        user_prompt = (
            f"Instruction: {instruction}\n"
            f"Existing workspace files: {files}\n"
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

    def _list_workspace_files(self) -> list[str]:
        """Retrieve a list of files in the workspace via tools."""

        result = self.use_tool("list_files", {})
        if not result.success or not isinstance(result.output, dict):
            raise CodingAgentError(f"Failed to list files: {result.error}")
        files = result.output.get("files")
        if not isinstance(files, list):
            raise CodingAgentError("Invalid list_files output format")
        return [str(item) for item in files]

    def _read_file(self, path: Path) -> str:
        """Read file contents via tools."""

        result = self.use_tool("read_file", {"path": str(path)})
        if not result.success or not isinstance(result.output, dict):
            raise CodingAgentError(f"Failed to read file {path}: {result.error}")
        content = result.output.get("content")
        if not isinstance(content, str):
            raise CodingAgentError(f"Invalid read_file output for {path}")
        return content
