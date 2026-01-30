"""Reviewer agent implementation."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import unified_diff
from pathlib import Path

from multiagent_dev.agents.base import Agent, AgentMessage
from multiagent_dev.orchestrator import OrchestratorError


@dataclass(frozen=True)
class ReviewDecision:
    """Represents a review decision made by the reviewer agent.

    Attributes:
        approved: Whether the changes are approved.
        comments: Review feedback or approval notes.
    """

    approved: bool
    comments: str


class ReviewerAgent(Agent):
    """Agent responsible for reviewing code changes."""

    def handle_message(self, message: AgentMessage) -> list[AgentMessage]:
        """Review changes and respond with approval or feedback.

        Args:
            message: Incoming message describing updated files.

        Returns:
            Messages directing next steps based on review outcome.
        """

        diff_text = self._collect_diffs(message)
        prompt = self._build_prompt(diff_text)
        response = self._llm_client.complete_chat(prompt)
        decision = self._parse_decision(response)

        summary = {
            "approved": decision.approved,
            "comments": decision.comments,
        }
        messages = [
            AgentMessage(
                sender=self.agent_id,
                recipient="planner",
                content=decision.comments,
                metadata=summary,
            )
        ]
        if decision.approved:
            return messages

        messages.append(
            AgentMessage(
                sender=self.agent_id,
                recipient="coder",
                content=decision.comments,
                metadata=summary,
            )
        )
        return messages

    def _collect_diffs(self, message: AgentMessage) -> str:
        """Collect diffs for modified files.

        Args:
            message: Incoming message that may include modified file paths.

        Returns:
            Combined diff text for review.
        """

        file_paths = message.metadata.get("files", [])
        vcs_diff = self._try_collect_vcs_diff(file_paths)
        if vcs_diff is not None:
            return vcs_diff
        session_id = message.metadata.get("task_id", "default")
        diffs: list[str] = []
        for file_path in file_paths:
            path = Path(file_path)
            new_content = self._read_file(path)
            previous = (
                self._memory.get_session_note(session_id, self._snapshot_key(path)) or ""
            )
            diff = self._compute_diff(previous, new_content, path)
            diffs.append(diff or f"No changes detected in {file_path}\n")
        return "\n".join(diffs).strip()

    def _try_collect_vcs_diff(self, file_paths: list[str]) -> str | None:
        """Attempt to collect diffs using the version control tool."""

        try:
            result = self.use_tool("vcs_diff", {"paths": file_paths or None})
        except OrchestratorError:
            return None
        if not result.success or not isinstance(result.output, dict):
            return None
        diff_text = result.output.get("diff")
        if isinstance(diff_text, str):
            return diff_text
        return None

    def _build_prompt(self, diff_text: str) -> list[dict[str, str]]:
        """Build the prompt for reviewing changes.

        Args:
            diff_text: Combined diff text for review.

        Returns:
            Chat messages for the LLM.
        """

        system_prompt = (
            "You are a reviewer agent. Evaluate the diff and state whether "
            "the changes are approved. Respond with clear approval or needed "
            "changes."
        )
        user_prompt = f"Review the following diff:\n{diff_text}"
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _parse_decision(self, response: str) -> ReviewDecision:
        """Parse the LLM response into an approval decision.

        Args:
            response: Raw LLM response text.

        Returns:
            Structured review decision.
        """

        normalized = response.strip()
        lower = normalized.lower()
        rejected_markers = ("reject", "changes requested", "not approve")
        approved = "approve" in lower and not any(marker in lower for marker in rejected_markers)
        return ReviewDecision(approved=approved, comments=normalized)

    def _snapshot_key(self, path: Path) -> str:
        """Generate the memory key for file snapshots.

        Args:
            path: File path relative to the workspace.

        Returns:
            Memory key for file snapshots.
        """

        return f"file_snapshot:{path}"

    def _read_file(self, path: Path) -> str:
        """Read file contents via the tool registry."""

        result = self.use_tool("read_file", {"path": str(path)})
        if not result.success or not isinstance(result.output, dict):
            return f"Unable to read {path}: {result.error or 'unknown error'}"
        content = result.output.get("content")
        if not isinstance(content, str):
            return f"Unable to read {path}: invalid content"
        return content

    def _compute_diff(self, old: str, new: str, path: Path) -> str:
        """Compute a unified diff for review output."""

        diff = unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=str(path),
            tofile=str(path),
        )
        return "".join(diff)
