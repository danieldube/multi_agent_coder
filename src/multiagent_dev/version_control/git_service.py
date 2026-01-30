"""Git-based version control service implementation."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from multiagent_dev.version_control.base import (
    VCSBranchResult,
    VCSCommitResult,
    VCSDiff,
    VCSStatus,
    VersionControlError,
    VersionControlService,
)


@dataclass(frozen=True)
class GitCommandResult:
    """Represents a completed git command execution."""

    stdout: str
    stderr: str
    exit_code: int


class GitService(VersionControlService):
    """Version control service backed by the git CLI."""

    def __init__(self, workspace_root: Path, git_binary: str = "git") -> None:
        """Initialize the Git service.

        Args:
            workspace_root: Path to the workspace root.
            git_binary: Git binary to invoke.
        """

        self._root = workspace_root.resolve()
        self._git = git_binary

    def status(self) -> VCSStatus:
        """Return git status for the workspace."""

        result = self._run_git(["status", "--porcelain"])
        entries = [line for line in result.stdout.splitlines() if line]
        return VCSStatus(entries=entries, clean=not entries)

    def diff(self, paths: list[str] | None = None) -> VCSDiff:
        """Return the git diff for the workspace or specific paths."""

        command = ["diff", "--"]
        if paths:
            command.extend(paths)
        result = self._run_git(command)
        return VCSDiff(diff=result.stdout)

    def commit(self, message: str, *, stage_all: bool = True) -> VCSCommitResult:
        """Create a git commit for the workspace state."""

        if stage_all:
            self._run_git(["add", "--all"])
        self._run_git(["commit", "-m", message])
        commit_hash = self._run_git(["rev-parse", "HEAD"]).stdout.strip()
        if not commit_hash:
            raise VersionControlError("Unable to resolve commit hash after commit.")
        return VCSCommitResult(commit_hash=commit_hash, message=message)

    def create_branch(self, name: str, *, checkout: bool = True) -> VCSBranchResult:
        """Create a new git branch."""

        if checkout:
            self._run_git(["checkout", "-b", name])
        else:
            self._run_git(["branch", name])
        return VCSBranchResult(branch_name=name)

    def _run_git(self, args: list[str]) -> GitCommandResult:
        """Run a git command in the workspace."""

        command = [self._git, *args]
        completed = subprocess.run(
            command,
            cwd=self._root,
            capture_output=True,
            text=True,
            check=False,
        )
        result = GitCommandResult(
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
            exit_code=completed.returncode,
        )
        if result.exit_code != 0:
            raise VersionControlError(result.stderr.strip() or "Git command failed.")
        return result
