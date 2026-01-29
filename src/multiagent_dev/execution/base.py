"""Execution engine base types and interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExecutionResult:
    """Result of executing a command.

    Attributes:
        command: The command executed as a list of strings.
        stdout: Captured standard output.
        stderr: Captured standard error.
        exit_code: Exit code returned by the process.
        duration_s: Duration of the execution in seconds.
    """

    command: list[str]
    stdout: str
    stderr: str
    exit_code: int
    duration_s: float


class CodeExecutor(ABC):
    """Abstract base class for command execution engines."""

    @abstractmethod
    def run(
        self,
        command: list[str],
        cwd: Path | None = None,
        timeout_s: int | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        """Run a command and capture its results.

        Args:
            command: The command to execute.
            cwd: Optional working directory for the command.
            timeout_s: Optional timeout in seconds.
            env: Optional environment variables to include.

        Returns:
            ExecutionResult containing stdout, stderr, exit code, and duration.
        """

