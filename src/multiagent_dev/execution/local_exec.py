"""Local execution engine implementation."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from multiagent_dev.execution.base import CodeExecutor, ExecutionResult


class LocalExecutor(CodeExecutor):
    """Execute commands on the local host."""

    def run(
        self,
        command: list[str],
        cwd: Path | None = None,
        timeout_s: int | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        """Run a command locally and capture its output.

        Args:
            command: The command to execute.
            cwd: Optional working directory.
            timeout_s: Optional timeout in seconds.
            env: Optional environment variables to include.

        Returns:
            ExecutionResult with stdout, stderr, exit code, and duration.
        """

        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)

        start = time.monotonic()
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd is not None else None,
            env=merged_env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        duration = time.monotonic() - start

        return ExecutionResult(
            command=list(command),
            stdout=completed.stdout,
            stderr=completed.stderr,
            exit_code=completed.returncode,
            duration_s=duration,
        )

