"""Docker-based execution engine implementation."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from multiagent_dev.execution.base import CodeExecutor, ExecutionResult
from multiagent_dev.util.logging import get_logger


class DockerExecutor(CodeExecutor):
    """Execute commands inside a Docker container."""

    def __init__(self, workspace_root: Path, image: str) -> None:
        """Initialize the executor.

        Args:
            workspace_root: Root directory to bind-mount into the container.
            image: Docker image to run.
        """

        self._workspace_root = workspace_root.resolve()
        self._image = image
        self._logger = get_logger(self.__class__.__name__)

    def run(
        self,
        command: list[str],
        cwd: Path | None = None,
        timeout_s: int | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        """Run a command inside a Docker container.

        Args:
            command: The command to execute.
            cwd: Optional working directory inside the workspace.
            timeout_s: Optional timeout in seconds.
            env: Optional environment variables to include.

        Returns:
            ExecutionResult with stdout, stderr, exit code, and duration.
        """

        if not command:
            raise ValueError("Command must contain at least one argument.")

        docker_command = self._build_docker_command(command, cwd=cwd, env=env)
        start = time.monotonic()
        self._logger.info("Running command in Docker: %s", command)
        completed = subprocess.run(
            docker_command,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        duration = time.monotonic() - start
        self._logger.info(
            "Docker command finished with exit code %s in %.2fs.",
            completed.returncode,
            duration,
        )

        return ExecutionResult(
            command=list(command),
            stdout=completed.stdout,
            stderr=completed.stderr,
            exit_code=completed.returncode,
            duration_s=duration,
        )

    def _build_docker_command(
        self,
        command: list[str],
        cwd: Path | None,
        env: dict[str, str] | None,
    ) -> list[str]:
        container_cwd = self._resolve_container_cwd(cwd)
        docker_command = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{self._workspace_root}:/workspace",
            "-w",
            container_cwd,
        ]
        for key, value in (env or {}).items():
            docker_command.extend(["-e", f"{key}={value}"])
        docker_command.append(self._image)
        docker_command.extend(command)
        return docker_command

    def _resolve_container_cwd(self, cwd: Path | None) -> str:
        if cwd is None:
            return "/workspace"
        resolved = cwd.resolve()
        try:
            relative = resolved.relative_to(self._workspace_root)
        except ValueError as exc:
            raise ValueError("Working directory must be inside workspace root") from exc
        return f"/workspace/{relative.as_posix()}"
