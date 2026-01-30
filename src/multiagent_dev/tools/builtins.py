"""Built-in tools for file and command execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from multiagent_dev.execution.base import CodeExecutor, ExecutionResult
from multiagent_dev.tools.base import Tool, ToolExecutionError, ToolResult
from multiagent_dev.tools.registry import ToolRegistry
from multiagent_dev.workspace.manager import WorkspaceManager


@dataclass
class RunCommandTool(Tool):
    """Tool that executes shell commands via the configured executor."""

    executor: CodeExecutor

    @property
    def name(self) -> str:  # noqa: D401 - short description
        return "run_command"

    @property
    def description(self) -> str:
        return "Run a command using the configured execution engine."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "command": "list[str]",
            "cwd": "string | null",
            "timeout_s": "int | null",
            "env": "dict[str, str] | null",
        }

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        command = arguments.get("command")
        if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
            raise ToolExecutionError("'command' must be a list of strings")
        cwd = arguments.get("cwd")
        timeout = arguments.get("timeout_s")
        env = arguments.get("env")

        try:
            result = self.executor.run(
                command=command,
                cwd=Path(cwd) if cwd else None,
                timeout_s=timeout,
                env=env,
            )
        except Exception as exc:  # pragma: no cover - defensive
            fallback = ExecutionResult(
                command=command,
                stdout="",
                stderr=str(exc),
                exit_code=1,
                duration_s=0.0,
            )
            return ToolResult(name=self.name, success=False, output=fallback, error=str(exc))
        return ToolResult(name=self.name, success=True, output=result)


@dataclass
class ReadFileTool(Tool):
    """Tool that reads a file from the workspace."""

    workspace: WorkspaceManager

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Read a text file from the workspace."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {"path": "string"}

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        path = arguments.get("path")
        if not isinstance(path, str):
            raise ToolExecutionError("'path' must be a string")
        try:
            content = self.workspace.read_text(Path(path))
        except FileNotFoundError as exc:
            return ToolResult(name=self.name, success=False, output=None, error=str(exc))
        except Exception as exc:  # pragma: no cover - defensive
            raise ToolExecutionError(str(exc)) from exc
        return ToolResult(name=self.name, success=True, output={"content": content})


@dataclass
class WriteFileTool(Tool):
    """Tool that writes a file to the workspace."""

    workspace: WorkspaceManager

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "Write a text file to the workspace."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {"path": "string", "content": "string"}

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        path = arguments.get("path")
        content = arguments.get("content")
        if not isinstance(path, str):
            raise ToolExecutionError("'path' must be a string")
        if not isinstance(content, str):
            raise ToolExecutionError("'content' must be a string")
        try:
            self.workspace.write_text(Path(path), content)
        except Exception as exc:  # pragma: no cover - defensive
            raise ToolExecutionError(str(exc)) from exc
        return ToolResult(name=self.name, success=True, output={"path": path})


@dataclass
class ListFilesTool(Tool):
    """Tool that lists files in the workspace."""

    workspace: WorkspaceManager

    @property
    def name(self) -> str:
        return "list_files"

    @property
    def description(self) -> str:
        return "List files in the workspace, optionally filtered by a glob."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {"pattern": "string | null"}

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        pattern = arguments.get("pattern")
        if pattern is not None and not isinstance(pattern, str):
            raise ToolExecutionError("'pattern' must be a string or null")
        files = [str(path) for path in self.workspace.list_files(pattern)]
        return ToolResult(name=self.name, success=True, output={"files": files})


@dataclass
class FileExistsTool(Tool):
    """Tool that checks whether a file exists in the workspace."""

    workspace: WorkspaceManager

    @property
    def name(self) -> str:
        return "file_exists"

    @property
    def description(self) -> str:
        return "Check whether a file exists in the workspace."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {"path": "string"}

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        path = arguments.get("path")
        if not isinstance(path, str):
            raise ToolExecutionError("'path' must be a string")
        exists = self.workspace.file_exists(Path(path))
        return ToolResult(name=self.name, success=True, output={"exists": exists})


def build_default_tool_registry(
    workspace: WorkspaceManager,
    executor: CodeExecutor,
) -> ToolRegistry:
    """Create a registry pre-populated with built-in tools.

    Args:
        workspace: Workspace manager to back file tools.
        executor: Executor to back command execution.

    Returns:
        ToolRegistry with built-in tools registered.
    """

    registry = ToolRegistry()
    registry.register(RunCommandTool(executor=executor))
    registry.register(ReadFileTool(workspace=workspace))
    registry.register(WriteFileTool(workspace=workspace))
    registry.register(ListFilesTool(workspace=workspace))
    registry.register(FileExistsTool(workspace=workspace))
    return registry
