from __future__ import annotations

from pathlib import Path

from multiagent_dev.execution.base import CodeExecutor, ExecutionResult
from multiagent_dev.tools.builtins import build_default_tool_registry
from multiagent_dev.tools.registry import ToolNotFoundError, ToolRegistry
from multiagent_dev.workspace.manager import WorkspaceManager


class FakeExecutor(CodeExecutor):
    def __init__(self, results: list[ExecutionResult]) -> None:
        self._results = list(results)
        self.commands: list[list[str]] = []

    def run(
        self,
        command: list[str],
        cwd: Path | None = None,
        timeout_s: int | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        self.commands.append(command)
        if not self._results:
            raise AssertionError("No more fake execution results available")
        return self._results.pop(0)


def test_tool_registry_registers_and_executes(tmp_path: Path) -> None:
    workspace = WorkspaceManager(tmp_path)
    results = [
        ExecutionResult(
            command=["echo", "hi"],
            stdout="hi",
            stderr="",
            exit_code=0,
            duration_s=0.01,
        )
    ]
    executor = FakeExecutor(results)
    registry = build_default_tool_registry(workspace, executor)

    write_result = registry.execute(
        "write_file", {"path": "notes.txt", "content": "hello"}
    )
    assert write_result.success is True

    read_result = registry.execute("read_file", {"path": "notes.txt"})
    assert read_result.success is True
    assert read_result.output == {"content": "hello"}

    list_result = registry.execute("list_files", {"pattern": "*.txt"})
    assert list_result.output == {"files": ["notes.txt"]}

    command_result = registry.execute("run_command", {"command": ["echo", "hi"]})
    assert command_result.success is True
    assert executor.commands == [["echo", "hi"]]


def test_tool_registry_missing_tool_raises() -> None:
    registry = ToolRegistry()

    try:
        registry.execute("missing", {})
    except ToolNotFoundError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("Expected ToolNotFoundError")
