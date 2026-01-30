from __future__ import annotations

from pathlib import Path

from multiagent_dev.execution.base import CodeExecutor, ExecutionResult
from multiagent_dev.tools.builtins import build_default_tool_registry
from multiagent_dev.tools.registry import ToolNotFoundError, ToolRegistry
from multiagent_dev.version_control.base import (
    VCSBranchResult,
    VCSCommitResult,
    VCSDiff,
    VCSStatus,
    VersionControlService,
)
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


class FakeVersionControl(VersionControlService):
    def __init__(self) -> None:
        self.commits: list[str] = []
        self.branches: list[str] = []
        self.diff_calls: list[list[str] | None] = []

    def status(self) -> VCSStatus:
        return VCSStatus(entries=[" M file.py"], clean=False)

    def diff(self, paths: list[str] | None = None) -> VCSDiff:
        self.diff_calls.append(paths)
        return VCSDiff(diff="diff --git a/file.py b/file.py")

    def commit(self, message: str, *, stage_all: bool = True) -> VCSCommitResult:
        self.commits.append(message)
        return VCSCommitResult(commit_hash="abc123", message=message)

    def create_branch(self, name: str, *, checkout: bool = True) -> VCSBranchResult:
        self.branches.append(name)
        return VCSBranchResult(branch_name=name)


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
    version_control = FakeVersionControl()
    registry = build_default_tool_registry(workspace, executor, version_control=version_control)

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

    diff_result = registry.execute("vcs_diff", {"paths": ["file.py"]})
    assert diff_result.output == {"diff": "diff --git a/file.py b/file.py"}
    assert version_control.diff_calls == [["file.py"]]

    commit_result = registry.execute(
        "vcs_commit",
        {"message": "Add change", "approved": True, "approver": "reviewer"},
    )
    assert commit_result.success is True
    assert commit_result.output == {
        "commit_hash": "abc123",
        "message": "Add change",
        "approver": "reviewer",
    }
    assert version_control.commits == ["Add change"]


def test_tool_registry_missing_tool_raises() -> None:
    registry = ToolRegistry()

    try:
        registry.execute("missing", {})
    except ToolNotFoundError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("Expected ToolNotFoundError")


def test_vcs_commit_requires_approval(tmp_path: Path) -> None:
    workspace = WorkspaceManager(tmp_path)
    executor = FakeExecutor([])
    version_control = FakeVersionControl()
    registry = build_default_tool_registry(workspace, executor, version_control=version_control)

    result = registry.execute("vcs_commit", {"message": "Nope", "approved": False})

    assert result.success is False
    assert "approval" in (result.error or "")
    assert version_control.commits == []
