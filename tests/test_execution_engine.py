from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from multiagent_dev.execution.docker_exec import DockerExecutor
from multiagent_dev.execution.local_exec import LocalExecutor


def test_local_executor_runs_command(monkeypatch: Any) -> None:
    calls: dict[str, Any] = {}

    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls["args"] = args
        calls["kwargs"] = kwargs
        return subprocess.CompletedProcess(args[0], 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    executor = LocalExecutor()
    result = executor.run(
        ["echo", "hi"],
        cwd=Path("/tmp"),
        timeout_s=5,
        env={"LOCAL_EXEC_TEST": "1"},
    )

    assert result.exit_code == 0
    assert result.stdout == "ok"
    assert result.stderr == ""
    assert result.duration_s >= 0

    kwargs = calls["kwargs"]
    assert kwargs["cwd"] == "/tmp"
    assert kwargs["timeout"] == 5
    assert kwargs["env"]["LOCAL_EXEC_TEST"] == "1"


def test_docker_executor_builds_command(monkeypatch: Any) -> None:
    calls: dict[str, Any] = {}

    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls["args"] = args
        calls["kwargs"] = kwargs
        return subprocess.CompletedProcess(args[0], 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    workspace = Path("/repo")
    executor = DockerExecutor(workspace_root=workspace, image="python:3.11-slim")
    executor.run(
        ["pytest", "-q"],
        cwd=workspace / "tests",
        timeout_s=10,
        env={"PYTHONUNBUFFERED": "1"},
    )

    command = calls["args"][0]
    expected_prefix = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{workspace.resolve()}:/workspace",
        "-w",
        "/workspace/tests",
        "-e",
        "PYTHONUNBUFFERED=1",
        "python:3.11-slim",
    ]
    assert command[: len(expected_prefix)] == expected_prefix
    assert command[len(expected_prefix) :] == ["pytest", "-q"]
    assert calls["kwargs"]["timeout"] == 10


def test_docker_executor_rejects_external_cwd() -> None:
    executor = DockerExecutor(workspace_root=Path("/repo"), image="python:3.11")

    try:
        executor.run(["ls"], cwd=Path("/tmp"))
    except ValueError as exc:
        assert "workspace" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_local_executor_merges_environment(monkeypatch: Any) -> None:
    calls: dict[str, Any] = {}

    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls["kwargs"] = kwargs
        return subprocess.CompletedProcess(args[0], 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    executor = LocalExecutor()
    executor.run(["true"], env={"EXTRA_ENV": "yes"})

    merged_env = calls["kwargs"]["env"]
    assert merged_env["EXTRA_ENV"] == "yes"
    assert os.environ.items() <= merged_env.items()

