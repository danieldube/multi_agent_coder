from __future__ import annotations

from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

from multiagent_dev.version_control.git_service import GitService


def test_git_service_status_parses_output(tmp_path: Path) -> None:
    service = GitService(tmp_path, git_binary="git")
    completed = CompletedProcess(
        args=["git", "status", "--porcelain"],
        returncode=0,
        stdout=" M file.py\n?? new.txt\n",
        stderr="",
    )
    with patch("multiagent_dev.version_control.git_service.subprocess.run") as run:
        run.return_value = completed
        status = service.status()

    assert status.clean is False
    assert status.entries == [" M file.py", "?? new.txt"]


def test_git_service_diff_returns_text(tmp_path: Path) -> None:
    service = GitService(tmp_path, git_binary="git")
    completed = CompletedProcess(
        args=["git", "diff", "--"],
        returncode=0,
        stdout="diff --git a/file.py b/file.py\n",
        stderr="",
    )
    with patch("multiagent_dev.version_control.git_service.subprocess.run") as run:
        run.return_value = completed
        diff = service.diff()

    assert "diff --git" in diff.diff
