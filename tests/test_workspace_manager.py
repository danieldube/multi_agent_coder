from __future__ import annotations

from pathlib import Path

import pytest

from multiagent_dev.workspace.manager import WorkspaceManager, WorkspacePathError


def test_write_and_read_roundtrip(tmp_path: Path) -> None:
    manager = WorkspaceManager(tmp_path)
    relative_path = Path("nested") / "file.txt"

    manager.write_text(relative_path, "hello world")

    assert manager.read_text(relative_path) == "hello world"
    assert manager.file_exists(relative_path) is True


def test_list_files_filters_by_pattern(tmp_path: Path) -> None:
    manager = WorkspaceManager(tmp_path)
    manager.write_text(Path("a.py"), "print('a')")
    manager.write_text(Path("b.txt"), "b")

    files = manager.list_files("*.py")

    assert files == [Path("a.py")]


def test_compute_unified_diff(tmp_path: Path) -> None:
    manager = WorkspaceManager(tmp_path)
    diff = manager.compute_unified_diff("one\n", "one\ntwo\n", Path("demo.txt"))

    assert diff.startswith("--- demo.txt")
    assert "+two" in diff


def test_prevents_path_traversal(tmp_path: Path) -> None:
    manager = WorkspaceManager(tmp_path)

    with pytest.raises(WorkspacePathError):
        manager.read_text(Path("..") / "outside.txt")
