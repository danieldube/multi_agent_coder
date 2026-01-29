"""Workspace manager for safe file operations within a repo root."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import unified_diff
from pathlib import Path


class WorkspacePathError(ValueError):
    """Raised when a path escapes the workspace root."""


class WorkspaceWriteError(RuntimeError):
    """Raised when a write is attempted but writes are disabled."""


@dataclass(frozen=True)
class WorkspaceManager:
    """Manage file operations scoped to a workspace root.

    Attributes:
        root: The root directory that bounds all file operations.
        allow_write: Whether write operations are permitted.
    """

    root: Path
    allow_write: bool = True

    def __post_init__(self) -> None:
        """Normalize the workspace root path."""

        object.__setattr__(self, "root", self.root.resolve())

    def list_files(self, pattern: str | None = None) -> list[Path]:
        """List files under the workspace.

        Args:
            pattern: Optional glob pattern to filter files.

        Returns:
            A list of paths relative to the workspace root.
        """

        if pattern is None:
            files = (path for path in self.root.rglob("*") if path.is_file())
        else:
            files = (path for path in self.root.rglob(pattern) if path.is_file())
        return sorted(path.relative_to(self.root) for path in files)

    def read_text(self, path: Path) -> str:
        """Read text content from a file within the workspace.

        Args:
            path: File path relative to the workspace root.

        Returns:
            File contents as text.
        """

        resolved = self._resolve_path(path)
        return resolved.read_text(encoding="utf-8")

    def write_text(self, path: Path, content: str) -> None:
        """Write text content to a file within the workspace.

        Args:
            path: File path relative to the workspace root.
            content: Text to write.
        """

        if not self.allow_write:
            raise WorkspaceWriteError("Workspace is read-only; writes are disabled.")
        resolved = self._resolve_path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")

    def file_exists(self, path: Path) -> bool:
        """Return whether a file exists within the workspace."""

        resolved = self._resolve_path(path)
        return resolved.exists()

    def compute_unified_diff(self, old: str, new: str, path: Path) -> str:
        """Compute a unified diff between old and new content.

        Args:
            old: Original text content.
            new: Updated text content.
            path: Path used for labeling the diff.

        Returns:
            Unified diff text.
        """

        diff = unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=str(path),
            tofile=str(path),
        )
        return "".join(diff)

    def _resolve_path(self, path: Path) -> Path:
        """Resolve a path to an absolute path within the workspace.

        Args:
            path: File path to resolve.

        Raises:
            WorkspacePathError: If the resolved path escapes the workspace root.
        """

        candidate = (self.root / path).resolve()
        if not candidate.is_relative_to(self.root):
            raise WorkspacePathError(f"Path '{path}' escapes workspace root")
        return candidate
