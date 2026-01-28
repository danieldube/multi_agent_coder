"""Configuration models for multiagent-dev."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    """Top-level configuration for the application.

    Attributes:
        workspace_root: Root path of the workspace to operate on.
    """

    workspace_root: Path = Path(".")
