"""Configuration models for multiagent-dev."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    """Top-level configuration for the application.

    Attributes:
        workspace_root: Root path of the workspace to operate on.
        llm: Configuration for the default LLM client.
    """

    workspace_root: Path = Path(".")
    llm: LLMConfig = field(default_factory=lambda: LLMConfig())


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for an LLM provider."""

    provider: str = "openai"
    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None
    timeout_s: float = 30.0
    max_retries: int = 2
