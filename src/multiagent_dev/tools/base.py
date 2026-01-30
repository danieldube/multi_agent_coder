"""Tool abstractions for orchestrated agent actions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


class ToolExecutionError(RuntimeError):
    """Raised when a tool fails to execute successfully."""


@dataclass(frozen=True)
class ToolResult:
    """Result of executing a tool.

    Attributes:
        name: Tool name that produced the result.
        success: Whether the tool executed successfully.
        output: Structured output produced by the tool.
        error: Error message when execution failed.
    """

    name: str
    success: bool
    output: Any | None = None
    error: str | None = None


class Tool(ABC):
    """Base class for tools available to agents."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique name of the tool."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a human-readable description of the tool."""

    @property
    @abstractmethod
    def input_schema(self) -> dict[str, Any]:
        """Return a schema describing expected tool input."""

    @abstractmethod
    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Execute the tool with the provided arguments.

        Args:
            arguments: Structured input arguments for the tool.

        Returns:
            ToolResult describing the execution outcome.

        Raises:
            ToolExecutionError: If execution fails unexpectedly.
        """
