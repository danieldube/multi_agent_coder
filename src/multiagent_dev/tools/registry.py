"""Registry for tool definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from multiagent_dev.tools.base import Tool, ToolExecutionError, ToolResult


class ToolRegistryError(RuntimeError):
    """Raised when tool registry operations fail."""


class ToolNotFoundError(ToolRegistryError):
    """Raised when a tool is not found in the registry."""


class ToolRegistrationError(ToolRegistryError):
    """Raised when a tool cannot be registered."""


@dataclass
class ToolRegistry:
    """Registry for tool instances.

    Attributes:
        _tools: Mapping of tool names to tool instances.
    """

    _tools: dict[str, Tool]

    def __init__(self) -> None:
        self._tools = {}

    def register(self, tool: Tool) -> None:
        """Register a tool instance by name.

        Args:
            tool: Tool to register.

        Raises:
            ToolRegistrationError: If a tool with the same name already exists.
        """

        if tool.name in self._tools:
            raise ToolRegistrationError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """Retrieve a tool by name.

        Args:
            name: Name of the tool to retrieve.

        Returns:
            Registered tool instance.

        Raises:
            ToolNotFoundError: If no tool exists with the given name.
        """

        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolNotFoundError(f"Tool '{name}' is not registered") from exc

    def list_tools(self) -> Iterable[Tool]:
        """Return all registered tools."""

        return list(self._tools.values())

    def execute(self, name: str, arguments: dict[str, object]) -> ToolResult:
        """Execute a named tool with the provided arguments.

        Args:
            name: Tool name to execute.
            arguments: Structured arguments for the tool.

        Returns:
            ToolResult produced by the tool.

        Raises:
            ToolExecutionError: If the tool raises during execution.
        """

        tool = self.get(name)
        return tool.execute(dict(arguments))
