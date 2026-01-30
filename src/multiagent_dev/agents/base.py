"""Base agent abstractions."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from multiagent_dev.execution.base import CodeExecutor
    from multiagent_dev.llm.base import LLMClient
    from multiagent_dev.memory.memory import MemoryService
    from multiagent_dev.memory.retrieval import RetrievalService
    from multiagent_dev.orchestrator import Orchestrator
    from multiagent_dev.workspace.manager import WorkspaceManager
from multiagent_dev.tools.base import ToolResult


@dataclass
class AgentMessage:
    """A message exchanged between agents via the orchestrator.

    Attributes:
        sender: Identifier of the sender agent.
        recipient: Identifier of the recipient agent.
        content: The textual payload of the message.
        metadata: Additional structured data about the message.
    """

    sender: str
    recipient: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class Agent(ABC):
    """Base class for all agents in the system."""

    def __init__(
        self,
        agent_id: str,
        role: str,
        llm_client: LLMClient,
        orchestrator: Orchestrator,
        workspace: WorkspaceManager,
        executor: CodeExecutor,
        memory: MemoryService,
        retrieval: RetrievalService,
    ) -> None:
        """Initialize the agent with its dependencies.

        Args:
            agent_id: Unique identifier for the agent.
            role: Human-readable role description.
            llm_client: Client used to query the language model.
            orchestrator: Orchestrator coordinating message flow.
            workspace: Workspace manager for file operations.
            executor: Code execution engine.
            memory: Memory service for storing conversation data.
            retrieval: Retrieval service for indexed project context.
        """

        self._agent_id = agent_id
        self._role = role
        self._llm_client = llm_client
        self._orchestrator = orchestrator
        self._workspace = workspace
        self._executor = executor
        self._memory = memory
        self._retrieval = retrieval

    @property
    def agent_id(self) -> str:
        """Return the agent's identifier."""

        return self._agent_id

    @property
    def role(self) -> str:
        """Return the agent's role description."""

        return self._role

    @abstractmethod
    def handle_message(self, message: AgentMessage) -> list[AgentMessage]:
        """Process a message and return new messages to send.

        Args:
            message: The incoming message from another agent.

        Returns:
            A list of new messages to enqueue via the orchestrator.
        """

    async def handle_message_async(self, message: AgentMessage) -> list[AgentMessage]:
        """Asynchronously process a message.

        By default this delegates to the synchronous ``handle_message`` in a thread
        to avoid blocking the orchestrator event loop. Agents may override this
        method for true async behavior.

        Args:
            message: The incoming message from another agent.

        Returns:
            A list of new messages to enqueue via the orchestrator.
        """

        return await asyncio.to_thread(self.handle_message, message)

    def use_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Request tool execution via the orchestrator.

        Args:
            name: Name of the tool to execute.
            arguments: Structured arguments for the tool.

        Returns:
            ToolResult from the executed tool.
        """

        return self._orchestrator.execute_tool_with_approval(
            name, arguments, caller=self.agent_id
        )
