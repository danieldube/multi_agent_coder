"""Core orchestrator for coordinating agents and tasks."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field

from multiagent_dev.agents.base import Agent, AgentMessage
from multiagent_dev.approvals import ApprovalDecision, ApprovalPolicy, ApprovalRequest
from multiagent_dev.memory.memory import MemoryService
from multiagent_dev.tools.base import ToolResult
from multiagent_dev.tools.registry import ToolNotFoundError, ToolRegistry
from multiagent_dev.util.logging import get_logger


class OrchestratorError(RuntimeError):
    """Raised when orchestrator operations fail."""


@dataclass(frozen=True)
class UserTask:
    """Represents a high-level user task submitted to the system.

    Attributes:
        task_id: Unique identifier for the task.
        description: The user-provided task description.
        initial_agent_id: Agent that should receive the first message.
    """

    task_id: str
    description: str
    initial_agent_id: str = "planner"


@dataclass
class TaskResult:
    """Represents the result of running a task through the orchestrator.

    Attributes:
        task_id: Identifier of the executed task.
        completed: Whether the task completed successfully.
        messages_processed: Number of messages handled by the loop.
        history: All messages processed during the run.
    """

    task_id: str
    completed: bool
    messages_processed: int
    history: list[AgentMessage] = field(default_factory=list)


class Orchestrator:
    """Coordinates agents and routes messages between them."""

    def __init__(
        self,
        memory: MemoryService,
        tool_registry: ToolRegistry,
        approval_policy: ApprovalPolicy | None = None,
    ) -> None:
        """Initialize the orchestrator with shared services.

        Args:
            memory: Memory service for storing conversation history.
            tool_registry: Registry of tools agents can invoke.
            approval_policy: Policy controlling approval checkpoints.
        """

        self._agents: dict[str, Agent] = {}
        self._memory = memory
        self._tool_registry = tool_registry
        self._queue: deque[AgentMessage] = deque()
        self._approval_policy = approval_policy or ApprovalPolicy()
        self._pending_approvals: dict[str, ApprovalRequest] = {}
        self._approval_counter = 0
        self._logger = get_logger(self.__class__.__name__)

    def register_agent(self, agent: Agent) -> None:
        """Register an agent with the orchestrator.

        Args:
            agent: The agent instance to register.
        """

        self._agents[agent.agent_id] = agent
        self._logger.info("Registered agent '%s' with role '%s'.", agent.agent_id, agent.role)

    def get_agent(self, agent_id: str) -> Agent | None:
        """Retrieve a registered agent by ID."""

        return self._agents.get(agent_id)

    def send_message(self, message: AgentMessage) -> None:
        """Queue a message for delivery to an agent.

        Args:
            message: The message to enqueue.
        """

        self._queue.append(message)
        self._logger.debug(
            "Queued message from '%s' to '%s'.", message.sender, message.recipient
        )

    def execute_tool(
        self,
        name: str,
        arguments: dict[str, object],
        *,
        caller: str | None = None,
    ) -> ToolResult:
        """Execute a tool by name using the registered tool registry.

        Args:
            name: Name of the tool to execute.
            arguments: Structured arguments for the tool.
            caller: Optional agent identifier making the request.

        Returns:
            ToolResult produced by the tool.

        Raises:
            OrchestratorError: If the tool is not registered.
        """

        if self._approval_policy.requires_approval(name):
            raise OrchestratorError(
                f"Tool '{name}' requires explicit approval via request_approval."
            )
        self._logger.info(
            "Executing tool '%s' requested by '%s'.", name, caller or "unknown"
        )
        try:
            result = self._tool_registry.execute(name, arguments)
        except ToolNotFoundError as exc:
            self._logger.error("Tool '%s' not found.", name)
            raise OrchestratorError(str(exc)) from exc
        self._logger.debug(
            "Tool '%s' completed with success=%s.", result.name, result.success
        )
        return result

    def request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        """Request human approval via the configured user proxy agent.

        Args:
            request: The approval request to submit.

        Returns:
            ApprovalDecision returned by the user proxy.

        Raises:
            OrchestratorError: If the user proxy agent is missing.
        """

        if self._approval_policy.mode != "approval-required":
            return ApprovalDecision(
                approved=True,
                approver="system",
                notes="Autonomous mode: approval bypassed.",
            )

        request_id = self._next_approval_id()
        self._pending_approvals[request_id] = request
        agent_id = self._approval_policy.user_proxy_agent_id
        agent = self.get_agent(agent_id)
        if agent is None:
            raise OrchestratorError(f"User proxy agent '{agent_id}' is not registered.")

        message = AgentMessage(
            sender="orchestrator",
            recipient=agent_id,
            content=request.description,
            metadata={
                "approval_request_id": request_id,
                "action": request.action,
                "metadata": dict(request.metadata),
            },
        )
        responses = agent.handle_message(message)
        decision = self._extract_decision(request_id, responses)
        if decision is None:
            raise OrchestratorError(
                f"User proxy did not return a decision for request '{request_id}'."
            )
        return decision

    def execute_tool_with_approval(
        self,
        name: str,
        arguments: dict[str, object],
        *,
        caller: str | None = None,
        request: ApprovalRequest | None = None,
    ) -> ToolResult:
        """Execute a tool, requesting approval when required.

        Args:
            name: Name of the tool to execute.
            arguments: Tool arguments.
            caller: Optional agent identifier making the request.
            request: Optional explicit approval request details.

        Returns:
            ToolResult produced by the tool.
        """

        if self._approval_policy.requires_approval(name):
            approval_request = request or ApprovalRequest(
                action=name,
                description=f"Approve tool execution for '{name}'.",
                metadata={"arguments": dict(arguments), "caller": caller},
            )
            decision = self.request_approval(approval_request)
            if not decision.approved:
                return ToolResult(
                    name=name,
                    success=False,
                    output=None,
                    error=f"Approval rejected by {decision.approver}: {decision.notes or ''}".strip(),
                )
            arguments = dict(arguments)
            arguments.setdefault("approved", True)
            arguments.setdefault("approver", decision.approver)

        return self.execute_tool(name, arguments, caller=caller)

    def _next_approval_id(self) -> str:
        self._approval_counter += 1
        return f"approval-{self._approval_counter}"

    def _extract_decision(
        self,
        request_id: str,
        responses: Iterable[AgentMessage],
    ) -> ApprovalDecision | None:
        for response in responses:
            metadata = response.metadata
            if metadata.get("approval_request_id") != request_id:
                continue
            approved = metadata.get("approved")
            approver = metadata.get("approver")
            if not isinstance(approved, bool) or not isinstance(approver, str):
                continue
            return ApprovalDecision(
                approved=approved,
                approver=approver,
                notes=metadata.get("notes"),
            )
        return None

    def _dispatch(self, message: AgentMessage) -> Iterable[AgentMessage]:
        """Dispatch a message to the appropriate agent.

        Args:
            message: The message to deliver.

        Returns:
            Messages generated by the receiving agent.

        Raises:
            OrchestratorError: If the recipient agent is not registered.
        """

        agent = self.get_agent(message.recipient)
        if agent is None:
            self._logger.error("Attempted to dispatch to unknown agent '%s'.", message.recipient)
            raise OrchestratorError(f"Unknown agent: {message.recipient}")
        return agent.handle_message(message)

    def run_task(self, task: UserTask, max_steps: int = 100) -> TaskResult:
        """Run a task by processing messages until completion.

        Args:
            task: The user task to execute.
            max_steps: Safety limit for message processing.

        Returns:
            Summary of the run including the message history.
        """

        initial_message = AgentMessage(
            sender="user",
            recipient=task.initial_agent_id,
            content=task.description,
            metadata={"task_id": task.task_id},
        )
        self.send_message(initial_message)
        self._logger.info("Starting task '%s' with initial agent '%s'.", task.task_id, task.initial_agent_id)
        history: list[AgentMessage] = []
        processed = 0

        while self._queue and processed < max_steps:
            message = self._queue.popleft()
            history.append(message)
            self._memory.append_message(task.task_id, message)
            responses = self._dispatch(message)
            for response in responses:
                response.metadata.setdefault("task_id", task.task_id)
                self.send_message(response)
            processed += 1

        completed = not self._queue
        if completed:
            self._logger.info("Task '%s' completed after %s messages.", task.task_id, processed)
        else:
            self._logger.warning(
                "Task '%s' halted after reaching max steps (%s).", task.task_id, max_steps
            )
        return TaskResult(
            task_id=task.task_id,
            completed=completed,
            messages_processed=processed,
            history=history,
        )
