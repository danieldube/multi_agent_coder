"""Core orchestrator for coordinating agents and tasks."""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from multiagent_dev.agents.base import Agent, AgentMessage
from multiagent_dev.approvals import ApprovalDecision, ApprovalPolicy, ApprovalRequest
from multiagent_dev.memory.memory import MemoryService
from multiagent_dev.tools.base import ToolExecutionError, ToolResult
from multiagent_dev.tools.registry import ToolNotFoundError, ToolRegistry
from multiagent_dev.util.logging import get_logger
from multiagent_dev.util.observability import ObservabilityManager, create_observability_manager


class OrchestratorError(RuntimeError):
    """Raised when orchestrator operations fail."""


@dataclass(frozen=True)
class UserTask:
    """Represents a high-level user task submitted to the system.

    Attributes:
        task_id: Unique identifier for the task.
        description: The user-provided task description.
        initial_agent_id: Agent that should receive the first message.
        initial_metadata: Metadata attached to the initial message.
    """

    task_id: str
    description: str
    initial_agent_id: str = "planner"
    initial_metadata: dict[str, Any] = field(default_factory=dict)


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


@dataclass
class WorkflowState:
    """Serializable workflow state for persisting and resuming tasks.

    Attributes:
        task_id: Identifier for the task.
        task_description: Description associated with the task.
        initial_agent_id: Initial agent that received the first message.
        pending_messages: Messages awaiting processing.
        history: Messages that have been processed.
        messages_processed: Count of messages processed so far.
        approval_counter: Counter used for approval request IDs.
        pending_approvals: Approvals currently awaiting a decision.
    """

    task_id: str
    task_description: str
    initial_agent_id: str
    pending_messages: list[AgentMessage]
    history: list[AgentMessage]
    messages_processed: int
    approval_counter: int
    pending_approvals: dict[str, ApprovalRequest]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation of the workflow state."""

        return {
            "task_id": self.task_id,
            "task_description": self.task_description,
            "initial_agent_id": self.initial_agent_id,
            "pending_messages": [_message_to_dict(msg) for msg in self.pending_messages],
            "history": [_message_to_dict(msg) for msg in self.history],
            "messages_processed": self.messages_processed,
            "approval_counter": self.approval_counter,
            "pending_approvals": {
                key: _approval_request_to_dict(value)
                for key, value in self.pending_approvals.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> WorkflowState:
        """Build a workflow state from its serialized representation."""

        pending = [
            _message_from_dict(item)
            for item in _expect_list(payload.get("pending_messages"))
        ]
        history = [
            _message_from_dict(item) for item in _expect_list(payload.get("history"))
        ]
        approvals_payload = _expect_dict(payload.get("pending_approvals"))
        pending_approvals = {
            key: _approval_request_from_dict(value)
            for key, value in approvals_payload.items()
        }
        return cls(
            task_id=_expect_str(payload.get("task_id")),
            task_description=_expect_str(payload.get("task_description")),
            initial_agent_id=_expect_str(payload.get("initial_agent_id")),
            pending_messages=pending,
            history=history,
            messages_processed=_expect_int(payload.get("messages_processed")),
            approval_counter=_expect_int(payload.get("approval_counter")),
            pending_approvals=pending_approvals,
        )


def _message_to_dict(message: AgentMessage) -> dict[str, object]:
    return {
        "sender": message.sender,
        "recipient": message.recipient,
        "content": message.content,
        "metadata": dict(message.metadata),
    }


def _message_from_dict(payload: object) -> AgentMessage:
    data = _expect_dict(payload)
    return AgentMessage(
        sender=_expect_str(data.get("sender")),
        recipient=_expect_str(data.get("recipient")),
        content=_expect_str(data.get("content")),
        metadata=dict(_expect_dict(data.get("metadata"))),
    )


def _approval_request_to_dict(request: ApprovalRequest) -> dict[str, object]:
    return {
        "action": request.action,
        "description": request.description,
        "metadata": dict(request.metadata),
    }


def _approval_request_from_dict(payload: object) -> ApprovalRequest:
    data = _expect_dict(payload)
    return ApprovalRequest(
        action=_expect_str(data.get("action")),
        description=_expect_str(data.get("description")),
        metadata=dict(_expect_dict(data.get("metadata"))),
    )


def _expect_str(value: object) -> str:
    if not isinstance(value, str):
        raise OrchestratorError("Serialized workflow state is missing a string value.")
    return value


def _expect_int(value: object) -> int:
    if not isinstance(value, int):
        raise OrchestratorError("Serialized workflow state is missing an integer value.")
    return value


def _expect_dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise OrchestratorError("Serialized workflow state is missing a dict value.")
    return value


def _expect_list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise OrchestratorError("Serialized workflow state is missing a list value.")
    return value


class Orchestrator:
    """Coordinates agents and routes messages between them."""

    def __init__(
        self,
        memory: MemoryService,
        tool_registry: ToolRegistry,
        approval_policy: ApprovalPolicy | None = None,
        observability: ObservabilityManager | None = None,
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
        self._agent_locks: dict[str, asyncio.Lock] = {}
        self._observability = observability or create_observability_manager()

    def register_agent(self, agent: Agent) -> None:
        """Register an agent with the orchestrator.

        Args:
            agent: The agent instance to register.
        """

        self._agents[agent.agent_id] = agent
        self._logger.info("Registered agent '%s' with role '%s'.", agent.agent_id, agent.role)
        self._observability.log_event(
            "agent.registered",
            {"agent_id": agent.agent_id, "role": agent.role},
        )

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
        self._observability.metrics.increment("orchestrator.messages_queued", 1)

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
        start = time.perf_counter()
        try:
            result = self._tool_registry.execute(name, arguments)
        except ToolNotFoundError as exc:
            self._logger.error("Tool '%s' not found.", name)
            self._observability.log_event(
                "tool.execution_failed",
                {"name": name, "caller": caller, "error": str(exc)},
            )
            raise OrchestratorError(str(exc)) from exc
        except ToolExecutionError as exc:
            duration = time.perf_counter() - start
            self._observability.metrics.increment("tool.executions", 1)
            self._observability.metrics.record_duration("tool.execution_time", duration)
            self._observability.log_event(
                "tool.execution_failed",
                {
                    "name": name,
                    "caller": caller,
                    "error": str(exc),
                    "duration_s": duration,
                },
            )
            raise OrchestratorError(str(exc)) from exc
        duration = time.perf_counter() - start
        self._observability.metrics.increment("tool.executions", 1)
        self._observability.metrics.record_duration("tool.execution_time", duration)
        self._observability.log_event(
            "tool.executed",
            {
                "name": name,
                "caller": caller,
                "success": result.success,
                "duration_s": duration,
            },
        )
        self._logger.debug(
            "Tool '%s' completed with success=%s.", result.name, result.success
        )
        return result

    def log_event(self, event_type: str, payload: dict[str, object]) -> None:
        """Log a structured event via the observability manager."""

        self._observability.log_event(event_type, payload)

    def metrics_snapshot(self) -> dict[str, object]:
        """Return the current metrics snapshot."""

        return self._observability.metrics.snapshot()

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

    async def _dispatch_async(self, message: AgentMessage) -> list[AgentMessage]:
        """Dispatch a message asynchronously, serializing access per agent."""

        agent = self.get_agent(message.recipient)
        if agent is None:
            self._logger.error("Attempted to dispatch to unknown agent '%s'.", message.recipient)
            raise OrchestratorError(f"Unknown agent: {message.recipient}")
        lock = self._agent_locks.setdefault(message.recipient, asyncio.Lock())
        async with lock:
            return await agent.handle_message_async(message)

    def _build_initial_state(self, task: UserTask) -> WorkflowState:
        metadata = {"task_id": task.task_id, **task.initial_metadata}
        initial_message = AgentMessage(
            sender="user",
            recipient=task.initial_agent_id,
            content=task.description,
            metadata=metadata,
        )
        return WorkflowState(
            task_id=task.task_id,
            task_description=task.description,
            initial_agent_id=task.initial_agent_id,
            pending_messages=[initial_message],
            history=[],
            messages_processed=0,
            approval_counter=self._approval_counter,
            pending_approvals=dict(self._pending_approvals),
        )

    def snapshot_state(
        self, task: UserTask, history: list[AgentMessage], processed: int
    ) -> WorkflowState:
        """Create a serializable snapshot of the current workflow state."""

        return WorkflowState(
            task_id=task.task_id,
            task_description=task.description,
            initial_agent_id=task.initial_agent_id,
            pending_messages=list(self._queue),
            history=list(history),
            messages_processed=processed,
            approval_counter=self._approval_counter,
            pending_approvals=dict(self._pending_approvals),
        )

    def save_state(self, state: WorkflowState, path: Path) -> None:
        """Persist workflow state to disk as JSON."""

        path.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")

    def load_state(self, path: Path) -> WorkflowState:
        """Load workflow state from disk."""

        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise OrchestratorError("Workflow state payload must be a JSON object.")
        return WorkflowState.from_dict(payload)

    async def run_task_async(
        self,
        task: UserTask,
        max_steps: int = 100,
        *,
        state: WorkflowState | None = None,
        checkpoint_path: Path | None = None,
    ) -> TaskResult:
        """Run a task asynchronously by processing messages until completion.

        Args:
            task: The user task to execute.
            max_steps: Safety limit for message processing.
            state: Optional workflow state for resuming a previous run.
            checkpoint_path: Optional path to persist workflow state after each batch.

        Returns:
            Summary of the run including the message history.
        """

        workflow_state = state or self._build_initial_state(task)
        if workflow_state.task_id != task.task_id:
            raise OrchestratorError("Task ID does not match workflow state.")
        if workflow_state.initial_agent_id != task.initial_agent_id:
            raise OrchestratorError("Initial agent does not match workflow state.")
        self._queue = deque(workflow_state.pending_messages)
        history: list[AgentMessage] = list(workflow_state.history)
        processed = workflow_state.messages_processed
        self._pending_approvals = dict(workflow_state.pending_approvals)
        self._approval_counter = workflow_state.approval_counter

        self._logger.info(
            "Starting task '%s' with initial agent '%s'.",
            task.task_id,
            task.initial_agent_id,
        )
        self._observability.log_event(
            "orchestrator.task_started",
            {"task_id": task.task_id, "initial_agent_id": task.initial_agent_id},
        )
        start = time.perf_counter()

        while self._queue and processed < max_steps:
            batch: list[AgentMessage] = []
            while self._queue:
                batch.append(self._queue.popleft())
            for message in batch:
                history.append(message)
                self._memory.append_message(task.task_id, message)
            self._observability.metrics.increment("orchestrator.batches", 1)
            self._observability.metrics.increment("orchestrator.messages_processed", len(batch))
            results = await asyncio.gather(
                *[self._dispatch_async(message) for message in batch]
            )
            for responses in results:
                for response in responses:
                    response.metadata.setdefault("task_id", task.task_id)
                    self.send_message(response)
            processed += len(batch)
            if checkpoint_path is not None:
                self.save_state(self.snapshot_state(task, history, processed), checkpoint_path)

        completed = not self._queue
        duration = time.perf_counter() - start
        self._observability.metrics.record_duration("orchestrator.task_duration", duration)
        self._observability.log_event(
            "orchestrator.task_finished",
            {
                "task_id": task.task_id,
                "completed": completed,
                "messages_processed": processed,
                "duration_s": duration,
            },
        )
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

    def resume_task(self, state: WorkflowState, max_steps: int = 100) -> TaskResult:
        """Resume a task from a serialized workflow state."""

        task = UserTask(
            task_id=state.task_id,
            description=state.task_description,
            initial_agent_id=state.initial_agent_id,
        )
        return asyncio.run(
            self.run_task_async(task, max_steps=max_steps, state=state)
        )

    def run_task(self, task: UserTask, max_steps: int = 100) -> TaskResult:
        """Run a task by processing messages until completion."""

        return asyncio.run(self.run_task_async(task, max_steps=max_steps))
