from __future__ import annotations

from typing import cast

from multiagent_dev.agents.base import Agent, AgentMessage
from multiagent_dev.approvals import ApprovalPolicy, ApprovalRequest
from multiagent_dev.execution.base import CodeExecutor
from multiagent_dev.llm.base import LLMClient
from multiagent_dev.memory.memory import MemoryService
from multiagent_dev.memory.retrieval import RetrievalService
from multiagent_dev.orchestrator import Orchestrator, OrchestratorError, UserTask
from multiagent_dev.tools.base import Tool, ToolResult
from multiagent_dev.tools.registry import ToolRegistry
from multiagent_dev.workspace.manager import WorkspaceManager


class StubAgent(Agent):
    def __init__(self, agent_id: str, responses: list[AgentMessage]) -> None:
        super().__init__(
            agent_id=agent_id,
            role="stub",
            llm_client=cast(LLMClient, object()),
            orchestrator=cast(Orchestrator, object()),
            workspace=cast(WorkspaceManager, object()),
            executor=cast(CodeExecutor, object()),
            memory=cast(MemoryService, object()),
            retrieval=cast(RetrievalService, object()),
        )
        self._responses = responses
        self.received: list[AgentMessage] = []

    def handle_message(self, message: AgentMessage) -> list[AgentMessage]:
        self.received.append(message)
        return list(self._responses)


class ApprovalAgent(Agent):
    def __init__(self, agent_id: str, approved: bool) -> None:
        super().__init__(
            agent_id=agent_id,
            role="approval",
            llm_client=cast(LLMClient, object()),
            orchestrator=cast(Orchestrator, object()),
            workspace=cast(WorkspaceManager, object()),
            executor=cast(CodeExecutor, object()),
            memory=cast(MemoryService, object()),
            retrieval=cast(RetrievalService, object()),
        )
        self._approved = approved

    def handle_message(self, message: AgentMessage) -> list[AgentMessage]:
        return [
            AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                content="approval response",
                metadata={
                    "approval_request_id": message.metadata.get("approval_request_id"),
                    "approved": self._approved,
                    "approver": "tester",
                },
            )
        ]


class EchoTool(Tool):
    @property
    def name(self) -> str:
        return "run_command"

    @property
    def description(self) -> str:
        return "Echo tool"

    @property
    def input_schema(self) -> dict[str, object]:
        return {"payload": "string"}

    def execute(self, arguments: dict[str, object]) -> ToolResult:
        return ToolResult(name=self.name, success=True, output=dict(arguments))


def test_orchestrator_routes_messages() -> None:
    memory = MemoryService()
    orchestrator = Orchestrator(memory, ToolRegistry())
    responder = StubAgent(
        agent_id="responder",
        responses=[],
    )
    sender_response = AgentMessage(
        sender="sender",
        recipient="responder",
        content="forwarded",
    )
    sender = StubAgent(agent_id="sender", responses=[sender_response])

    orchestrator.register_agent(sender)
    orchestrator.register_agent(responder)

    task = UserTask(task_id="task-1", description="start", initial_agent_id="sender")
    result = orchestrator.run_task(task)

    assert result.completed is True
    assert result.messages_processed == 2
    assert responder.received[0].content == "forwarded"
    assert len(memory.get_messages("task-1")) == 2


def test_orchestrator_unknown_agent_raises() -> None:
    memory = MemoryService()
    orchestrator = Orchestrator(memory, ToolRegistry())
    task = UserTask(task_id="task-2", description="start", initial_agent_id="missing")

    try:
        orchestrator.run_task(task)
    except OrchestratorError as exc:
        assert "Unknown agent" in str(exc)
    else:
        raise AssertionError("Expected OrchestratorError")


def test_orchestrator_requires_approval_for_tool_execution() -> None:
    memory = MemoryService()
    registry = ToolRegistry()
    registry.register(EchoTool())
    policy = ApprovalPolicy(
        mode="approval-required",
        require_execution_approval=True,
        user_proxy_agent_id="user_proxy",
    )
    orchestrator = Orchestrator(memory, registry, approval_policy=policy)
    orchestrator.register_agent(ApprovalAgent("user_proxy", approved=False))

    result = orchestrator.execute_tool_with_approval(
        "run_command",
        {"payload": "value"},
        caller="tester",
        request=ApprovalRequest(
            action="run_command",
            description="Run command?",
            metadata={"payload": "value"},
        ),
    )

    assert result.success is False
    assert "Approval rejected" in (result.error or "")
