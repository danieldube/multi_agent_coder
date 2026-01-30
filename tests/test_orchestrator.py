from __future__ import annotations

from typing import cast

from multiagent_dev.agents.base import Agent, AgentMessage
from multiagent_dev.execution.base import CodeExecutor
from multiagent_dev.llm.base import LLMClient
from multiagent_dev.memory.memory import MemoryService
from multiagent_dev.memory.retrieval import RetrievalService
from multiagent_dev.orchestrator import Orchestrator, OrchestratorError, UserTask
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
