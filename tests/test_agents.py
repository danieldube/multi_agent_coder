from __future__ import annotations

from pathlib import Path
from typing import cast

from multiagent_dev.agents.base import AgentMessage
from multiagent_dev.agents.coder import CodingAgent, CodingAgentError
from multiagent_dev.agents.planner import PlannerAgent
from multiagent_dev.agents.reviewer import ReviewerAgent
from multiagent_dev.agents.tester import TesterAgent
from multiagent_dev.execution.base import CodeExecutor, ExecutionResult
from multiagent_dev.llm.base import LLMClient
from multiagent_dev.memory.memory import MemoryService
from multiagent_dev.memory.retrieval import InMemoryRetrievalService
from multiagent_dev.orchestrator import Orchestrator
from multiagent_dev.tools.builtins import build_default_tool_registry
from multiagent_dev.version_control.base import (
    VCSBranchResult,
    VCSCommitResult,
    VCSDiff,
    VCSStatus,
    VersionControlService,
)
from multiagent_dev.workspace.manager import WorkspaceManager


class FakeLLM(LLMClient):
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[list[dict[str, str]]] = []

    def complete_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        self.calls.append(messages)
        if not self._responses:
            raise AssertionError("No more fake responses available")
        return self._responses.pop(0)


class FakeExecutor(CodeExecutor):
    def __init__(self, results: list[ExecutionResult]) -> None:
        self._results = list(results)
        self.commands: list[list[str]] = []

    def run(
        self,
        command: list[str],
        cwd: Path | None = None,
        timeout_s: int | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        self.commands.append(command)
        if not self._results:
            raise AssertionError("No more fake execution results available")
        return self._results.pop(0)


class FakeVersionControl(VersionControlService):
    def __init__(self, diff_text: str) -> None:
        self._diff_text = diff_text

    def status(self) -> VCSStatus:
        return VCSStatus(entries=[], clean=True)

    def diff(self, paths: list[str] | None = None) -> VCSDiff:
        return VCSDiff(diff=self._diff_text)

    def commit(self, message: str, *, stage_all: bool = True) -> VCSCommitResult:
        raise AssertionError("Commit should not be called in reviewer tests")

    def create_branch(self, name: str, *, checkout: bool = True) -> VCSBranchResult:
        raise AssertionError("Branch creation should not be called in reviewer tests")


def test_planner_agent_creates_plan_messages() -> None:
    memory = MemoryService()
    retrieval = InMemoryRetrievalService()
    llm = FakeLLM(["1. Implement feature\n2. Run tests"])  # noqa: E501
    workspace = WorkspaceManager(Path("."))
    executor = FakeExecutor([])
    tool_registry = build_default_tool_registry(workspace, executor)
    orchestrator = Orchestrator(memory, tool_registry)
    agent = PlannerAgent(
        agent_id="planner",
        role="planner",
        llm_client=llm,
        orchestrator=orchestrator,
        workspace=workspace,
        executor=executor,
        memory=memory,
        retrieval=retrieval,
    )

    message = AgentMessage(
        sender="user",
        recipient="planner",
        content="Add a feature",
        metadata={"task_id": "task-123"},
    )
    responses = agent.handle_message(message)

    recipients = {response.recipient for response in responses}
    assert recipients == {"coder", "tester", "reviewer"}
    assert (
        memory.get_session_note("task-123", "plan:task-123")
        == "1. Implement feature\n2. Run tests"
    )


def test_coding_agent_writes_files_and_snapshots(tmp_path: Path) -> None:
    memory = MemoryService()
    retrieval = InMemoryRetrievalService()
    llm = FakeLLM(
        [
            "FILE: src/new_file.py\nCODE:\nprint('hello')",
        ]
    )
    workspace = WorkspaceManager(tmp_path)
    executor = FakeExecutor([])
    tool_registry = build_default_tool_registry(workspace, executor)
    orchestrator = Orchestrator(memory, tool_registry)
    agent = CodingAgent(
        agent_id="coder",
        role="coder",
        llm_client=llm,
        orchestrator=orchestrator,
        workspace=workspace,
        executor=executor,
        memory=memory,
        retrieval=retrieval,
    )

    message = AgentMessage(
        sender="planner",
        recipient="coder",
        content="Create a new file",
    )
    responses = agent.handle_message(message)

    file_path = tmp_path / "src" / "new_file.py"
    assert file_path.read_text(encoding="utf-8") == "print('hello')\n"
    assert (
        memory.get_session_note("default", "file_snapshot:src/new_file.py") == ""
    )
    assert {response.recipient for response in responses} == {"reviewer", "planner"}


def test_coding_agent_requires_file_blocks(tmp_path: Path) -> None:
    memory = MemoryService()
    retrieval = InMemoryRetrievalService()
    llm = FakeLLM(["No file updates here"])  # noqa: E501
    workspace = WorkspaceManager(tmp_path)
    executor = FakeExecutor([])
    tool_registry = build_default_tool_registry(workspace, executor)
    orchestrator = Orchestrator(memory, tool_registry)
    agent = CodingAgent(
        agent_id="coder",
        role="coder",
        llm_client=llm,
        orchestrator=orchestrator,
        workspace=workspace,
        executor=executor,
        memory=memory,
        retrieval=retrieval,
    )

    message = AgentMessage(
        sender="planner",
        recipient="coder",
        content="Do something",
    )

    try:
        agent.handle_message(message)
    except CodingAgentError as exc:
        assert "No file updates" in str(exc)
    else:
        raise AssertionError("Expected CodingAgentError")


def test_reviewer_agent_approves_changes(tmp_path: Path) -> None:
    memory = MemoryService()
    retrieval = InMemoryRetrievalService()
    llm = FakeLLM(["Approved. Looks good."])  # noqa: E501
    workspace = WorkspaceManager(tmp_path)
    executor = FakeExecutor([])
    tool_registry = build_default_tool_registry(workspace, executor)
    orchestrator = Orchestrator(memory, tool_registry)
    file_path = tmp_path / "example.py"
    file_path.write_text("print('new')\n", encoding="utf-8")
    memory.save_session_note("default", "file_snapshot:example.py", "print('old')\n")

    agent = ReviewerAgent(
        agent_id="reviewer",
        role="reviewer",
        llm_client=llm,
        orchestrator=orchestrator,
        workspace=workspace,
        executor=executor,
        memory=memory,
        retrieval=retrieval,
    )

    message = AgentMessage(
        sender="coder",
        recipient="reviewer",
        content="Please review",
        metadata={"files": ["example.py"]},
    )
    responses = agent.handle_message(message)

    assert responses[0].metadata["approved"] is True
    assert responses[0].recipient == "planner"


def test_reviewer_agent_uses_vcs_diff(tmp_path: Path) -> None:
    memory = MemoryService()
    retrieval = InMemoryRetrievalService()
    llm = FakeLLM(["Approved."])  # noqa: E501
    workspace = WorkspaceManager(tmp_path)
    executor = FakeExecutor([])
    version_control = FakeVersionControl("diff --git a/file.py b/file.py")
    tool_registry = build_default_tool_registry(
        workspace, executor, version_control=version_control
    )
    orchestrator = Orchestrator(memory, tool_registry)

    agent = ReviewerAgent(
        agent_id="reviewer",
        role="reviewer",
        llm_client=llm,
        orchestrator=orchestrator,
        workspace=workspace,
        executor=executor,
        memory=memory,
        retrieval=retrieval,
    )

    message = AgentMessage(
        sender="coder",
        recipient="reviewer",
        content="Please review",
        metadata={"files": ["file.py"]},
    )
    agent.handle_message(message)

    prompt = llm.calls[0][1]["content"]
    assert "diff --git a/file.py b/file.py" in prompt


def test_tester_agent_runs_commands() -> None:
    memory = MemoryService()
    retrieval = InMemoryRetrievalService()
    llm = cast(LLMClient, object())
    workspace = WorkspaceManager(Path("."))
    results = [
        ExecutionResult(command=["pytest"], stdout="ok", stderr="", exit_code=0, duration_s=0.1),
        ExecutionResult(command=["ruff"], stdout="", stderr="fail", exit_code=1, duration_s=0.2),
    ]
    executor = FakeExecutor(results)
    tool_registry = build_default_tool_registry(workspace, executor)
    orchestrator = Orchestrator(memory, tool_registry)
    agent = TesterAgent(
        agent_id="tester",
        role="tester",
        llm_client=llm,
        orchestrator=orchestrator,
        workspace=workspace,
        executor=executor,
        memory=memory,
        retrieval=retrieval,
        test_commands=[["pytest"], ["ruff"]],
    )

    message = AgentMessage(
        sender="planner",
        recipient="tester",
        content="Run tests",
    )
    responses = agent.handle_message(message)

    assert executor.commands == [["pytest"], ["ruff"]]
    assert {response.recipient for response in responses} == {"reviewer", "planner"}
    assert responses[0].metadata["succeeded"] is False
