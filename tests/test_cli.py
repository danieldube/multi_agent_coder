from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from multiagent_dev.cli.main import app
from multiagent_dev.orchestrator import TaskResult


def test_cli_init_creates_config_file(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["init", str(tmp_path)])

    assert result.exit_code == 0
    config_path = tmp_path / "multiagent_dev.yaml"
    assert config_path.exists()
    data = json.loads(config_path.read_text(encoding="utf-8"))
    assert data["workspace_root"] == str(tmp_path.resolve())


def test_cli_plan_command_invokes_run_plan(monkeypatch: object) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    def fake_run_plan(
        *, description: str, workspace: Path, allow_write: bool, execution_mode: str
    ) -> tuple[TaskResult, str]:
        captured["description"] = description
        captured["workspace"] = workspace
        captured["allow_write"] = allow_write
        captured["execution_mode"] = execution_mode
        return TaskResult(task_id="task-1", completed=True, messages_processed=2), "Step 1"

    monkeypatch.setattr("multiagent_dev.cli.main.run_plan", fake_run_plan)

    result = runner.invoke(app, ["plan", "Do the thing", "--exec-mode", "local"])

    assert result.exit_code == 0
    assert "Plan summary:" in result.output
    assert "Step 1" in result.output
    assert "Task task-1 completed after 2 steps." in result.output
    assert captured["description"] == "Do the thing"
    assert captured["execution_mode"] == "local"


def test_cli_exec_command_invokes_run_agent(monkeypatch: object) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    def fake_run_agent(
        *, description: str, workspace: Path, agent_id: str, allow_write: bool, execution_mode: str
    ) -> TaskResult:
        captured["description"] = description
        captured["workspace"] = workspace
        captured["agent_id"] = agent_id
        captured["allow_write"] = allow_write
        captured["execution_mode"] = execution_mode
        return TaskResult(task_id="task-2", completed=False, messages_processed=1)

    monkeypatch.setattr("multiagent_dev.cli.main.run_agent", fake_run_agent)

    result = runner.invoke(app, ["exec", "planner", "Run it", "--exec-mode", "docker"])

    assert result.exit_code == 0
    assert "Task task-2 incomplete after 1 steps." in result.output
    assert captured["agent_id"] == "planner"
    assert captured["description"] == "Run it"
    assert captured["execution_mode"] == "docker"


def test_plan_only_returns_summary_message(tmp_path: Path) -> None:
    from multiagent_dev.agents.base import AgentMessage
    from multiagent_dev.agents.planner import PlannerAgent
    from multiagent_dev.execution.base import CodeExecutor, ExecutionResult
    from multiagent_dev.llm.base import LLMClient
    from multiagent_dev.memory.memory import MemoryService
    from multiagent_dev.memory.retrieval import InMemoryRetrievalService
    from multiagent_dev.orchestrator import Orchestrator
    from multiagent_dev.tools.builtins import build_default_tool_registry
    from multiagent_dev.workspace.manager import WorkspaceManager

    class FakeLLMClient(LLMClient):
        def complete_chat(
            self,
            messages: list[dict[str, str]],
            temperature: float = 0.2,
            max_tokens: int | None = None,
        ) -> str:
            return "1. Step one\n2. Step two"

    class FakeCodeExecutor(CodeExecutor):
        def run(
            self,
            command: list[str],
            cwd: Path | None = None,
            timeout_s: int | None = None,
            env: dict[str, str] | None = None,
        ) -> ExecutionResult:
            raise AssertionError("Executor should not run in plan-only test")

    memory = MemoryService()
    retrieval = InMemoryRetrievalService()
    llm = FakeLLMClient()
    workspace = WorkspaceManager(tmp_path)
    executor = FakeCodeExecutor()
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
        content="Plan the work",
        metadata={"task_id": "task-123", "plan_only": True},
    )

    responses = agent.handle_message(message)

    assert len(responses) == 1
    assert responses[0].recipient == "planner"
    assert responses[0].metadata["plan_only_summary"] is True
    assert responses[0].content == "- 1. Step one\n- 2. Step two"
