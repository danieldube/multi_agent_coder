"""Application wiring for CLI-friendly orchestration."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from multiagent_dev.agents.base import Agent
from multiagent_dev.agents.coder import CodingAgent
from multiagent_dev.agents.planner import PlannerAgent
from multiagent_dev.agents.reviewer import ReviewerAgent
from multiagent_dev.agents.tester import TesterAgent
from multiagent_dev.config import (
    AgentConfig,
    AppConfig,
    config_to_dict,
    load_config,
    update_executor_mode,
    update_workspace_root,
)
from multiagent_dev.execution.base import CodeExecutor
from multiagent_dev.execution.docker_exec import DockerExecutor
from multiagent_dev.execution.local_exec import LocalExecutor
from multiagent_dev.llm.base import LLMClient
from multiagent_dev.llm.registry import create_llm_client
from multiagent_dev.memory.memory import MemoryService
from multiagent_dev.orchestrator import Orchestrator, TaskResult, UserTask
from multiagent_dev.workspace.manager import WorkspaceManager


class AppConfigError(RuntimeError):
    """Raised when configuration or runtime setup fails."""


@dataclass(frozen=True)
class RuntimeContext:
    """Container for runtime services used by the orchestrator."""

    orchestrator: Orchestrator
    memory: MemoryService
    workspace: WorkspaceManager
    executor: CodeExecutor
    llm_client: LLMClient


def initialize_config(workspace: Path) -> Path:
    """Create a default configuration file in the workspace.

    Args:
        workspace: Workspace directory where the config should be written.

    Returns:
        Path to the generated configuration file.

    Raises:
        AppConfigError: If the config file already exists.
    """

    workspace = workspace.resolve()
    config_path = workspace / "multiagent_dev.yaml"
    if config_path.exists():
        raise AppConfigError(f"Config file already exists at {config_path}")
    config_path.write_text(
        json.dumps(config_to_dict(AppConfig(workspace_root=workspace)), indent=2),
        encoding="utf-8",
    )
    return config_path


def run_task(
    description: str,
    workspace: Path,
    allow_write: bool = True,
    execution_mode: str | None = None,
) -> TaskResult:
    """Run a task description through the orchestrator workflow.

    Args:
        description: User task description.
        workspace: Path to the workspace root.
        allow_write: Whether file writes are permitted.
        execution_mode: Optional override for the executor mode.

    Returns:
        TaskResult summarizing the orchestration run.
    """

    config = load_config(workspace)
    config = update_workspace_root(config, workspace.resolve())
    if execution_mode is not None:
        config = update_executor_mode(config, execution_mode)

    runtime = build_runtime(config, allow_write=allow_write)
    task = UserTask(task_id=str(uuid.uuid4()), description=description)
    return runtime.orchestrator.run_task(task)


def build_runtime(
    config: AppConfig,
    *,
    allow_write: bool = True,
    llm_client: LLMClient | None = None,
    executor: CodeExecutor | None = None,
) -> RuntimeContext:
    """Build runtime services for the orchestrator.

    Args:
        config: Application configuration.
        allow_write: Whether workspace writes are allowed.
        llm_client: Optional pre-built LLM client (for testing).
        executor: Optional pre-built executor (for testing).

    Returns:
        RuntimeContext with initialized services.
    """

    workspace = WorkspaceManager(config.workspace_root, allow_write=allow_write)
    memory = MemoryService()
    llm = llm_client or create_llm_client(config.llm)
    executor_instance = executor or _build_executor(config)
    orchestrator = Orchestrator(memory)
    agents = _build_agents(config.agents, orchestrator, workspace, executor_instance, memory, llm)
    for agent in agents:
        orchestrator.register_agent(agent)
    return RuntimeContext(
        orchestrator=orchestrator,
        memory=memory,
        workspace=workspace,
        executor=executor_instance,
        llm_client=llm,
    )


def _build_executor(config: AppConfig) -> CodeExecutor:
    mode = config.executor.mode.lower()
    if mode == "local":
        return LocalExecutor()
    if mode == "docker":
        return DockerExecutor(config.workspace_root, config.executor.docker_image)
    raise AppConfigError(f"Unknown executor mode: {config.executor.mode}")


def _build_agents(
    agent_configs: Iterable[AgentConfig],
    orchestrator: Orchestrator,
    workspace: WorkspaceManager,
    executor: CodeExecutor,
    memory: MemoryService,
    llm_client: LLMClient,
) -> list[Agent]:
    agents: list[Agent] = []
    for agent_config in agent_configs:
        agent = _build_agent(
            agent_config,
            orchestrator=orchestrator,
            workspace=workspace,
            executor=executor,
            memory=memory,
            llm_client=llm_client,
        )
        agents.append(agent)
    return agents


def _build_agent(
    agent_config: AgentConfig,
    *,
    orchestrator: Orchestrator,
    workspace: WorkspaceManager,
    executor: CodeExecutor,
    memory: MemoryService,
    llm_client: LLMClient,
) -> Agent:
    agent_type = agent_config.type.lower()
    if agent_type == "planner":
        return PlannerAgent(
            agent_id=agent_config.agent_id,
            role=agent_config.role,
            llm_client=llm_client,
            orchestrator=orchestrator,
            workspace=workspace,
            executor=executor,
            memory=memory,
        )
    if agent_type == "coder":
        return CodingAgent(
            agent_id=agent_config.agent_id,
            role=agent_config.role,
            llm_client=llm_client,
            orchestrator=orchestrator,
            workspace=workspace,
            executor=executor,
            memory=memory,
        )
    if agent_type == "reviewer":
        return ReviewerAgent(
            agent_id=agent_config.agent_id,
            role=agent_config.role,
            llm_client=llm_client,
            orchestrator=orchestrator,
            workspace=workspace,
            executor=executor,
            memory=memory,
        )
    if agent_type == "tester":
        return TesterAgent(
            agent_id=agent_config.agent_id,
            role=agent_config.role,
            llm_client=llm_client,
            orchestrator=orchestrator,
            workspace=workspace,
            executor=executor,
            memory=memory,
            test_commands=agent_config.test_commands,
        )
    raise AppConfigError(f"Unknown agent type: {agent_config.type}")
