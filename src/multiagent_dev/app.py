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
from multiagent_dev.agents.user_proxy import UserProxyAgent
from multiagent_dev.approvals import ApprovalPolicy
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
from multiagent_dev.memory.retrieval import InMemoryRetrievalService, RetrievalService
from multiagent_dev.orchestrator import Orchestrator, TaskResult, UserTask
from multiagent_dev.tools.builtins import build_default_tool_registry
from multiagent_dev.util.logging import get_logger
from multiagent_dev.util.observability import ObservabilityManager, create_observability_manager
from multiagent_dev.version_control.base import VersionControlService
from multiagent_dev.version_control.git_service import GitService
from multiagent_dev.workspace.manager import WorkspaceManager


class AppConfigError(RuntimeError):
    """Raised when configuration or runtime setup fails."""


@dataclass(frozen=True)
class RuntimeContext:
    """Container for runtime services used by the orchestrator."""

    orchestrator: Orchestrator
    memory: MemoryService
    retrieval: RetrievalService
    workspace: WorkspaceManager
    executor: CodeExecutor
    llm_client: LLMClient
    version_control: VersionControlService | None
    observability: ObservabilityManager


_LOGGER = get_logger("multiagent_dev.app")


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
        raise AppConfigError(
            f"Config file already exists at {config_path}. Remove it or choose another "
            "workspace."
        )
    config_path.write_text(
        json.dumps(config_to_dict(AppConfig(workspace_root=workspace)), indent=2),
        encoding="utf-8",
    )
    _LOGGER.info("Initialized configuration at %s", config_path)
    return config_path


def run_task(
    description: str,
    workspace: Path,
    allow_write: bool = True,
    allow_exec: bool = True,
    execution_mode: str | None = None,
    agent_profile: str | None = None,
) -> TaskResult:
    """Run a task description through the orchestrator workflow.

    Args:
        description: User task description.
        workspace: Path to the workspace root.
        allow_write: Whether file writes are permitted.
        allow_exec: Whether command execution tools are enabled.
        execution_mode: Optional override for the executor mode.
        agent_profile: Optional agent profile name to select a subset of agents.

    Returns:
        TaskResult summarizing the orchestration run.
    """

    runtime = _build_runtime_from_workspace(
        workspace,
        allow_write=allow_write,
        allow_exec=allow_exec,
        execution_mode=execution_mode,
        agent_profile=agent_profile,
    )
    task = UserTask(task_id=str(uuid.uuid4()), description=description)
    return runtime.orchestrator.run_task(task)


def run_plan(
    description: str,
    workspace: Path,
    allow_write: bool = True,
    allow_exec: bool = True,
    execution_mode: str | None = None,
    agent_profile: str | None = None,
) -> tuple[TaskResult, str | None]:
    """Generate a plan for the task without running implementation agents."""

    runtime = _build_runtime_from_workspace(
        workspace,
        allow_write=allow_write,
        allow_exec=allow_exec,
        execution_mode=execution_mode,
        agent_profile=agent_profile,
    )
    task = UserTask(
        task_id=str(uuid.uuid4()),
        description=description,
        initial_agent_id="planner",
        initial_metadata={"plan_only": True},
    )
    result = runtime.orchestrator.run_task(task)
    plan_summary = next(
        (
            message.content
            for message in result.history
            if message.metadata.get("plan_only_summary")
        ),
        None,
    )
    return result, plan_summary


def run_agent(
    description: str,
    workspace: Path,
    agent_id: str,
    allow_write: bool = True,
    allow_exec: bool = True,
    execution_mode: str | None = None,
    agent_profile: str | None = None,
) -> TaskResult:
    """Run a task starting from a specific agent."""

    runtime = _build_runtime_from_workspace(
        workspace,
        allow_write=allow_write,
        allow_exec=allow_exec,
        execution_mode=execution_mode,
        agent_profile=agent_profile,
    )
    task = UserTask(
        task_id=str(uuid.uuid4()),
        description=description,
        initial_agent_id=agent_id,
    )
    return runtime.orchestrator.run_task(task)


def _build_runtime_from_workspace(
    workspace: Path,
    *,
    allow_write: bool,
    allow_exec: bool,
    execution_mode: str | None,
    agent_profile: str | None,
) -> RuntimeContext:
    _LOGGER.info("Loading configuration from workspace %s", workspace)
    config = load_config(workspace)
    config = update_workspace_root(config, workspace.resolve())
    if execution_mode is not None:
        config = update_executor_mode(config, execution_mode)

    _LOGGER.info(
        "Starting task execution with execution mode '%s'.", config.executor.mode
    )
    return build_runtime(
        config,
        allow_write=allow_write,
        allow_exec=allow_exec,
        agent_profile=agent_profile,
    )


def build_runtime(
    config: AppConfig,
    *,
    allow_write: bool = True,
    allow_exec: bool = True,
    agent_profile: str | None = None,
    llm_client: LLMClient | None = None,
    executor: CodeExecutor | None = None,
) -> RuntimeContext:
    """Build runtime services for the orchestrator.

    Args:
        config: Application configuration.
        allow_write: Whether workspace writes are allowed.
        allow_exec: Whether command execution tools are enabled.
        agent_profile: Optional profile name to filter configured agents.
        llm_client: Optional pre-built LLM client (for testing).
        executor: Optional pre-built executor (for testing).

    Returns:
        RuntimeContext with initialized services.
    """

    workspace = WorkspaceManager(config.workspace_root, allow_write=allow_write)
    memory = MemoryService()
    retrieval = InMemoryRetrievalService()
    observability = create_observability_manager()
    llm = llm_client or create_llm_client(config.llm, observability=observability)
    executor_instance = executor or _build_executor(config)
    version_control = _build_version_control(config)
    tool_registry = build_default_tool_registry(
        workspace,
        executor_instance,
        version_control=version_control,
        allow_exec=allow_exec,
    )
    approval_policy = ApprovalPolicy(
        mode=config.approvals.mode,
        require_execution_approval=config.approvals.require_execution_approval,
        require_commit_approval=config.approvals.require_commit_approval,
        user_proxy_agent_id=config.approvals.user_proxy_agent_id,
    )
    orchestrator = Orchestrator(
        memory,
        tool_registry,
        approval_policy=approval_policy,
        observability=observability,
    )
    agent_configs = _select_agent_configs(config, agent_profile)
    agents = _build_agents(
        agent_configs,
        orchestrator,
        workspace,
        executor_instance,
        memory,
        retrieval,
        llm,
    )
    for agent in agents:
        orchestrator.register_agent(agent)
    _LOGGER.info("Runtime initialized with %s agents.", len(agents))
    return RuntimeContext(
        orchestrator=orchestrator,
        memory=memory,
        retrieval=retrieval,
        workspace=workspace,
        executor=executor_instance,
        llm_client=llm,
        version_control=version_control,
        observability=observability,
    )


def _select_agent_configs(config: AppConfig, profile: str | None) -> list[AgentConfig]:
    if profile is None:
        return list(config.agents)
    if profile not in config.agent_profiles:
        raise AppConfigError(f"Unknown agent profile: {profile}")
    selected_ids = set(config.agent_profiles[profile])
    return [agent for agent in config.agents if agent.agent_id in selected_ids]


def _build_executor(config: AppConfig) -> CodeExecutor:
    mode = config.executor.mode.lower()
    if mode == "local":
        return LocalExecutor()
    if mode == "docker":
        return DockerExecutor(
            config.workspace_root,
            config.executor.docker_image,
            docker_user=config.executor.docker_user,
        )
    raise AppConfigError(f"Unknown executor mode: {config.executor.mode}")


def _build_version_control(config: AppConfig) -> VersionControlService | None:
    if not config.version_control.enabled:
        return None
    provider = config.version_control.provider.lower()
    if provider == "git":
        return GitService(
            config.workspace_root,
            git_binary=config.version_control.git_binary,
        )
    raise AppConfigError(f"Unknown version control provider: {config.version_control.provider}")


def _build_agents(
    agent_configs: Iterable[AgentConfig],
    orchestrator: Orchestrator,
    workspace: WorkspaceManager,
    executor: CodeExecutor,
    memory: MemoryService,
    retrieval: RetrievalService,
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
            retrieval=retrieval,
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
    retrieval: RetrievalService,
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
            retrieval=retrieval,
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
            retrieval=retrieval,
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
            retrieval=retrieval,
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
            retrieval=retrieval,
            test_commands=agent_config.test_commands,
        )
    if agent_type == "user_proxy":
        return UserProxyAgent(
            agent_id=agent_config.agent_id,
            role=agent_config.role,
            llm_client=llm_client,
            orchestrator=orchestrator,
            workspace=workspace,
            executor=executor,
            memory=memory,
            retrieval=retrieval,
        )
    raise AppConfigError(f"Unknown agent type: {agent_config.type}")
