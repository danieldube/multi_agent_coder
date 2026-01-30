"""Configuration models and loaders for multiagent-dev."""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

DEFAULT_TEST_COMMANDS: list[list[str]] = [["pytest", "-q"]]


@dataclass(frozen=True)
class AppConfig:
    """Top-level configuration for the application.

    Attributes:
        workspace_root: Root path of the workspace to operate on.
        llm: Configuration for the default LLM client.
        executor: Configuration for code execution.
        version_control: Configuration for version control integration.
        agents: List of configured agents.
        test_commands: Commands to use for testing when not overridden.
        approvals: Configuration for human approval checkpoints.
    """

    workspace_root: Path = Path(".")
    llm: LLMConfig = field(default_factory=lambda: LLMConfig())
    executor: ExecutorConfig = field(default_factory=lambda: ExecutorConfig())
    version_control: VersionControlConfig = field(
        default_factory=lambda: VersionControlConfig()
    )
    agents: list[AgentConfig] = field(default_factory=lambda: default_agent_configs())
    test_commands: list[list[str]] = field(default_factory=lambda: list(DEFAULT_TEST_COMMANDS))
    approvals: ApprovalConfig = field(default_factory=lambda: ApprovalConfig())


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for an LLM provider."""

    provider: str = "openai"
    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None
    timeout_s: float = 30.0
    max_retries: int = 2


@dataclass(frozen=True)
class ExecutorConfig:
    """Configuration for the code execution engine."""

    mode: str = "local"
    docker_image: str = "python:3.11-slim"
    timeout_s: int | None = None
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class VersionControlConfig:
    """Configuration for version control integration."""

    enabled: bool = False
    provider: str = "git"
    git_binary: str = "git"


@dataclass(frozen=True)
class ApprovalConfig:
    """Configuration for human-in-the-loop approvals."""

    mode: str = "autonomous"
    require_execution_approval: bool = False
    require_commit_approval: bool = True
    user_proxy_agent_id: str = "user_proxy"


@dataclass(frozen=True)
class AgentConfig:
    """Configuration for an individual agent."""

    agent_id: str
    role: str
    type: str
    test_commands: list[list[str]] | None = None


def default_agent_configs() -> list[AgentConfig]:
    """Return default agent configurations for the standard workflow."""

    return [
        AgentConfig(agent_id="user_proxy", role="User proxy agent", type="user_proxy"),
        AgentConfig(agent_id="planner", role="Planner agent", type="planner"),
        AgentConfig(agent_id="coder", role="Coding agent", type="coder"),
        AgentConfig(agent_id="tester", role="Tester agent", type="tester"),
        AgentConfig(agent_id="reviewer", role="Reviewer agent", type="reviewer"),
    ]


def load_config(path: Path | None = None) -> AppConfig:
    """Load application configuration from disk.

    Args:
        path: Optional path to a configuration file or workspace directory.

    Returns:
        Parsed AppConfig with defaults applied when no config exists.
    """

    config_path = _resolve_config_path(path)
    if config_path is None:
        return AppConfig()

    if config_path.suffix in {".yaml", ".yml"}:
        raw_data = _load_yaml(config_path)
    elif config_path.name == "pyproject.toml" or config_path.suffix == ".toml":
        raw_data = _load_toml(config_path)
    else:
        raise ValueError(f"Unsupported config file type: {config_path}")

    return _parse_app_config(raw_data, base_path=config_path.parent)


def config_to_dict(config: AppConfig) -> dict[str, Any]:
    """Serialize an AppConfig into a JSON-compatible dictionary."""

    return {
        "workspace_root": str(config.workspace_root),
        "llm": {
            "provider": config.llm.provider,
            "api_key": config.llm.api_key,
            "base_url": config.llm.base_url,
            "model": config.llm.model,
            "timeout_s": config.llm.timeout_s,
            "max_retries": config.llm.max_retries,
        },
        "executor": {
            "mode": config.executor.mode,
            "docker_image": config.executor.docker_image,
            "timeout_s": config.executor.timeout_s,
            "env": dict(config.executor.env),
        },
        "version_control": {
            "enabled": config.version_control.enabled,
            "provider": config.version_control.provider,
            "git_binary": config.version_control.git_binary,
        },
        "agents": [
            {
                "id": agent.agent_id,
                "role": agent.role,
                "type": agent.type,
                "test_commands": agent.test_commands,
            }
            for agent in config.agents
        ],
        "test_commands": config.test_commands,
        "approvals": {
            "mode": config.approvals.mode,
            "require_execution_approval": config.approvals.require_execution_approval,
            "require_commit_approval": config.approvals.require_commit_approval,
            "user_proxy_agent_id": config.approvals.user_proxy_agent_id,
        },
    }


def update_workspace_root(config: AppConfig, workspace_root: Path) -> AppConfig:
    """Return a config copy with an updated workspace root."""

    return replace(config, workspace_root=workspace_root)


def update_executor_mode(config: AppConfig, mode: str) -> AppConfig:
    """Return a config copy with an updated executor mode."""

    return replace(config, executor=replace(config.executor, mode=mode))


def _resolve_config_path(path: Path | None) -> Path | None:
    candidate_paths: list[Path] = []
    if path is None:
        candidate_paths.append(Path("multiagent_dev.yaml"))
        candidate_paths.append(Path("multiagent_dev.yml"))
        candidate_paths.append(Path("pyproject.toml"))
    elif path.is_dir():
        candidate_paths.append(path / "multiagent_dev.yaml")
        candidate_paths.append(path / "multiagent_dev.yml")
        candidate_paths.append(path / "pyproject.toml")
    else:
        candidate_paths.append(path)

    for candidate in candidate_paths:
        if candidate.exists():
            return candidate
    return None


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    if path.name == "pyproject.toml":
        tool_config = data.get("tool", {}).get("multiagent_dev", {})
        if not isinstance(tool_config, dict):
            raise ValueError("tool.multiagent_dev must be a mapping.")
        return tool_config
    if not isinstance(data, dict):
        raise ValueError("TOML configuration must be a mapping.")
    return data


def _load_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None
    if data is not None:
        if not isinstance(data, dict):
            raise ValueError("YAML configuration must be a mapping.")
        return data
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to parse non-JSON YAML configuration files."
        ) from exc
    parsed = yaml.safe_load(text)
    if not isinstance(parsed, dict):
        raise ValueError("YAML configuration must be a mapping.")
    return parsed


def _parse_app_config(raw_data: dict[str, Any], base_path: Path) -> AppConfig:
    llm_config = _parse_llm_config(raw_data.get("llm", {}))
    executor_config = _parse_executor_config(raw_data.get("executor", {}))
    version_control_config = _parse_version_control_config(
        raw_data.get("version_control", {})
    )
    approvals_config = _parse_approval_config(raw_data.get("approvals", {}))
    test_commands = _parse_test_commands(raw_data.get("test_commands", None))
    agents = _parse_agent_configs(raw_data.get("agents", None), test_commands)

    workspace_root = Path(raw_data.get("workspace_root", ".")) if raw_data else Path(".")
    if not workspace_root.is_absolute():
        workspace_root = (base_path / workspace_root).resolve()

    return AppConfig(
        workspace_root=workspace_root,
        llm=llm_config,
        executor=executor_config,
        version_control=version_control_config,
        approvals=approvals_config,
        agents=agents,
        test_commands=test_commands,
    )


def _parse_llm_config(raw: Any) -> LLMConfig:
    if not isinstance(raw, dict):
        return LLMConfig()
    return LLMConfig(
        provider=str(raw.get("provider", "openai")),
        api_key=_optional_str(raw.get("api_key")),
        base_url=_optional_str(raw.get("base_url")),
        model=_optional_str(raw.get("model")),
        timeout_s=float(raw.get("timeout_s", 30.0)),
        max_retries=int(raw.get("max_retries", 2)),
    )


def _parse_executor_config(raw: Any) -> ExecutorConfig:
    if not isinstance(raw, dict):
        return ExecutorConfig()
    env = raw.get("env", {})
    env_map: dict[str, str] = {}
    if isinstance(env, dict):
        env_map = {str(key): str(value) for key, value in env.items()}
    return ExecutorConfig(
        mode=str(raw.get("mode", "local")),
        docker_image=str(raw.get("docker_image", "python:3.11-slim")),
        timeout_s=_optional_int(raw.get("timeout_s")),
        env=env_map,
    )


def _parse_version_control_config(raw: Any) -> VersionControlConfig:
    if not isinstance(raw, dict):
        return VersionControlConfig()
    return VersionControlConfig(
        enabled=bool(raw.get("enabled", False)),
        provider=str(raw.get("provider", "git")),
        git_binary=str(raw.get("git_binary", "git")),
    )


def _parse_approval_config(raw: Any) -> ApprovalConfig:
    if not isinstance(raw, dict):
        return ApprovalConfig()
    return ApprovalConfig(
        mode=str(raw.get("mode", "autonomous")),
        require_execution_approval=bool(raw.get("require_execution_approval", False)),
        require_commit_approval=bool(raw.get("require_commit_approval", True)),
        user_proxy_agent_id=str(raw.get("user_proxy_agent_id", "user_proxy")),
    )


def _parse_agent_configs(
    raw: Any,
    test_commands: list[list[str]],
) -> list[AgentConfig]:
    if not raw:
        return default_agent_configs()
    if not isinstance(raw, list):
        raise ValueError("Agents configuration must be a list of agent definitions.")
    agents: list[AgentConfig] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Agent definition must be a mapping.")
        agent_id = str(item.get("id") or item.get("agent_id") or "").strip()
        agent_type = str(item.get("type") or "").strip()
        if not agent_id or not agent_type:
            raise ValueError("Agent definitions require 'id' and 'type'.")
        role = str(item.get("role", f"{agent_type.title()} agent"))
        agent_test_commands = _parse_test_commands(item.get("test_commands", None))
        if not agent_test_commands:
            agent_test_commands = test_commands
        agents.append(
            AgentConfig(
                agent_id=agent_id,
                role=role,
                type=agent_type,
                test_commands=agent_test_commands,
            )
        )
    return agents


def _parse_test_commands(raw: Any) -> list[list[str]]:
    if raw is None:
        return list(DEFAULT_TEST_COMMANDS)
    if not isinstance(raw, list):
        raise ValueError("test_commands must be a list of command lists.")
    commands: list[list[str]] = []
    for entry in raw:
        if not isinstance(entry, list) or not entry:
            raise ValueError("Each test command must be a non-empty list.")
        commands.append([str(item) for item in entry])
    return commands


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
