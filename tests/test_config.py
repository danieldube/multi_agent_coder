from pathlib import Path

from multiagent_dev.config import AppConfig, load_config


def test_app_config_defaults() -> None:
    config = AppConfig()
    assert config.workspace_root == Path(".")
    assert config.executor.mode == "local"
    assert config.version_control.enabled is False
    assert config.agents


def test_load_config_from_pyproject(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
[tool.multiagent_dev]
workspace_root = "workspace"

[tool.multiagent_dev.llm]
provider = "openai"
model = "unit-test-model"

[tool.multiagent_dev.executor]
mode = "docker"
docker_image = "python:3.11"

[[tool.multiagent_dev.agents]]
id = "planner"
type = "planner"
role = "Planner agent"

[tool.multiagent_dev.version_control]
enabled = true
provider = "git"
git_binary = "git"
""",
        encoding="utf-8",
    )

    config = load_config(pyproject)

    assert config.workspace_root == (tmp_path / "workspace").resolve()
    assert config.llm.model == "unit-test-model"
    assert config.executor.mode == "docker"
    assert config.version_control.enabled is True
    assert config.agents[0].agent_id == "planner"
