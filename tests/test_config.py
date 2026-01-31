from pathlib import Path

import pytest

from multiagent_dev.config import AppConfig, load_config


def test_app_config_defaults() -> None:
    config = AppConfig()
    assert config.workspace_root == Path(".")
    assert config.project.languages == ["python"]
    assert config.executor.mode == "local"
    assert config.version_control.enabled is False
    assert config.agents
    assert {agent.agent_id for agent in config.agents} >= {"planner", "user_proxy"}


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
    assert any(agent.agent_id == "planner" for agent in config.agents)


def test_project_config_drives_test_commands(tmp_path: Path) -> None:
    config_path = tmp_path / "multiagent_dev.yaml"
    config_path.write_text(
        """
{
  "project": {
    "languages": ["cpp", "python"],
    "build_systems": ["cmake", "pip"],
    "test_commands_by_language": {
      "python": [["pytest", "-q", "--disable-warnings"]]
    }
  }
}
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.project.languages == ["cpp", "python"]
    assert config.project.build_systems == ["cmake", "pip"]
    assert config.test_commands == [
        ["ctest", "--output-on-failure"],
        ["pytest", "-q", "--disable-warnings"],
    ]


def test_load_config_from_non_json_yaml(tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    config_path = tmp_path / "multiagent_dev.yaml"
    config_path.write_text(
        """
project:
  languages:
    - python
  build_systems:
    - pip
llm:
  model: yaml-model
executor:
  mode: docker
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.project.languages == ["python"]
    assert config.project.build_systems == ["pip"]
    assert config.llm.model == "yaml-model"
    assert config.executor.mode == "docker"
