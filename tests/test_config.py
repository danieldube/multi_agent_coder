from pathlib import Path

from multiagent_dev.config import AppConfig


def test_app_config_defaults() -> None:
    config = AppConfig()
    assert config.workspace_root == Path(".")
