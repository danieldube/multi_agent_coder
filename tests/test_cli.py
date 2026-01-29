from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from multiagent_dev.cli.main import app


def test_cli_init_creates_config_file(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["init", str(tmp_path)])

    assert result.exit_code == 0
    config_path = tmp_path / "multiagent_dev.yaml"
    assert config_path.exists()
    data = json.loads(config_path.read_text(encoding="utf-8"))
    assert data["workspace_root"] == str(tmp_path.resolve())
