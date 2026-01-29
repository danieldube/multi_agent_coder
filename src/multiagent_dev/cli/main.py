"""CLI entrypoints for multiagent-dev."""

from __future__ import annotations

from pathlib import Path

import typer

from multiagent_dev.app import AppConfigError, initialize_config, run_task

app = typer.Typer(help="Multi-agent development workflow CLI.")


@app.command()
def init(workspace: Path = typer.Argument(Path("."))) -> None:
    """Initialize configuration for a repository workspace."""

    try:
        config_path = initialize_config(workspace)
    except AppConfigError as exc:
        typer.echo(f"Error: {exc}")
        raise typer.Exit(code=1) from exc
    typer.echo(f"Created configuration at {config_path}")


@app.command("run-task")
def run_task_command(
    description: str = typer.Argument(..., help="Task description to execute."),
    workspace: Path = typer.Option(
        Path("."),
        "--workspace",
        "-w",
        help="Path to the workspace root.",
    ),
    allow_write: bool = True,
    execution_mode: str = typer.Option(
        "local",
        "--exec-mode",
        help="Execution mode: local|docker",
    ),
) -> None:
    """Run a multi-agent workflow for the provided task description."""

    try:
        result = run_task(
            description=description,
            workspace=workspace,
            allow_write=allow_write,
            execution_mode=execution_mode,
        )
    except Exception as exc:
        typer.echo(f"Error: {exc}")
        raise typer.Exit(code=1) from exc

    status = "completed" if result.completed else "incomplete"
    typer.echo(f"Task {result.task_id} {status} after {result.messages_processed} steps.")
