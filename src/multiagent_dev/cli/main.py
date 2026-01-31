"""CLI entrypoints for multiagent-dev."""

from __future__ import annotations

from pathlib import Path

import typer

from multiagent_dev.app import (
    AppConfigError,
    initialize_config,
    run_agent,
    run_plan,
    run_task,
)
from multiagent_dev.util.logging import configure_logging

app = typer.Typer(help="Multi-agent development workflow CLI.")


@app.callback()
def main(
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Logging level (e.g., DEBUG, INFO, WARNING).",
    ),
) -> None:
    """Configure CLI-level options."""

    configure_logging(log_level)


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


@app.command("plan")
def plan_command(
    description: str = typer.Argument(..., help="Task description to plan."),
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
    """Generate a plan for a task without executing it."""

    try:
        result, plan_summary = run_plan(
            description=description,
            workspace=workspace,
            allow_write=allow_write,
            execution_mode=execution_mode,
        )
    except Exception as exc:
        typer.echo(f"Error: {exc}")
        raise typer.Exit(code=1) from exc

    if plan_summary:
        typer.echo("Plan summary:")
        typer.echo(plan_summary)
    status = "completed" if result.completed else "incomplete"
    typer.echo(f"Task {result.task_id} {status} after {result.messages_processed} steps.")


@app.command("exec")
def exec_command(
    agent_id: str = typer.Argument(..., help="Agent identifier to start with."),
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
    """Execute a task starting from a specific agent."""

    try:
        result = run_agent(
            description=description,
            workspace=workspace,
            agent_id=agent_id,
            allow_write=allow_write,
            execution_mode=execution_mode,
        )
    except Exception as exc:
        typer.echo(f"Error: {exc}")
        raise typer.Exit(code=1) from exc

    status = "completed" if result.completed else "incomplete"
    typer.echo(f"Task {result.task_id} {status} after {result.messages_processed} steps.")
