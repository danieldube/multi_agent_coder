# Multiagent Dev

Multi-agent software development framework for orchestrating LLM-powered agents.

## Features

- Role-based agents (planner, coder, tester, reviewer)
- Pluggable LLM provider layer
- Workspace sandboxing for safe file access
- Local or Docker-based command execution
- CLI entry point for end-to-end workflows

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install .
```

## Configuration

Create a configuration file in your repository:

```bash
multiagent-dev init /path/to/your/repo
```

The generated `multiagent_dev.yaml` is JSON-compatible. You can override defaults such
as the LLM model or execution mode there.

### OpenAI-compatible LLM setup

Set the required environment variables:

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_MODEL="gpt-4o-mini"
```

## Usage

Run a task against a workspace:

```bash
multiagent-dev run-task "Add a new feature" --workspace /path/to/your/repo
```

Run with Docker-based execution:

```bash
multiagent-dev run-task "Refactor module" --workspace /path/to/your/repo --exec-mode docker
```

Enable debug logging for more detail:

```bash
multiagent-dev --log-level DEBUG run-task "Analyze failing tests"
```

## Example workflow

1. Initialize the config in a repository.
2. Describe a task using `run-task`.
3. Planner breaks down the work, Coding agent applies updates, Tester runs tests, and
   Reviewer validates the changes.

This project is under active development.
