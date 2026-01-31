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

The generated `multiagent_dev.yaml` uses JSON-formatted YAML, and the loader also
supports full YAML syntax via PyYAML. You can override defaults such as the LLM model
or execution mode there.

### OpenAI-compatible LLM setup

Set the required environment variables:

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_MODEL="gpt-4o-mini"
```

### Microsoft Copilot LLM setup

If your organization only allows Microsoft Copilot, configure the framework to use
Copilot's Azure OpenAI-backed endpoint or your company's Copilot API gateway. The
project supports two common patterns depending on how your Copilot license is provisioned:

#### Option A: Copilot via Azure OpenAI (recommended when your org provisions Azure OpenAI)

1. Ask your Microsoft/Copilot admin for the **Azure OpenAI endpoint**, **deployment name**,
   **API version**, and **API key** (or set them in your CI secret store).
2. Update `multiagent_dev.yaml` (or `pyproject.toml`) to use the Azure provider:

   ```yaml
   llm:
     provider: azure
     api_key: "${AZURE_OPENAI_API_KEY}"
     base_url: "https://<your-resource-name>.openai.azure.com"
     azure_deployment: "<your-copilot-deployment-name>"
     api_version: "2024-02-15-preview"
     model: "gpt-4o-mini"
   ```

3. Export the environment variables used above:

   ```bash
   export AZURE_OPENAI_API_KEY="your-azure-openai-key"
   ```

> **Note:** When using the `azure` provider, the `azure_deployment` and `api_version` fields
> are required for requests to succeed.

#### Option B: Copilot via an OpenAI-compatible gateway (if your org provides one)

Some enterprises expose Copilot through an OpenAI-compatible API gateway. In that case:

1. Ask your admin for the **gateway base URL**, **API key**, and **model name**.
2. Configure the OpenAI-compatible provider:

   ```yaml
   llm:
     provider: openai-compatible
     api_key: "${COPILOT_API_KEY}"
     base_url: "https://<your-copilot-gateway>/v1"
     model: "<copilot-model-id>"
   ```

3. Export the environment variables used above:

   ```bash
   export COPILOT_API_KEY="your-copilot-gateway-key"
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
