"""LLM client registry and factory utilities."""

from __future__ import annotations

from multiagent_dev.config import LLMConfig
from multiagent_dev.llm.base import LLMClient
from multiagent_dev.llm.copilot_client import GitHubCopilotClient
from multiagent_dev.llm.generic_client import AzureOpenAIClient, GenericOpenAICompatibleClient
from multiagent_dev.llm.openai_client import OpenAIClient
from multiagent_dev.util.observability import ObservabilityManager


def create_llm_client(
    config: LLMConfig, *, observability: ObservabilityManager | None = None
) -> LLMClient:
    """Create an LLM client instance from configuration.

    Args:
        config: LLM configuration settings.

    Returns:
        An initialized LLM client.

    Raises:
        ValueError: If the provider is unknown.
    """

    provider = config.provider.lower()
    if provider == "openai":
        return OpenAIClient(
            api_key=config.api_key,
            base_url=config.base_url,
            model=config.model,
            timeout_s=config.timeout_s,
            max_retries=config.max_retries,
            observability=observability,
        )
    if provider in {"openai-compatible", "openai_compatible"}:
        return GenericOpenAICompatibleClient(
            api_key=config.api_key,
            base_url=config.base_url,
            model=config.model,
            timeout_s=config.timeout_s,
            max_retries=config.max_retries,
            observability=observability,
        )
    if provider == "azure":
        return AzureOpenAIClient(
            api_key=config.api_key,
            base_url=config.base_url,
            model=config.model,
            azure_deployment=config.azure_deployment,
            api_version=config.api_version,
            timeout_s=config.timeout_s,
            max_retries=config.max_retries,
            observability=observability,
        )
    if provider in {"github-copilot", "github_copilot", "copilot"}:
        return GitHubCopilotClient(
            device_key=config.copilot_device_key,
            github_token=config.copilot_github_token,
            client_id=config.copilot_client_id,
            base_url=config.copilot_base_url,
            model=config.model,
            timeout_s=config.timeout_s,
            max_retries=config.max_retries,
            observability=observability,
        )
    raise ValueError(f"Unknown LLM provider: {config.provider}")
