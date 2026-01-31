"""LLM client registry and factory utilities."""

from __future__ import annotations

from multiagent_dev.config import LLMConfig
from multiagent_dev.llm.base import LLMClient
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
    raise ValueError(f"Unknown LLM provider: {config.provider}")
