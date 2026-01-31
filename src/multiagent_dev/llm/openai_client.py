"""OpenAI-compatible client implementation."""

from __future__ import annotations

import os
from multiagent_dev.llm.base import LLMConfigurationError
from multiagent_dev.llm.generic_client import GenericOpenAICompatibleClient
from multiagent_dev.util.observability import ObservabilityManager


class OpenAIClient(GenericOpenAICompatibleClient):
    """LLM client for OpenAI-compatible HTTP APIs."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_s: float = 30.0,
        max_retries: int = 2,
        session: requests.Session | None = None,
        observability: ObservabilityManager | None = None,
    ) -> None:
        """Initialize the OpenAI client.

        Args:
            api_key: API key for authentication.
            base_url: Base URL for the OpenAI-compatible endpoint.
            model: Model name to use for completions.
            timeout_s: Request timeout in seconds.
            max_retries: Number of retries for transient failures.
            session: Optional requests session for testing or reuse.
        """

        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_api_key:
            raise LLMConfigurationError("OPENAI_API_KEY is required for OpenAIClient.")

        resolved_base_url = base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        resolved_model = model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"

        super().__init__(
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            model=resolved_model,
            timeout_s=timeout_s,
            max_retries=max_retries,
            session=session,
            observability=observability,
        )
