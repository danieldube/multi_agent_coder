"""OpenAI-compatible client implementation."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, cast

import requests  # type: ignore[import-untyped]

from multiagent_dev.llm.base import LLMClient, LLMClientError, LLMConfigurationError


@dataclass(frozen=True)
class OpenAIClientConfig:
    """Configuration for the OpenAI-compatible client."""

    api_key: str
    base_url: str
    model: str
    timeout_s: float
    max_retries: int


class OpenAIClient(LLMClient):
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

        self._config = OpenAIClientConfig(
            api_key=resolved_api_key,
            base_url=resolved_base_url.rstrip("/"),
            model=resolved_model,
            timeout_s=timeout_s,
            max_retries=max_retries,
        )
        self._session = session or requests.Session()

    def complete_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a chat completion response."""

        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        response_data = self._post("/chat/completions", payload)
        try:
            content = response_data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMClientError("Unexpected response format from OpenAI API.") from exc
        if not isinstance(content, str):
            raise LLMClientError("Unexpected response format from OpenAI API.")
        return content

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._config.base_url}{path}"
        headers = {"Authorization": f"Bearer {self._config.api_key}"}
        last_error: Exception | None = None

        for attempt in range(self._config.max_retries + 1):
            try:
                response = self._session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=self._config.timeout_s,
                )
                if response.status_code >= 400:
                    if self._should_retry(response.status_code) and attempt < self._config.max_retries:
                        time.sleep(0.5 * (attempt + 1))
                        continue
                    raise LLMClientError(
                        f"OpenAI API request failed with status {response.status_code}: {response.text}"
                    )
                data = response.json()
                if not isinstance(data, dict):
                    raise LLMClientError("Unexpected response format from OpenAI API.")
                return cast(dict[str, Any], data)
            except requests.RequestException as exc:
                last_error = exc
                if attempt < self._config.max_retries:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise LLMClientError("OpenAI API request failed.") from exc

        raise LLMClientError("OpenAI API request failed.") from last_error

    @staticmethod
    def _should_retry(status_code: int) -> bool:
        return status_code in {429, 500, 502, 503, 504}
