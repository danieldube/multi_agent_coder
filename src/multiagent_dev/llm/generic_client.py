"""Generic OpenAI-compatible client implementations."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, cast

import requests  # type: ignore[import-untyped]

from multiagent_dev.llm.base import LLMClient, LLMClientError, LLMConfigurationError
from multiagent_dev.util.logging import get_logger
from multiagent_dev.util.observability import ObservabilityManager


@dataclass(frozen=True)
class GenericOpenAICompatibleConfig:
    """Configuration for OpenAI-compatible clients."""

    api_key: str
    base_url: str
    model: str
    timeout_s: float
    max_retries: int


class GenericOpenAICompatibleClient(LLMClient):
    """LLM client for OpenAI-compatible HTTP APIs."""

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str | None,
        model: str | None,
        timeout_s: float = 30.0,
        max_retries: int = 2,
        session: requests.Session | None = None,
        observability: ObservabilityManager | None = None,
    ) -> None:
        """Initialize the OpenAI-compatible client.

        Args:
            api_key: API key for authentication.
            base_url: Base URL for the OpenAI-compatible endpoint.
            model: Model name to use for completions.
            timeout_s: Request timeout in seconds.
            max_retries: Number of retries for transient failures.
            session: Optional requests session for testing or reuse.
        """

        if not api_key:
            raise LLMConfigurationError("api_key is required for OpenAI-compatible clients.")
        if not base_url:
            raise LLMConfigurationError("base_url is required for OpenAI-compatible clients.")
        if not model:
            raise LLMConfigurationError("model is required for OpenAI-compatible clients.")

        self._config = GenericOpenAICompatibleConfig(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            model=model,
            timeout_s=timeout_s,
            max_retries=max_retries,
        )
        self._session = session or requests.Session()
        self._logger = get_logger(self.__class__.__name__)
        self._observability = observability

    def complete_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a chat completion response."""

        payload = self._build_payload(messages, temperature=temperature, max_tokens=max_tokens)

        self._logger.debug("Requesting chat completion with model '%s'.", self._config.model)
        start = time.perf_counter()
        if self._observability:
            self._observability.log_event(
                "llm.chat_requested",
                {
                    "model": self._config.model,
                    "message_count": len(messages),
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )
        try:
            response_data = self._post(self._chat_completions_path(), payload)
            usage = self._extract_usage(response_data)
            duration = time.perf_counter() - start
            if self._observability:
                self._observability.metrics.record_duration("llm.chat_duration", duration)
                if usage:
                    self._observability.metrics.record_tokens(**usage)
                self._observability.log_event(
                    "llm.chat_completed",
                    {
                        "model": self._config.model,
                        "duration_s": duration,
                        "usage": usage or {},
                    },
                )
        except Exception as exc:
            if self._observability:
                self._observability.log_event(
                    "llm.chat_failed",
                    {"model": self._config.model, "error": str(exc)},
                )
            raise
        try:
            content = response_data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMClientError("Unexpected response format from OpenAI API.") from exc
        if not isinstance(content, str):
            raise LLMClientError("Unexpected response format from OpenAI API.")
        return content

    def _build_payload(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        return payload

    def _chat_completions_path(self) -> str:
        return "/chat/completions"

    def _build_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._config.api_key}"}

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._config.base_url}{path}"
        headers = self._build_headers()
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
                    if self._should_retry(response.status_code) and attempt < (
                        self._config.max_retries
                    ):
                        time.sleep(0.5 * (attempt + 1))
                        continue
                    raise LLMClientError(
                        "OpenAI API request failed with status "
                        f"{response.status_code}: {response.text}"
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

    @staticmethod
    def _extract_usage(response_data: dict[str, Any]) -> dict[str, int] | None:
        usage = response_data.get("usage")
        if not isinstance(usage, dict):
            return None
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
        if not all(isinstance(value, int) for value in (prompt_tokens, completion_tokens, total_tokens)):
            return None
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }


class AzureOpenAIClient(GenericOpenAICompatibleClient):
    """LLM client for Azure OpenAI-compatible APIs."""

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str | None,
        azure_deployment: str | None,
        api_version: str | None,
        model: str | None = None,
        timeout_s: float = 30.0,
        max_retries: int = 2,
        session: requests.Session | None = None,
        observability: ObservabilityManager | None = None,
    ) -> None:
        if not azure_deployment:
            raise LLMConfigurationError("azure_deployment is required for AzureOpenAIClient.")
        if not api_version:
            raise LLMConfigurationError("api_version is required for AzureOpenAIClient.")
        resolved_model = model or azure_deployment
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            model=resolved_model,
            timeout_s=timeout_s,
            max_retries=max_retries,
            session=session,
            observability=observability,
        )
        self._azure_deployment = azure_deployment
        self._api_version = api_version

    def _build_payload(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        return payload

    def _chat_completions_path(self) -> str:
        return (
            f"/openai/deployments/{self._azure_deployment}/chat/completions"
            f"?api-version={self._api_version}"
        )

    def _build_headers(self) -> dict[str, str]:
        return {"api-key": self._config.api_key}
