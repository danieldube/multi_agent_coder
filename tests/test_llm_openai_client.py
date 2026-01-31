from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from multiagent_dev.config import LLMConfig
from multiagent_dev.llm.base import LLMConfigurationError
from multiagent_dev.llm.generic_client import AzureOpenAIClient, GenericOpenAICompatibleClient
from multiagent_dev.llm.copilot_client import GitHubCopilotClient
from multiagent_dev.llm.openai_client import OpenAIClient
from multiagent_dev.llm.registry import create_llm_client
from multiagent_dev.util.observability import MetricsCollector, ObservabilityManager, EventLogger


class _MockResponse:
    def __init__(self, status_code: int, payload: dict[str, Any]) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = "mock response"

    def json(self) -> dict[str, Any]:
        return self._payload


def test_openai_client_reads_env_and_returns_content(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")

    client = OpenAIClient()

    def mock_post(*_args: Any, **_kwargs: Any) -> _MockResponse:
        return _MockResponse(
            200,
            {"choices": [{"message": {"content": "hello"}}]},
        )

    client._session.post = mock_post

    result = client.complete_chat([{"role": "user", "content": "hi"}])
    assert result == "hello"


def test_openai_client_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMConfigurationError):
        OpenAIClient()


def test_openai_client_retries_on_transient_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = OpenAIClient(max_retries=1)
    responses = [
        _MockResponse(500, {}),
        _MockResponse(200, {"choices": [{"message": {"content": "retry"}}]}),
    ]

    def mock_post(*_args: Any, **_kwargs: Any) -> _MockResponse:
        return responses.pop(0)

    client._session.post = mock_post

    result = client.complete_chat([{"role": "user", "content": "hi"}])
    assert result == "retry"


def test_create_llm_client_uses_openai_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config = LLMConfig(provider="openai", model="test-model")
    client = create_llm_client(config)
    assert isinstance(client, OpenAIClient)


def test_create_llm_client_uses_openai_compatible_config() -> None:
    config = LLMConfig(
        provider="openai-compatible",
        api_key="test-key",
        base_url="https://example.test/v1",
        model="test-model",
    )
    client = create_llm_client(config)
    assert isinstance(client, GenericOpenAICompatibleClient)


def test_create_llm_client_uses_azure_config() -> None:
    config = LLMConfig(
        provider="azure",
        api_key="test-key",
        base_url="https://azure.example.test",
        azure_deployment="test-deployment",
        api_version="2024-01-01",
    )
    client = create_llm_client(config)
    assert isinstance(client, AzureOpenAIClient)


def test_create_llm_client_uses_copilot_config() -> None:
    config = LLMConfig(
        provider="copilot",
        copilot_device_key="device-code",
    )
    client = create_llm_client(config)
    assert isinstance(client, GitHubCopilotClient)


def test_azure_client_requires_deployment() -> None:
    with pytest.raises(LLMConfigurationError):
        AzureOpenAIClient(
            api_key="test-key",
            base_url="https://azure.example.test",
            azure_deployment=None,
            api_version="2024-01-01",
        )


def test_openai_compatible_requires_base_url() -> None:
    with pytest.raises(LLMConfigurationError):
        GenericOpenAICompatibleClient(
            api_key="test-key",
            base_url=None,
            model="test-model",
        )


def test_openai_client_records_usage_and_logs(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    metrics = MetricsCollector()
    event_logger = EventLogger("test.llm.events")
    observability = ObservabilityManager(events=event_logger, metrics=metrics)

    client = OpenAIClient(observability=observability)

    def mock_post(*_args: Any, **_kwargs: Any) -> _MockResponse:
        return _MockResponse(
            200,
            {
                "choices": [{"message": {"content": "hello"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            },
        )

    client._session.post = mock_post

    caplog.set_level(logging.INFO, logger="test.llm.events")
    result = client.complete_chat([{"role": "user", "content": "hi"}])
    assert result == "hello"

    snapshot = metrics.snapshot()
    assert snapshot["tokens"]["total"] == 3

    logged = [record.message for record in caplog.records if "llm.chat_completed" in record.message]
    assert logged, "Expected llm.chat_completed event"
    payload = json.loads(logged[-1])
    assert payload["event_type"] == "llm.chat_completed"
