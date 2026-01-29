from __future__ import annotations

from typing import Any

import pytest

from multiagent_dev.config import LLMConfig
from multiagent_dev.llm.base import LLMConfigurationError
from multiagent_dev.llm.openai_client import OpenAIClient
from multiagent_dev.llm.registry import create_llm_client


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

    client._session.post = mock_post  # type: ignore[method-assign]

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

    client._session.post = mock_post  # type: ignore[method-assign]

    result = client.complete_chat([{"role": "user", "content": "hi"}])
    assert result == "retry"


def test_create_llm_client_uses_openai_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config = LLMConfig(provider="openai", model="test-model")
    client = create_llm_client(config)
    assert isinstance(client, OpenAIClient)
