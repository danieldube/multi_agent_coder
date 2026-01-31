from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from multiagent_dev.llm.base import LLMClientError, LLMConfigurationError
from multiagent_dev.llm.copilot_client import (
    DEFAULT_COPILOT_BASE_URL,
    DEFAULT_COPILOT_CLIENT_ID,
    DEFAULT_COPILOT_TOKEN_URL,
    CopilotToken,
    GitHubCopilotClient,
)


class _MockResponse:
    def __init__(self, status_code: int, payload: dict[str, Any]) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = "mock response"

    def json(self) -> dict[str, Any]:
        return self._payload


def test_copilot_requires_device_key_or_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COPILOT_DEVICE_KEY", raising=False)
    monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
    with pytest.raises(LLMConfigurationError):
        GitHubCopilotClient()


def test_copilot_exchanges_device_key_then_fetches_token() -> None:
    device_token_response = _MockResponse(
        200,
        {"access_token": "gh-token", "token_type": "bearer"},
    )
    copilot_token_response = _MockResponse(
        200,
        {"token": "copilot-token", "expires_at": datetime.now(tz=UTC).isoformat()},
    )
    chat_response = _MockResponse(
        200,
        {"choices": [{"message": {"content": "hello copilot"}}]},
    )
    responses = [device_token_response, copilot_token_response, chat_response]

    client = GitHubCopilotClient(device_key="device-code")

    def mock_post(*_args: Any, **_kwargs: Any) -> _MockResponse:
        return responses.pop(0)

    def mock_get(*_args: Any, **_kwargs: Any) -> _MockResponse:
        return responses.pop(0)

    client._session.post = mock_post
    client._session.get = mock_get

    result = client.complete_chat([{"role": "user", "content": "hi"}])
    assert result == "hello copilot"


def test_copilot_reuses_cached_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "gh-token")
    client = GitHubCopilotClient()
    expires_at = datetime.now(tz=UTC) + timedelta(minutes=5)
    client._copilot_token = CopilotToken(value="cached-token", expires_at=expires_at)

    def mock_post(*_args: Any, **_kwargs: Any) -> _MockResponse:
        raise AssertionError("Device flow should not be invoked.")

    client._session.post = mock_post

    def mock_get(*_args: Any, **_kwargs: Any) -> _MockResponse:
        raise AssertionError("Copilot token fetch should not be invoked.")

    client._session.get = mock_get

    def mock_chat_post(*_args: Any, **_kwargs: Any) -> _MockResponse:
        return _MockResponse(
            200,
            {"choices": [{"message": {"content": "cached response"}}]},
        )

    client._session.post = mock_chat_post
    result = client.complete_chat([{"role": "user", "content": "hi"}])
    assert result == "cached response"


def test_copilot_raises_on_token_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "gh-token")
    client = GitHubCopilotClient()

    def mock_get(*_args: Any, **_kwargs: Any) -> _MockResponse:
        return _MockResponse(500, {"error": "bad"})

    client._session.get = mock_get

    with pytest.raises(LLMClientError):
        client.complete_chat([{"role": "user", "content": "hi"}])


def test_copilot_defaults_are_exposed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "gh-token")
    client = GitHubCopilotClient()

    assert client._config.base_url == DEFAULT_COPILOT_BASE_URL
    assert client._client_id == DEFAULT_COPILOT_CLIENT_ID
    assert client._token_url == DEFAULT_COPILOT_TOKEN_URL
