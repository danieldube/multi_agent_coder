"""GitHub Copilot client implementation."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import requests  # type: ignore[import-untyped]

from multiagent_dev.llm.base import LLMClientError, LLMConfigurationError
from multiagent_dev.llm.generic_client import GenericOpenAICompatibleClient
from multiagent_dev.util.logging import get_logger
from multiagent_dev.util.observability import ObservabilityManager

DEFAULT_COPILOT_CLIENT_ID = "Iv1.6d910195a5d91f44"
DEFAULT_COPILOT_BASE_URL = "https://api.githubcopilot.com"
DEFAULT_COPILOT_TOKEN_URL = "https://api.githubcopilot.com/copilot_internal/v2/token"
DEFAULT_GITHUB_DEVICE_TOKEN_URL = "https://github.com/login/oauth/access_token"
DEFAULT_GITHUB_API_VERSION = "2022-11-28"


@dataclass
class CopilotToken:
    """Represents a GitHub Copilot access token."""

    value: str
    expires_at: datetime


class GitHubCopilotClient(GenericOpenAICompatibleClient):
    """LLM client for GitHub Copilot's OpenAI-compatible API."""

    def __init__(
        self,
        *,
        device_key: str | None = None,
        github_token: str | None = None,
        client_id: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_s: float = 30.0,
        max_retries: int = 2,
        session: requests.Session | None = None,
        observability: ObservabilityManager | None = None,
    ) -> None:
        """Initialize the GitHub Copilot client.

        Args:
            device_key: GitHub device key (device_code) for OAuth device flow.
            github_token: Existing GitHub OAuth token (skips device flow if set).
            client_id: OAuth client ID for the device flow.
            base_url: Base URL for GitHub Copilot API.
            model: Model identifier to use for chat completions.
            timeout_s: Request timeout in seconds.
            max_retries: Number of retries for transient failures.
            session: Optional requests session for testing or reuse.
        """

        resolved_device_key = device_key or os.getenv("COPILOT_DEVICE_KEY")
        resolved_github_token = github_token or os.getenv("COPILOT_GITHUB_TOKEN")
        resolved_client_id = client_id or os.getenv("COPILOT_CLIENT_ID") or DEFAULT_COPILOT_CLIENT_ID
        resolved_base_url = (
            base_url or os.getenv("COPILOT_BASE_URL") or DEFAULT_COPILOT_BASE_URL
        )
        resolved_model = model or os.getenv("COPILOT_MODEL") or "gpt-4o-mini"

        if not resolved_device_key and not resolved_github_token:
            raise LLMConfigurationError(
                "COPILOT_DEVICE_KEY or COPILOT_GITHUB_TOKEN is required for GitHubCopilotClient."
            )

        self._device_key = resolved_device_key
        self._github_token = resolved_github_token
        self._client_id = resolved_client_id
        self._token_url = os.getenv("COPILOT_TOKEN_URL") or DEFAULT_COPILOT_TOKEN_URL
        self._github_device_token_url = (
            os.getenv("COPILOT_GITHUB_DEVICE_TOKEN_URL") or DEFAULT_GITHUB_DEVICE_TOKEN_URL
        )
        self._github_api_version = (
            os.getenv("COPILOT_GITHUB_API_VERSION") or DEFAULT_GITHUB_API_VERSION
        )
        self._copilot_token: CopilotToken | None = None
        self._logger = get_logger(self.__class__.__name__)
        self._session = session or requests.Session()

        super().__init__(
            api_key="copilot",
            base_url=resolved_base_url,
            model=resolved_model,
            timeout_s=timeout_s,
            max_retries=max_retries,
            session=self._session,
            observability=observability,
        )

    def _build_headers(self) -> dict[str, str]:
        token = self._get_copilot_token()
        return {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }

    def _get_copilot_token(self) -> str:
        if self._copilot_token and not self._token_expired(self._copilot_token):
            return self._copilot_token.value
        github_token = self._resolve_github_token()
        self._copilot_token = self._fetch_copilot_token(github_token)
        return self._copilot_token.value

    def _resolve_github_token(self) -> str:
        if self._github_token:
            return self._github_token
        if not self._device_key:
            raise LLMConfigurationError("COPILOT_DEVICE_KEY is required to fetch a GitHub token.")
        self._github_token = self._exchange_device_key(self._device_key)
        return self._github_token

    def _exchange_device_key(self, device_key: str) -> str:
        payload = {
            "client_id": self._client_id,
            "device_code": device_key,
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
        }
        headers = {"Accept": "application/json"}
        deadline = time.monotonic() + 60
        while True:
            response = self._session.post(
                self._github_device_token_url,
                data=payload,
                headers=headers,
                timeout=self._config.timeout_s,
            )
            data = self._parse_json_response(response)
            access_token = data.get("access_token")
            if isinstance(access_token, str) and access_token:
                return access_token
            error = data.get("error")
            if error != "authorization_pending":
                description = data.get("error_description")
                message = description or error or "unknown error"
                raise LLMClientError(
                    "GitHub device key exchange failed: "
                    f"{message}"
                )
            interval = data.get("interval")
            wait_s = float(interval) if isinstance(interval, (int, float)) else 5.0
            if time.monotonic() + wait_s > deadline:
                raise LLMClientError("GitHub device key exchange timed out.")
            time.sleep(wait_s)

    def _fetch_copilot_token(self, github_token: str) -> CopilotToken:
        headers = {
            "Authorization": f"token {github_token}",
            "Accept": "application/json",
            "X-GitHub-Api-Version": self._github_api_version,
        }
        response = self._session.get(
            self._token_url,
            headers=headers,
            timeout=self._config.timeout_s,
        )
        data = self._parse_json_response(response)
        token = data.get("token")
        if not isinstance(token, str) or not token:
            raise LLMClientError("GitHub Copilot token response missing token value.")
        expires_at = self._parse_expires_at(data.get("expires_at"))
        return CopilotToken(value=token, expires_at=expires_at)

    @staticmethod
    def _parse_json_response(response: requests.Response) -> dict[str, Any]:
        if response.status_code >= 400:
            raise LLMClientError(
                "GitHub Copilot request failed with status "
                f"{response.status_code}: {response.text}"
            )
        data = response.json()
        if not isinstance(data, dict):
            raise LLMClientError("Unexpected response format from GitHub Copilot API.")
        return data

    @staticmethod
    def _parse_expires_at(value: Any) -> datetime:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=UTC)
        if isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=UTC)
                return parsed
            except ValueError:
                pass
        return datetime.now(tz=UTC) + timedelta(minutes=10)

    @staticmethod
    def _token_expired(token: CopilotToken) -> bool:
        return datetime.now(tz=UTC) + timedelta(seconds=30) >= token.expires_at
