"""Base interfaces for LLM clients."""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMClientError(RuntimeError):
    """Base exception for LLM client failures."""


class LLMConfigurationError(LLMClientError):
    """Raised when LLM client configuration is invalid or incomplete."""


class LLMClient(ABC):
    """Abstract interface for LLM clients."""

    @abstractmethod
    def complete_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a chat completion response.

        Args:
            messages: Ordered list of chat messages.
            temperature: Sampling temperature for the model.
            max_tokens: Optional limit on generated tokens.

        Returns:
            The assistant response text.
        """
