"""Logging utilities for the multiagent-dev framework."""

from __future__ import annotations

import logging
from typing import Final

DEFAULT_LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_LEVELS: Final[dict[str, int]] = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}


def configure_logging(level: str = "INFO", fmt: str | None = None) -> None:
    """Configure application logging.

    Args:
        level: Logging level name (e.g., "INFO", "DEBUG").
        fmt: Optional logging format string. Defaults to a standard structured format.
    """

    logging.basicConfig(
        level=_normalize_level(level),
        format=fmt or DEFAULT_LOG_FORMAT,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module or component."""

    return logging.getLogger(name)


def _normalize_level(level: str) -> int:
    normalized = level.strip().upper()
    return _LEVELS.get(normalized, logging.INFO)
