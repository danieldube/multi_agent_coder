"""Observability helpers for structured logging and metrics."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from multiagent_dev.util.logging import get_logger


@dataclass(frozen=True)
class LogEvent:
    """Structured log event payload.

    Attributes:
        event_type: Machine-readable event name.
        timestamp: Unix timestamp in seconds.
        payload: Structured data associated with the event.
        context: Optional shared context fields.
    """

    event_type: str
    timestamp: float
    payload: dict[str, Any]
    context: dict[str, Any] = field(default_factory=dict)


class EventLogger:
    """Logger that emits machine-readable JSON events."""

    def __init__(self, logger_name: str, context: dict[str, Any] | None = None) -> None:
        """Initialize the event logger.

        Args:
            logger_name: Logger name used for output.
            context: Optional shared context to attach to every event.
        """

        self._logger = get_logger(logger_name)
        self._context = context or {}

    def log(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        level: str = "INFO",
        context: dict[str, Any] | None = None,
    ) -> None:
        """Emit a structured log event.

        Args:
            event_type: Machine-readable event name.
            payload: Structured event data.
            level: Logging level string (default: INFO).
            context: Optional context overrides for this event.
        """

        event = LogEvent(
            event_type=event_type,
            timestamp=time.time(),
            payload=payload,
            context={**self._context, **(context or {})},
        )
        message = json.dumps(event.__dict__, sort_keys=True)
        self._logger.log(_normalize_level(level), message)


@dataclass
class MetricsCollector:
    """Collects simple counters, durations, and token metrics."""

    counters: dict[str, int] = field(default_factory=dict)
    durations: dict[str, list[float]] = field(default_factory=dict)
    tokens: dict[str, int] = field(default_factory=lambda: {"prompt": 0, "completion": 0, "total": 0})

    def increment(self, name: str, value: int = 1) -> None:
        """Increment a named counter.

        Args:
            name: Counter name.
            value: Increment amount.
        """

        self.counters[name] = self.counters.get(name, 0) + value

    def record_duration(self, name: str, duration_s: float) -> None:
        """Record a duration value for a named metric.

        Args:
            name: Duration metric name.
            duration_s: Duration in seconds.
        """

        self.durations.setdefault(name, []).append(duration_s)

    def record_tokens(
        self,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
    ) -> None:
        """Record token usage totals.

        Args:
            prompt_tokens: Number of prompt tokens.
            completion_tokens: Number of completion tokens.
            total_tokens: Total tokens.
        """

        self.tokens["prompt"] += prompt_tokens
        self.tokens["completion"] += completion_tokens
        self.tokens["total"] += total_tokens

    def snapshot(self) -> dict[str, Any]:
        """Return a snapshot of the collected metrics."""

        duration_summary: dict[str, dict[str, float]] = {}
        for name, values in self.durations.items():
            total = sum(values)
            count = len(values)
            avg = total / count if count else 0.0
            duration_summary[name] = {"count": float(count), "total_s": total, "avg_s": avg}
        return {
            "counters": dict(self.counters),
            "durations": duration_summary,
            "tokens": dict(self.tokens),
        }


@dataclass(frozen=True)
class ObservabilityManager:
    """Container for structured logging and metrics collection."""

    events: EventLogger
    metrics: MetricsCollector

    def log_event(self, event_type: str, payload: dict[str, Any]) -> None:
        """Log an event with the configured event logger."""

        self.events.log(event_type, payload)

    @contextmanager
    def track_duration(self, metric_name: str) -> Iterator[None]:
        """Track duration of a code block as a metric."""

        start = time.perf_counter()
        try:
            yield
        finally:
            self.metrics.record_duration(metric_name, time.perf_counter() - start)


def create_observability_manager() -> ObservabilityManager:
    """Create a default observability manager with standard loggers."""

    return ObservabilityManager(
        events=EventLogger("multiagent_dev.events"),
        metrics=MetricsCollector(),
    )


def _normalize_level(level: str) -> int:
    import logging

    normalized = level.strip().upper()
    return getattr(logging, normalized, logging.INFO)
