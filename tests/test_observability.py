from __future__ import annotations

import json
import logging

from multiagent_dev.util.observability import EventLogger, MetricsCollector


def test_metrics_collector_snapshot() -> None:
    metrics = MetricsCollector()
    metrics.increment("calls", 2)
    metrics.record_duration("latency", 1.5)
    metrics.record_tokens(prompt_tokens=3, completion_tokens=2, total_tokens=5)

    snapshot = metrics.snapshot()

    assert snapshot["counters"]["calls"] == 2
    assert snapshot["durations"]["latency"]["count"] == 1.0
    assert snapshot["tokens"]["total"] == 5


def test_event_logger_emits_json(caplog) -> None:
    logger = EventLogger("test.events")
    caplog.set_level(logging.INFO, logger="test.events")

    logger.log("sample.event", {"value": 42})

    assert caplog.records
    payload = json.loads(caplog.records[-1].message)
    assert payload["event_type"] == "sample.event"
    assert payload["payload"]["value"] == 42
