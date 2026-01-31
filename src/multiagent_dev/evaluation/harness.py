"""Evaluation harness for running predefined tasks."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Iterable

from multiagent_dev.orchestrator import TaskResult, UserTask
from multiagent_dev.util.observability import EventLogger, MetricsCollector


@dataclass(frozen=True)
class EvaluationTask:
    """Represents a single evaluation task.

    Attributes:
        task_id: Identifier for the task.
        description: Task description to execute.
        expected_completed: Expected completion outcome.
        max_steps: Maximum steps for the orchestrator.
    """

    task_id: str
    description: str
    expected_completed: bool = True
    max_steps: int = 100


@dataclass(frozen=True)
class EvaluationResult:
    """Represents the outcome of an evaluation task."""

    task_id: str
    completed: bool
    expected_completed: bool
    success: bool
    messages_processed: int
    duration_s: float
    error: str | None = None


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregated summary of evaluation results."""

    results: list[EvaluationResult]
    passed: int
    failed: int
    duration_s: float


class EvaluationHarness:
    """Execute a suite of evaluation tasks using a task runner."""

    def __init__(
        self,
        task_runner: Callable[[UserTask, int], TaskResult],
        *,
        event_logger: EventLogger | None = None,
        metrics: MetricsCollector | None = None,
    ) -> None:
        """Initialize the evaluation harness.

        Args:
            task_runner: Callable that executes a UserTask and returns a TaskResult.
            event_logger: Optional event logger for structured evaluation events.
            metrics: Optional metrics collector for evaluation statistics.
        """

        self._task_runner = task_runner
        self._event_logger = event_logger or EventLogger("multiagent_dev.evaluation")
        self._metrics = metrics or MetricsCollector()

    def run(self, tasks: Iterable[EvaluationTask]) -> EvaluationSummary:
        """Run all evaluation tasks and return a summary.

        Args:
            tasks: Iterable of evaluation tasks.

        Returns:
            Aggregated evaluation summary.
        """

        start = time.perf_counter()
        results: list[EvaluationResult] = []
        passed = 0
        failed = 0

        for task in tasks:
            result = self._run_task(task)
            results.append(result)
            if result.success:
                passed += 1
                self._metrics.increment("evaluation.passed")
            else:
                failed += 1
                self._metrics.increment("evaluation.failed")

        duration = time.perf_counter() - start
        self._metrics.record_duration("evaluation.total_duration", duration)
        summary = EvaluationSummary(
            results=results,
            passed=passed,
            failed=failed,
            duration_s=duration,
        )
        self._event_logger.log(
            "evaluation.completed",
            {
                "passed": passed,
                "failed": failed,
                "duration_s": duration,
            },
        )
        return summary

    def metrics_snapshot(self) -> dict[str, object]:
        """Return a snapshot of evaluation metrics."""

        return self._metrics.snapshot()

    def _run_task(self, task: EvaluationTask) -> EvaluationResult:
        self._metrics.increment("evaluation.tasks")
        start = time.perf_counter()
        try:
            result = self._task_runner(
                UserTask(task_id=task.task_id, description=task.description),
                task.max_steps,
            )
            duration = time.perf_counter() - start
            success = result.completed == task.expected_completed
            evaluation_result = EvaluationResult(
                task_id=task.task_id,
                completed=result.completed,
                expected_completed=task.expected_completed,
                success=success,
                messages_processed=result.messages_processed,
                duration_s=duration,
            )
            self._event_logger.log(
                "evaluation.task_completed",
                {
                    "task_id": task.task_id,
                    "completed": result.completed,
                    "expected_completed": task.expected_completed,
                    "success": success,
                    "messages_processed": result.messages_processed,
                    "duration_s": duration,
                },
            )
            return evaluation_result
        except Exception as exc:  # pragma: no cover - exercised in tests
            duration = time.perf_counter() - start
            self._event_logger.log(
                "evaluation.task_failed",
                {
                    "task_id": task.task_id,
                    "error": str(exc),
                    "duration_s": duration,
                },
            )
            return EvaluationResult(
                task_id=task.task_id,
                completed=False,
                expected_completed=task.expected_completed,
                success=False,
                messages_processed=0,
                duration_s=duration,
                error=str(exc),
            )
