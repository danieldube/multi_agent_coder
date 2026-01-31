from __future__ import annotations

from multiagent_dev.evaluation.harness import EvaluationHarness, EvaluationTask
from multiagent_dev.orchestrator import TaskResult, UserTask


def test_evaluation_harness_runs_tasks() -> None:
    def runner(task: UserTask, max_steps: int) -> TaskResult:
        return TaskResult(
            task_id=task.task_id,
            completed=True,
            messages_processed=max_steps,
            history=[],
        )

    harness = EvaluationHarness(runner)
    tasks = [EvaluationTask(task_id="task-1", description="test", max_steps=3)]

    summary = harness.run(tasks)

    assert summary.passed == 1
    assert summary.failed == 0
    assert summary.results[0].messages_processed == 3
