"""Tester agent implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from multiagent_dev.agents.base import Agent, AgentMessage
from multiagent_dev.execution.base import ExecutionResult

if TYPE_CHECKING:
    from multiagent_dev.execution.base import CodeExecutor
    from multiagent_dev.llm.base import LLMClient
    from multiagent_dev.memory.memory import MemoryService
    from multiagent_dev.memory.retrieval import RetrievalService
    from multiagent_dev.orchestrator import Orchestrator
    from multiagent_dev.workspace.manager import WorkspaceManager


@dataclass(frozen=True)
class TestRunSummary:
    """Summary of executed test commands.

    Attributes:
        results: List of execution results.
        succeeded: True if all commands returned exit code 0.
    """

    results: list[ExecutionResult]
    succeeded: bool


class TesterAgent(Agent):
    """Agent responsible for running tests via the executor."""

    __test__ = False

    def __init__(
        self,
        agent_id: str,
        role: str,
        llm_client: LLMClient,
        orchestrator: Orchestrator,
        workspace: WorkspaceManager,
        executor: CodeExecutor,
        memory: MemoryService,
        retrieval: RetrievalService,
        test_commands: list[list[str]] | None = None,
    ) -> None:
        """Initialize the tester agent.

        Args:
            agent_id: Unique identifier for the agent.
            role: Human-readable role description.
            llm_client: Client used to query the language model.
            orchestrator: Orchestrator coordinating message flow.
            workspace: Workspace manager for file operations.
            executor: Code execution engine.
            memory: Memory service for storing conversation data.
            retrieval: Retrieval service for indexed project context.
            test_commands: Optional list of commands to execute.
        """

        super().__init__(
            agent_id=agent_id,
            role=role,
            llm_client=llm_client,
            orchestrator=orchestrator,
            workspace=workspace,
            executor=executor,
            memory=memory,
            retrieval=retrieval,
        )
        self._test_commands = test_commands or [["pytest", "-q"]]

    def handle_message(self, message: AgentMessage) -> list[AgentMessage]:
        """Execute tests and report results.

        Args:
            message: Incoming message to trigger testing.

        Returns:
            Messages summarizing test results.
        """

        summary = self._run_tests()
        summary_text = self._format_summary(summary)
        metadata = {
            "succeeded": summary.succeeded,
            "results": [self._serialize_result(result) for result in summary.results],
        }
        self.log_event(
            "agent.tests_completed",
            {
                "task_id": message.metadata.get("task_id", "default"),
                "succeeded": summary.succeeded,
            },
        )
        return [
            AgentMessage(
                sender=self.agent_id,
                recipient="reviewer",
                content=summary_text,
                metadata=metadata,
            ),
            AgentMessage(
                sender=self.agent_id,
                recipient="planner",
                content=summary_text,
                metadata=metadata,
            ),
        ]

    def _run_tests(self) -> TestRunSummary:
        """Run configured test commands.

        Returns:
            Summary of executed commands.
        """

        results: list[ExecutionResult] = []
        for command in self._test_commands:
            tool_result = self.use_tool("run_command", {"command": command})
            execution = tool_result.output
            if isinstance(execution, ExecutionResult):
                results.append(execution)
            else:
                results.append(
                    ExecutionResult(
                        command=command,
                        stdout="",
                        stderr=tool_result.error or "Tool execution failed",
                        exit_code=1,
                        duration_s=0.0,
                    )
                )
        succeeded = all(result.exit_code == 0 for result in results)
        return TestRunSummary(results=results, succeeded=succeeded)

    def _format_summary(self, summary: TestRunSummary) -> str:
        """Format execution results into human-readable summary.

        Args:
            summary: Test run summary.

        Returns:
            Summary text.
        """

        lines = ["Test results:"]
        for result in summary.results:
            status = "PASS" if result.exit_code == 0 else "FAIL"
            lines.append(f"- {' '.join(result.command)}: {status}")
        lines.append(f"Overall status: {'PASS' if summary.succeeded else 'FAIL'}")
        return "\n".join(lines)

    def _serialize_result(self, result: ExecutionResult) -> dict[str, object]:
        """Serialize an execution result into simple data.

        Args:
            result: ExecutionResult to serialize.

        Returns:
            Dict representation of the execution result.
        """

        return {
            "command": list(result.command),
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.exit_code,
            "duration_s": result.duration_s,
        }
