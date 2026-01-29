"""Planner agent implementation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from multiagent_dev.agents.base import Agent, AgentMessage


@dataclass(frozen=True)
class PlanResult:
    """Represents a parsed plan produced by the planner agent.

    Attributes:
        steps: Ordered list of plan steps.
        raw_text: Raw LLM response for the plan.
    """

    steps: list[str]
    raw_text: str


class PlannerAgent(Agent):
    """Agent responsible for decomposing user tasks into steps."""

    def handle_message(self, message: AgentMessage) -> list[AgentMessage]:
        """Generate a plan and notify other agents.

        Args:
            message: Incoming task description message.

        Returns:
            Messages to the coding, testing, and reviewing agents.
        """

        response = self._llm_client.complete_chat(
            self._build_prompt(message.content)
        )
        plan = self._parse_plan(response)
        self._memory.save_note(self._plan_key(message), plan.raw_text)

        plan_text = self._format_steps(plan.steps)
        return [
            AgentMessage(
                sender=self.agent_id,
                recipient="coder",
                content=f"Implement the following plan:\n{plan_text}",
                metadata={"steps": list(plan.steps)},
            ),
            AgentMessage(
                sender=self.agent_id,
                recipient="tester",
                content="Prepare to run tests after implementation.",
                metadata={"steps": list(plan.steps)},
            ),
            AgentMessage(
                sender=self.agent_id,
                recipient="reviewer",
                content="Review changes after implementation.",
                metadata={"steps": list(plan.steps)},
            ),
        ]

    def _build_prompt(self, task_description: str) -> list[dict[str, str]]:
        """Build the prompt used to request a plan from the LLM.

        Args:
            task_description: User-provided task description.

        Returns:
            Chat messages for the LLM.
        """

        system_prompt = (
            "You are a planning agent. Break the task into clear, actionable steps."
        )
        user_prompt = (
            "Create an ordered list of steps to implement the following task:\n"
            f"{task_description}"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _parse_plan(self, response: str) -> PlanResult:
        """Parse the plan response into structured steps.

        Args:
            response: Raw response from the LLM.

        Returns:
            Parsed plan result with steps.
        """

        steps = [self._clean_step(line) for line in response.splitlines()]
        steps = [step for step in steps if step]
        if not steps:
            steps = [response.strip()]
        return PlanResult(steps=steps, raw_text=response.strip())

    def _clean_step(self, line: str) -> str:
        """Normalize a single plan step.

        Args:
            line: Raw line from the LLM output.

        Returns:
            Cleaned step text.
        """

        stripped = line.strip().lstrip("-*")
        if stripped[:2].isdigit() and stripped[2:3] == ".":
            stripped = stripped[3:]
        return stripped.strip()

    def _format_steps(self, steps: Iterable[str]) -> str:
        """Format steps as a bullet list.

        Args:
            steps: Iterable of step strings.

        Returns:
            Bullet list as a string.
        """

        return "\n".join(f"- {step}" for step in steps)

    def _plan_key(self, message: AgentMessage) -> str:
        """Build a key for storing the plan in memory.

        Args:
            message: Incoming message that may contain task metadata.

        Returns:
            A memory key for the plan.
        """

        task_id = message.metadata.get("task_id", "default")
        return f"plan:{task_id}"
