"""User proxy agent for human-in-the-loop approvals."""

from __future__ import annotations

from dataclasses import dataclass

from multiagent_dev.agents.base import Agent, AgentMessage


@dataclass(frozen=True)
class UserDecision:
    """Represents a user decision returned by the proxy agent.

    Attributes:
        approved: Whether the action is approved.
        approver: Identifier for the user.
        notes: Optional notes from the user.
    """

    approved: bool
    approver: str
    notes: str | None = None


class UserProxyAgent(Agent):
    """Agent representing a human user for approvals."""

    def handle_message(self, message: AgentMessage) -> list[AgentMessage]:
        """Return a default approval decision based on message metadata.

        Args:
            message: Incoming approval request message.

        Returns:
            A response message containing approval metadata.
        """

        decision = self._parse_decision(message)
        metadata = {
            "approval_request_id": message.metadata.get("approval_request_id"),
            "approved": decision.approved,
            "approver": decision.approver,
            "notes": decision.notes,
        }
        content = "Approved" if decision.approved else "Rejected"
        return [
            AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                content=content,
                metadata=metadata,
            )
        ]

    def _parse_decision(self, message: AgentMessage) -> UserDecision:
        """Parse a decision from incoming metadata.

        Args:
            message: Approval request message.

        Returns:
            UserDecision based on provided metadata or default approval.
        """

        metadata = message.metadata
        approved = metadata.get("approved")
        if not isinstance(approved, bool):
            approved = True
        approver = metadata.get("approver")
        if not isinstance(approver, str) or not approver:
            approver = "user"
        notes = metadata.get("notes")
        if not isinstance(notes, str):
            notes = None
        return UserDecision(approved=approved, approver=approver, notes=notes)
