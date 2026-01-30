"""Approval request and decision models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ApprovalRequest:
    """Represents a request for human approval.

    Attributes:
        action: Short label describing the action requiring approval.
        description: Human-readable description of the request.
        metadata: Additional structured information about the request.
    """

    action: str
    description: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ApprovalDecision:
    """Represents the decision for an approval request.

    Attributes:
        approved: True if the action is approved.
        approver: Identifier for the approving user.
        notes: Optional notes captured during approval.
    """

    approved: bool
    approver: str
    notes: str | None = None


@dataclass(frozen=True)
class ApprovalPolicy:
    """Represents configuration for approval checkpoints.

    Attributes:
        mode: Either "autonomous" or "approval-required".
        require_execution_approval: Whether command execution requires approval.
        require_commit_approval: Whether commits require approval.
        user_proxy_agent_id: Agent ID to use for user approvals.
    """

    mode: str = "autonomous"
    require_execution_approval: bool = False
    require_commit_approval: bool = True
    user_proxy_agent_id: str = "user_proxy"

    def requires_approval(self, tool_name: str) -> bool:
        """Return True if the tool requires approval under this policy."""

        if self.mode != "approval-required":
            return False
        if tool_name == "run_command":
            return self.require_execution_approval
        if tool_name == "vcs_commit":
            return self.require_commit_approval
        return False
