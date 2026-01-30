"""Version control tools exposed to agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from multiagent_dev.tools.base import Tool, ToolExecutionError, ToolResult
from multiagent_dev.version_control.base import VersionControlService


@dataclass
class VCSStatusTool(Tool):
    """Tool that retrieves version control status."""

    service: VersionControlService

    @property
    def name(self) -> str:
        return "vcs_status"

    @property
    def description(self) -> str:
        return "Return version control status information."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {}

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        status = self.service.status()
        return ToolResult(
            name=self.name,
            success=True,
            output={"entries": status.entries, "clean": status.clean},
        )


@dataclass
class VCSDiffTool(Tool):
    """Tool that retrieves version control diffs."""

    service: VersionControlService

    @property
    def name(self) -> str:
        return "vcs_diff"

    @property
    def description(self) -> str:
        return "Return version control diff for optional paths."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {"paths": "list[str] | null"}

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        paths = arguments.get("paths")
        if paths is not None and (
            not isinstance(paths, list) or not all(isinstance(item, str) for item in paths)
        ):
            raise ToolExecutionError("'paths' must be a list of strings or null")
        diff = self.service.diff(paths)
        return ToolResult(name=self.name, success=True, output={"diff": diff.diff})


@dataclass
class VCSCommitTool(Tool):
    """Tool that creates version control commits."""

    service: VersionControlService

    @property
    def name(self) -> str:
        return "vcs_commit"

    @property
    def description(self) -> str:
        return "Create a version control commit after approval."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "message": "string",
            "approved": "bool",
            "approver": "string | null",
            "stage_all": "bool | null",
        }

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        approved = arguments.get("approved")
        if approved is not True:
            return ToolResult(
                name=self.name,
                success=False,
                output=None,
                error="Commit requires explicit approval.",
            )
        message = arguments.get("message")
        if not isinstance(message, str) or not message.strip():
            raise ToolExecutionError("'message' must be a non-empty string")
        stage_all = arguments.get("stage_all")
        if stage_all is not None and not isinstance(stage_all, bool):
            raise ToolExecutionError("'stage_all' must be a bool or null")
        result = self.service.commit(message.strip(), stage_all=stage_all if stage_all is not None else True)
        output = {
            "commit_hash": result.commit_hash,
            "message": result.message,
            "approver": arguments.get("approver"),
        }
        return ToolResult(name=self.name, success=True, output=output)


@dataclass
class VCSBranchTool(Tool):
    """Tool that creates branches in version control."""

    service: VersionControlService

    @property
    def name(self) -> str:
        return "vcs_create_branch"

    @property
    def description(self) -> str:
        return "Create a new version control branch."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {"name": "string", "checkout": "bool | null"}

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        name = arguments.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ToolExecutionError("'name' must be a non-empty string")
        checkout = arguments.get("checkout")
        if checkout is not None and not isinstance(checkout, bool):
            raise ToolExecutionError("'checkout' must be a bool or null")
        result = self.service.create_branch(name.strip(), checkout=checkout if checkout is not None else True)
        return ToolResult(name=self.name, success=True, output={"branch_name": result.branch_name})
