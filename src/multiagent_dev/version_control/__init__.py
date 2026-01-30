"""Version control abstractions and implementations."""

from multiagent_dev.version_control.base import (
    VCSBranchResult,
    VCSCommitResult,
    VCSDiff,
    VCSStatus,
    VersionControlError,
    VersionControlService,
)
from multiagent_dev.version_control.git_service import GitService

__all__ = [
    "VCSBranchResult",
    "VCSCommitResult",
    "VCSDiff",
    "VCSStatus",
    "VersionControlError",
    "VersionControlService",
    "GitService",
]
