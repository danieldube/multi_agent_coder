"""Abstract interfaces for version control services."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class VersionControlError(RuntimeError):
    """Raised when version control operations fail."""


@dataclass(frozen=True)
class VCSStatus:
    """Represents the status of the working tree."""

    entries: list[str]
    clean: bool


@dataclass(frozen=True)
class VCSDiff:
    """Represents a diff response from version control."""

    diff: str


@dataclass(frozen=True)
class VCSCommitResult:
    """Represents a version control commit."""

    commit_hash: str
    message: str


@dataclass(frozen=True)
class VCSBranchResult:
    """Represents the creation of a branch."""

    branch_name: str


class VersionControlService(ABC):
    """Abstract interface for version control operations."""

    @abstractmethod
    def status(self) -> VCSStatus:
        """Return the status of the workspace."""

    @abstractmethod
    def diff(self, paths: list[str] | None = None) -> VCSDiff:
        """Return a diff for the workspace or specific paths."""

    @abstractmethod
    def commit(self, message: str, *, stage_all: bool = True) -> VCSCommitResult:
        """Create a commit for the current workspace state."""

    @abstractmethod
    def create_branch(self, name: str, *, checkout: bool = True) -> VCSBranchResult:
        """Create a branch in the version control system."""
