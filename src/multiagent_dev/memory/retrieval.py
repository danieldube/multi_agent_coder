"""Retrieval abstractions for indexing and querying project context."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import re
from typing import Iterable


@dataclass(frozen=True)
class CodeChunk:
    """Represents a chunk of code stored for retrieval.

    Attributes:
        chunk_id: Unique identifier for the chunk.
        path: File path associated with the chunk.
        content: Raw content of the chunk.
        summary: Optional summary of the chunk.
    """

    chunk_id: str
    path: str
    content: str
    summary: str | None = None


@dataclass(frozen=True)
class RetrievalResult:
    """Represents a retrieval result and its relevance score."""

    chunk: CodeChunk
    score: float


class RetrievalService(ABC):
    """Abstract interface for retrieval backends."""

    @abstractmethod
    def index_text(self, path: str, text: str) -> None:
        """Index a file's text for future retrieval.

        Args:
            path: Path to the file being indexed.
            text: Full text content of the file.
        """

    @abstractmethod
    def get_file_summary(self, path: str) -> str | None:
        """Return a stored summary for the given path."""

    @abstractmethod
    def query(self, query: str, limit: int = 5) -> list[RetrievalResult]:
        """Retrieve the most relevant chunks for a query."""


class InMemoryRetrievalService(RetrievalService):
    """Naive in-memory retrieval implementation using token overlap."""

    def __init__(self, *, max_chunk_lines: int = 40) -> None:
        """Initialize the in-memory retrieval store.

        Args:
            max_chunk_lines: Maximum number of lines per chunk.
        """

        self._max_chunk_lines = max_chunk_lines
        self._file_summaries: dict[str, str] = {}
        self._chunks: list[CodeChunk] = []

    def index_text(self, path: str, text: str) -> None:
        """Index text by creating a summary and chunks.

        Args:
            path: File path to index.
            text: File content to store.
        """

        self._file_summaries[path] = _summarize_text(text)
        self._chunks = [chunk for chunk in self._chunks if chunk.path != path]
        for index, chunk_text in enumerate(_chunk_text(text, self._max_chunk_lines)):
            chunk_id = f"{path}:{index}"
            self._chunks.append(
                CodeChunk(chunk_id=chunk_id, path=path, content=chunk_text)
            )

    def get_file_summary(self, path: str) -> str | None:
        """Return a stored summary for the given path."""

        return self._file_summaries.get(path)

    def query(self, query: str, limit: int = 5) -> list[RetrievalResult]:
        """Retrieve chunks by naive token overlap scoring."""

        query_terms = _tokenize(query)
        scored = [
            RetrievalResult(chunk=chunk, score=_score_chunk(chunk, query_terms))
            for chunk in self._chunks
        ]
        scored = [result for result in scored if result.score > 0]
        scored.sort(key=lambda result: result.score, reverse=True)
        return scored[:limit]


def _tokenize(text: str) -> set[str]:
    return {match.lower() for match in re.findall(r"[A-Za-z0-9_]+", text)}


def _summarize_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    summary_lines = lines[:3]
    return " ".join(summary_lines)


def _chunk_text(text: str, max_lines: int) -> Iterable[str]:
    lines = text.splitlines()
    if not lines:
        return []
    chunks: list[str] = []
    for start in range(0, len(lines), max_lines):
        chunk_lines = lines[start : start + max_lines]
        chunks.append("\n".join(chunk_lines))
    return chunks


def _score_chunk(chunk: CodeChunk, query_terms: set[str]) -> float:
    chunk_terms = _tokenize(chunk.content)
    overlap = query_terms & chunk_terms
    if not overlap:
        return 0.0
    return float(len(overlap))
