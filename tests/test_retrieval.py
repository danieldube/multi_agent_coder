from multiagent_dev.memory.retrieval import InMemoryRetrievalService


def test_retrieval_indexes_and_queries() -> None:
    retrieval = InMemoryRetrievalService(max_chunk_lines=2)
    retrieval.index_text("src/example.py", "def add(a, b):\n    return a + b\n")

    summary = retrieval.get_file_summary("src/example.py")
    assert summary == "def add(a, b): return a + b"

    results = retrieval.query("add function", limit=3)
    assert results
    assert results[0].chunk.path == "src/example.py"
