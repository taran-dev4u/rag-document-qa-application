import numpy as np
import pytest
from rag_engine.chunking import DocumentChunk
from rag_engine.vector_store import InMemoryVectorStore


def test_vector_store_add_and_search():
    store = InMemoryVectorStore()
    chunk1 = DocumentChunk("c1", "d1", "Python programming language", 0, 30, {"topic": "code"})
    chunk2 = DocumentChunk("c2", "d2", "Database query optimization", 0, 30, {"topic": "db"})

    vec1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    vec2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    matrix = np.vstack([vec1, vec2])

    store.add_chunks([chunk1, chunk2], matrix)
    assert store.total_chunks == 2

    # Query matching vec1
    q_vec = np.array([0.9, 0.1, 0.0], dtype=np.float32)
    results = store.search(q_vec, top_k=2)

    assert len(results) == 2
    assert results[0][0].chunk_id == "c1"
    assert results[0][1] > results[1][1]


def test_vector_store_metadata_filter():
    store = InMemoryVectorStore()
    chunk1 = DocumentChunk("c1", "d1", "Alpha", 0, 5, {"env": "prod"})
    chunk2 = DocumentChunk("c2", "d2", "Beta", 0, 4, {"env": "dev"})
    matrix = np.eye(2, dtype=np.float32)

    store.add_chunks([chunk1, chunk2], matrix)

    results = store.search(
        np.array([1.0, 1.0], dtype=np.float32),
        top_k=5,
        metadata_filter=lambda m: m.get("env") == "dev",
    )
    assert len(results) == 1
    assert results[0][0].chunk_id == "c2"


def test_vector_store_delete_document():
    store = InMemoryVectorStore()
    c1 = DocumentChunk("c1", "doc_A", "text 1", 0, 6)
    c2 = DocumentChunk("c2", "doc_B", "text 2", 0, 6)
    store.add_chunks([c1, c2], np.eye(2, dtype=np.float32))

    deleted = store.delete_document("doc_A")
    assert deleted == 1
    assert store.total_chunks == 1
    assert store._chunks[0].chunk_id == "c2"


def test_vector_store_mismatched_lengths():
    store = InMemoryVectorStore()
    c1 = DocumentChunk("c1", "doc_A", "text", 0, 4)
    with pytest.raises(ValueError):
        store.add_chunks([c1], np.zeros((2, 3), dtype=np.float32))
