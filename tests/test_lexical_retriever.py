from rag_engine.chunking import DocumentChunk
from rag_engine.lexical_retriever import BM25Retriever


def test_bm25_retriever_ranking():
    retriever = BM25Retriever()
    c1 = DocumentChunk("c1", "d1", "PostgreSQL database indexing and query tuning", 0, 45)
    c2 = DocumentChunk("c2", "d2", "Python web development with FastAPI and AsyncIO", 0, 48)
    c3 = DocumentChunk("c3", "d3", "PostgreSQL connection pooling with PgBouncer", 0, 44)

    retriever.add_chunks([c1, c2, c3])
    assert retriever.total_chunks == 3

    results = retriever.search("PostgreSQL pooling", top_k=2)
    assert len(results) >= 1
    # c3 contains both PostgreSQL and pooling
    assert results[0][0].chunk_id == "c3"


def test_bm25_empty_query():
    retriever = BM25Retriever()
    c1 = DocumentChunk("c1", "d1", "Sample text", 0, 11)
    retriever.add_chunks([c1])

    assert retriever.search("", top_k=5) == []
    assert retriever.search("   ", top_k=5) == []


def test_bm25_clear():
    retriever = BM25Retriever()
    c1 = DocumentChunk("c1", "d1", "Sample text", 0, 11)
    retriever.add_chunks([c1])
    assert retriever.total_chunks == 1

    retriever.clear()
    assert retriever.total_chunks == 0
    assert retriever.search("sample", top_k=5) == []
