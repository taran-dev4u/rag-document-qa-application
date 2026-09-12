from rag_engine.chunking import DocumentChunk
from rag_engine.embeddings import HashEmbeddingProvider
from rag_engine.hybrid_search import HybridRetriever, reciprocal_rank_fusion
from rag_engine.lexical_retriever import BM25Retriever
from rag_engine.vector_store import InMemoryVectorStore


def test_reciprocal_rank_fusion():
    c1 = DocumentChunk("c1", "d1", "Alpha text", 0, 10)
    c2 = DocumentChunk("c2", "d2", "Beta text", 0, 9)
    c3 = DocumentChunk("c3", "d3", "Gamma text", 0, 10)

    # List 1: c1 rank 1, c2 rank 2
    # List 2: c2 rank 1, c3 rank 2
    list1 = [(c1, 0.9), (c2, 0.7)]
    list2 = [(c2, 0.95), (c3, 0.6)]

    fused = reciprocal_rank_fusion([list1, list2], k=60)
    assert len(fused) == 3
    # c2 appeared in both lists, so its combined RRF score is highest
    assert fused[0][0].chunk_id == "c2"


def test_hybrid_retriever_pipeline():
    vstore = InMemoryVectorStore()
    lretriever = BM25Retriever()
    embedder = HashEmbeddingProvider(dimension=64)
    hybrid = HybridRetriever(vstore, lretriever, embedder)

    c1 = DocumentChunk("c1", "d1", "Docker containerization and Kubernetes orchestration", 0, 52)
    c2 = DocumentChunk("c2", "d2", "PostgreSQL relational database schema design", 0, 44)
    c3 = DocumentChunk("c3", "d3", "Kubernetes cluster ingress and service networking", 0, 49)

    indexed = hybrid.index_chunks([c1, c2, c3])
    assert indexed == 3

    results = hybrid.search("Kubernetes networking orchestration", top_k=2)
    assert len(results) == 2
    result_ids = [chunk.chunk_id for chunk, _ in results]
    assert "c1" in result_ids or "c3" in result_ids
