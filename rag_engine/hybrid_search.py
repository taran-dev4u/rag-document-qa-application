from typing import Dict, List, Optional, Tuple

from .chunking import DocumentChunk
from .embeddings import DenseEmbeddingProvider
from .lexical_retriever import BM25Retriever
from .vector_store import InMemoryVectorStore


def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[DocumentChunk, float]]],
    weights: Optional[List[float]] = None,
    k: int = 60,
) -> List[Tuple[DocumentChunk, float]]:
    """Merges multiple ranked lists using weighted Reciprocal Rank Fusion (RRF).

    Formula: RRF_score(d) = sum_m ( w_m / (k + rank_m(d)) )
    """
    if not ranked_lists:
        return []

    num_lists = len(ranked_lists)
    w = weights if weights and len(weights) == num_lists else [1.0] * num_lists

    scores: Dict[str, float] = {}
    chunk_map: Dict[str, DocumentChunk] = {}

    for list_idx, rank_list in enumerate(ranked_lists):
        weight = w[list_idx]
        for rank, (chunk, _) in enumerate(rank_list, start=1):
            cid = chunk.chunk_id
            chunk_map[cid] = chunk
            rrf_val = weight / (k + rank)
            scores[cid] = scores.get(cid, 0.0) + rrf_val

    sorted_chunks = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [(chunk_map[cid], score) for cid, score in sorted_chunks]


class HybridRetriever:
    """Hybrid retrieval orchestrator combining dense vector search and BM25 lexical ranking."""

    def __init__(
        self,
        vector_store: InMemoryVectorStore,
        lexical_retriever: BM25Retriever,
        embedding_provider: DenseEmbeddingProvider,
    ):
        self.vector_store = vector_store
        self.lexical_retriever = lexical_retriever
        self.embedding_provider = embedding_provider

    def index_chunks(self, chunks: List[DocumentChunk]) -> int:
        if not chunks:
            return 0

        # Dense indexing
        texts = [c.text for c in chunks]
        embeddings = self.embedding_provider.embed_documents(texts)
        self.vector_store.add_chunks(chunks, embeddings)

        # Lexical indexing
        self.lexical_retriever.add_chunks(chunks)
        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        dense_weight: float = 1.0,
        lexical_weight: float = 1.0,
        rrf_k: int = 60,
    ) -> List[Tuple[DocumentChunk, float]]:
        if not query.strip():
            return []

        # Retrieve larger candidate set for fusion
        fetch_k = max(top_k * 3, 20)

        # 1. Dense retrieval
        q_vec = self.embedding_provider.embed_query(query)
        dense_results = self.vector_store.search(q_vec, top_k=fetch_k)

        # 2. Lexical retrieval
        lexical_results = self.lexical_retriever.search(query, top_k=fetch_k)

        # 3. Fuse with RRF
        fused = reciprocal_rank_fusion(
            ranked_lists=[dense_results, lexical_results],
            weights=[dense_weight, lexical_weight],
            k=rrf_k,
        )

        return fused[:top_k]
