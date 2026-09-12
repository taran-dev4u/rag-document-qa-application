"""Hybrid retrieval-augmented generation engine."""

from .chunking import CharacterChunker, DocumentChunk, SlidingWindowChunker
from .embeddings import DenseEmbeddingProvider, HashEmbeddingProvider
from .evaluation import evaluate_retrieval
from .generator import ContextBuilder, LLMResponseGenerator, MockLLMClient
from .hybrid_search import HybridRetriever, reciprocal_rank_fusion
from .lexical_retriever import BM25Retriever
from .vector_store import InMemoryVectorStore

__all__ = [
    "CharacterChunker",
    "SlidingWindowChunker",
    "DocumentChunk",
    "DenseEmbeddingProvider",
    "HashEmbeddingProvider",
    "InMemoryVectorStore",
    "BM25Retriever",
    "reciprocal_rank_fusion",
    "HybridRetriever",
    "ContextBuilder",
    "LLMResponseGenerator",
    "MockLLMClient",
    "evaluate_retrieval",
]
