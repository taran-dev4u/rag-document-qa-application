"""
RAG-based Document Q&A Application.
FastAPI, LangChain, Vector Embeddings (Qdrant/FAISS), and LLM generation.
"""
from typing import List, Dict, Any, Optional
import numpy as np

class DocumentChunker:
    """Splits documents into overlapping semantic chunks for embedding generation."""
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[str]:
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk = " ".join(words[i:i + self.chunk_size])
            chunks.append(chunk)
            i += self.chunk_size - self.overlap
        return chunks

class VectorStore:
    """In-memory cosine similarity vector index mimicking Qdrant/FAISS vector store."""
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.vectors = []
        self.documents = []

    def add_documents(self, docs: List[str], embeddings: List[List[float]]):
        for doc, emb in zip(docs, embeddings):
            norm_emb = np.array(emb) / (np.linalg.norm(emb) + 1e-9)
            self.vectors.append(norm_emb)
            self.documents.append(doc)

    def similarity_search(self, query_emb: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        if not self.vectors:
            return []
        q_norm = np.array(query_emb) / (np.linalg.norm(query_emb) + 1e-9)
        scores = [float(np.dot(q_norm, v)) for v in self.vectors]
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [{"document": self.documents[idx], "score": scores[idx]} for idx in top_indices]

class RAGPipeline:
    """End-to-end Retrieval Augmented Generation pipeline."""
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def generate_contextual_prompt(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        context = "
---
".join([doc["document"] for doc in retrieved_docs])
        return f"Context:
{context}

Question: {query}
Answer based on the provided context accurately:"
