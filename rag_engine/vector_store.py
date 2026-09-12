from typing import Callable, List, Optional, Tuple
import numpy as np

from .chunking import DocumentChunk


class InMemoryVectorStore:
    """In-memory vector store performing vectorized cosine similarity search."""

    def __init__(self):
        self._chunks: List[DocumentChunk] = []
        self._matrix: Optional[np.ndarray] = None

    @property
    def total_chunks(self) -> int:
        return len(self._chunks)

    def add_chunks(self, chunks: List[DocumentChunk], embeddings: np.ndarray) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Chunk count must match embedding matrix rows")
        if not chunks:
            return

        # Ensure unit normalization for fast cosine similarity via dot product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = embeddings / norms

        if self._matrix is None or len(self._chunks) == 0:
            self._matrix = normalized
            self._chunks = list(chunks)
        else:
            self._matrix = np.vstack([self._matrix, normalized])
            self._chunks.extend(chunks)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        metadata_filter: Optional[Callable[[dict], bool]] = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        if self._matrix is None or len(self._chunks) == 0 or top_k <= 0:
            return []

        q_norm = np.linalg.norm(query_vector)
        if q_norm == 0:
            return []
        q_unit = query_vector / q_norm

        scores = np.dot(self._matrix, q_unit)
        ranked_indices = np.argsort(scores)[::-1]

        results: List[Tuple[DocumentChunk, float]] = []
        for idx in ranked_indices:
            chunk = self._chunks[idx]
            if metadata_filter and not metadata_filter(chunk.metadata):
                continue
            results.append((chunk, float(scores[idx])))
            if len(results) >= top_k:
                break

        return results

    def delete_document(self, doc_id: str) -> int:
        if not self._chunks or self._matrix is None:
            return 0

        keep_indices = [i for i, c in enumerate(self._chunks) if c.doc_id != doc_id]
        deleted_count = len(self._chunks) - len(keep_indices)

        if not keep_indices:
            self._chunks = []
            self._matrix = None
        else:
            self._chunks = [self._chunks[i] for i in keep_indices]
            self._matrix = self._matrix[keep_indices]

        return deleted_count

    def clear(self) -> None:
        self._chunks = []
        self._matrix = None
