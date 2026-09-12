import hashlib
from typing import List, Protocol
import numpy as np


class DenseEmbeddingProvider(Protocol):
    """Protocol for dense vector embedding generators."""

    def embed_query(self, text: str) -> np.ndarray:
        ...

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        ...


class HashEmbeddingProvider:
    """Deterministic, zero-dependency feature-hash embedding provider.

    Projects token n-grams into a fixed-dimensional unit-normalized vector space.
    Useful for offline testing, CI environments, and local execution without heavyweight model weights.
    """

    def __init__(self, dimension: int = 128):
        if dimension <= 0:
            raise ValueError("Embedding dimension must be positive")
        self.dimension = dimension

    def _hash_token(self, token: str, dim: int) -> int:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big") % dim

    def embed_text(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        words = text.lower().split()
        if not words:
            return vec

        # Unigrams and bigrams
        tokens = list(words)
        for i in range(len(words) - 1):
            tokens.append(f"{words[i]}_{words[i+1]}")

        for t in tokens:
            idx = self._hash_token(t, self.dimension)
            vec[idx] += 1.0

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_text(text)

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        return np.vstack([self.embed_text(t) for t in texts])
