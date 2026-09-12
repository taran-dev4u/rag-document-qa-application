import math
import re
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

from .chunking import DocumentChunk


class BM25Retriever:
    """Okapi BM25 lexical ranking engine for exact keyword and term frequency scoring."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._chunks: List[DocumentChunk] = []
        self._doc_lens: List[int] = []
        self._avg_doc_len: float = 0.0
        self._doc_freqs: Dict[str, int] = {}
        self._term_freqs: List[Counter] = []
        self._idf_cache: Dict[str, float] = {}

    @property
    def total_chunks(self) -> int:
        return len(self._chunks)

    def _tokenize(self, text: str) -> List[str]:
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        return [t for t in cleaned.split() if len(t) > 1]

    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        if not chunks:
            return

        for chunk in chunks:
            tokens = self._tokenize(chunk.text)
            self._chunks.append(chunk)
            self._doc_lens.append(len(tokens))
            tf = Counter(tokens)
            self._term_freqs.append(tf)

            unique_terms: Set[str] = set(tokens)
            for term in unique_terms:
                self._doc_freqs[term] = self._doc_freqs.get(term, 0) + 1

        self._avg_doc_len = sum(self._doc_lens) / max(1, len(self._doc_lens))
        self._idf_cache.clear()

    def _get_idf(self, term: str) -> float:
        if term in self._idf_cache:
            return self._idf_cache[term]

        n_q = self._doc_freqs.get(term, 0)
        n_docs = len(self._chunks)
        # Probabilistic IDF with smoothing to ensure non-negative weights
        idf = math.log((n_docs - n_q + 0.5) / (n_q + 0.5) + 1.0)
        self._idf_cache[term] = max(0.0, idf)
        return self._idf_cache[term]

    def search(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        if not self._chunks or top_k <= 0:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores: List[float] = [0.0] * len(self._chunks)

        for term in query_tokens:
            idf = self._get_idf(term)
            if idf <= 0.0:
                continue

            for idx, tf_dict in enumerate(self._term_freqs):
                freq = tf_dict.get(term, 0)
                if freq > 0:
                    doc_len = self._doc_lens[idx]
                    denominator = freq + self.k1 * (1.0 - self.b + self.b * (doc_len / self._avg_doc_len))
                    term_score = idf * (freq * (self.k1 + 1.0)) / denominator
                    scores[idx] += term_score

        scored_pairs = [(self._chunks[i], scores[i]) for i in range(len(self._chunks)) if scores[i] > 0.0]
        scored_pairs.sort(key=lambda x: x[1], reverse=True)

        return scored_pairs[:top_k]

    def clear(self) -> None:
        self._chunks.clear()
        self._doc_lens.clear()
        self._avg_doc_len = 0.0
        self._doc_freqs.clear()
        self._term_freqs.clear()
        self._idf_cache.clear()
