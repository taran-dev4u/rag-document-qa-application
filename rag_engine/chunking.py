from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    doc_id: str
    text: str
    start_char: int
    end_char: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class CharacterChunker:
    """Chunks text based on character lengths with separator preservation."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, separator: str = "\n\n"):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator

    def chunk(self, text: str, doc_id: str, metadata: Optional[Dict[str, Any]] = None) -> List[DocumentChunk]:
        if not text.strip():
            return []

        meta = metadata or {}
        chunks: List[DocumentChunk] = []
        start = 0
        text_len = len(text)
        chunk_idx = 0

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            
            # If not at the end of text, attempt to find a natural break near the boundary
            if end < text_len and self.separator in text[start:end]:
                sep_pos = text.rfind(self.separator, start, end)
                if sep_pos > start + (self.chunk_size // 3):
                    end = sep_pos + len(self.separator)

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk = DocumentChunk(
                    chunk_id=f"{doc_id}_chunk_{chunk_idx}",
                    doc_id=doc_id,
                    text=chunk_text,
                    start_char=start,
                    end_char=end,
                    metadata=meta,
                )
                chunks.append(chunk)
                chunk_idx += 1

            if end >= text_len:
                break

            # Slide window back by overlap
            start = max(start + 1, end - self.chunk_overlap)

        return chunks


class SlidingWindowChunker:
    """Word-level sliding window chunker with controlled token overlap."""

    def __init__(self, words_per_chunk: int = 100, word_overlap: int = 20):
        if word_overlap >= words_per_chunk:
            raise ValueError("word_overlap must be strictly less than words_per_chunk")
        self.words_per_chunk = words_per_chunk
        self.word_overlap = word_overlap

    def chunk(self, text: str, doc_id: str, metadata: Optional[Dict[str, Any]] = None) -> List[DocumentChunk]:
        if not text.strip():
            return []

        words = text.split()
        if not words:
            return []

        meta = metadata or {}
        chunks: List[DocumentChunk] = []
        step = self.words_per_chunk - self.word_overlap
        chunk_idx = 0

        for i in range(0, len(words), step):
            window = words[i : i + self.words_per_chunk]
            chunk_str = " ".join(window)
            
            # Approximate character offsets
            start_char = text.find(chunk_str) if chunk_str in text else 0
            end_char = start_char + len(chunk_str)

            chunks.append(
                DocumentChunk(
                    chunk_id=f"{doc_id}_sw_{chunk_idx}",
                    doc_id=doc_id,
                    text=chunk_str,
                    start_char=max(0, start_char),
                    end_char=end_char,
                    metadata=meta,
                )
            )
            chunk_idx += 1

            if i + self.words_per_chunk >= len(words):
                break

        return chunks
