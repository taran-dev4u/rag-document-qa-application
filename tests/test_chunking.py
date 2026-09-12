import pytest
from rag_engine.chunking import CharacterChunker, SlidingWindowChunker


def test_character_chunker_basic():
    text = "Paragraph one of text.\n\nParagraph two with more details.\n\nParagraph three."
    chunker = CharacterChunker(chunk_size=40, chunk_overlap=10, separator="\n\n")
    chunks = chunker.chunk(text, doc_id="doc_1", metadata={"category": "tech"})

    assert len(chunks) >= 2
    assert all(c.doc_id == "doc_1" for c in chunks)
    assert all(c.metadata.get("category") == "tech" for c in chunks)
    assert chunks[0].chunk_id == "doc_1_chunk_0"


def test_character_chunker_empty_input():
    chunker = CharacterChunker()
    assert chunker.chunk("", doc_id="empty") == []
    assert chunker.chunk("   \n\t  ", doc_id="whitespace") == []


def test_character_chunker_invalid_overlap():
    with pytest.raises(ValueError):
        CharacterChunker(chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValueError):
        CharacterChunker(chunk_size=100, chunk_overlap=150)


def test_sliding_window_chunker():
    words = [f"word{i}" for i in range(50)]
    text = " ".join(words)
    chunker = SlidingWindowChunker(words_per_chunk=20, word_overlap=5)
    chunks = chunker.chunk(text, doc_id="doc_sw")

    assert len(chunks) == 3
    assert len(chunks[0].text.split()) == 20
    assert len(chunks[1].text.split()) == 20
    assert chunks[0].chunk_id == "doc_sw_sw_0"


def test_sliding_window_empty():
    chunker = SlidingWindowChunker()
    assert chunker.chunk("", doc_id="none") == []
