import pytest
from rag_app.rag_engine import DocumentChunker, VectorStore, RAGPipeline

def test_document_chunking():
    chunker = DocumentChunker(chunk_size=10, overlap=2)
    text = " ".join([f"word{i}" for i in range(25)])
    chunks = chunker.chunk_text(text)
    assert len(chunks) >= 3
    assert "word0" in chunks[0]

def test_vector_similarity_search():
    store = VectorStore(embedding_dim=3)
    docs = ["Kubernetes cluster orchestration", "Deep learning transformer models", "FastAPI microservices"]
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ]
    store.add_documents(docs, embeddings)
    results = store.similarity_search([0.9, 0.1, 0.0], top_k=1)
    assert len(results) == 1
    assert "Kubernetes" in results[0]["document"]
