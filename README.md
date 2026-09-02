# Enterprise RAG Document Q&A Application

Retrieval-Augmented Generation (RAG) backend engineered with FastAPI, LangChain, and Qdrant Vector Database for semantic document querying and grounded citation synthesis.

## Architecture

```
User Query ───► Embedding Model ───► Qdrant Vector Search (HNSW) ───► Relevant Chunks
                                                                            │
Document PDF/Docx ───► Text Chunker ────────────────────────────────────────┤
                                                                            ▼
                                                                 LLM Context Prompt
                                                                            │
                                                                            ▼
                                                                Grounded Answer + Citations
```

- **Document Chunking:** Recursive text splitting with configurable chunk sizes and token overlap.
- **Vector Retrieval:** Qdrant HNSW indexing for sub-10ms similarity search over dense embeddings.
- **REST Endpoints:** Asynchronous FastAPI endpoints for document upload, index management, and query answering.

## Quick Start

```bash
pip install -r requirements.txt
pytest tests/
```
