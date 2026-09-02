# Enterprise Retrieval-Augmented Generation (RAG) Document Q&A Platform

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Async%20REST%20API-teal.svg)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-LLM%20Framework-purple.svg)](https://www.langchain.com/)
[![Vector DB](https://img.shields.io/badge/Vector%20DB-Qdrant-red.svg)](https://qdrant.tech/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue.svg)](https://www.docker.com/)

---

## 📌 Executive Summary & Architecture

This repository contains a high-performance **Enterprise Retrieval-Augmented Generation (RAG)** platform engineered to perform accurate, hallucination-resistant semantic question-answering over heterogeneous enterprise document repositories (PDFs, Markdown, Docx, Technical Specs).

Built on an asynchronous microservice architecture using **FastAPI**, **LangChain**, and **Qdrant Vector Database**, the platform implements chunking strategies with sliding context windows, dense semantic embedding generation, cosine similarity vector search, and strict citation grounding.

```
+--------------------+      PDF / Docx Ingest       +---------------------+
| Document Repository| ---------------------------> | Recursive Chunking  |
+--------------------+                              | (500 tokens, 10% ov)|
                                                    +---------------------+
                                                               |
                                                     Embedding Generation
                                                     (BGE / MiniLM / ADA)
                                                               v
+--------------------+      Dense Vector Search     +---------------------+
| User Question      | ---------------------------> | Qdrant Vector DB    |
+--------------------+                              | (HNSW Indexing)     |
          |                                         +---------------------+
          |                                                    |
          |           Context + Query Prompt                   v
          +---------------------------------------> +---------------------+
                                                    | LLM Generation      |
                                                    | Grounded Citations  |
                                                    +---------------------+
```

---

## 🚀 Key Technical Features

### 1. Document Ingestion & Chunking
- **Recursive Character Text Splitting:** Respects structural headings, paragraphs, and markdown syntax with configurable chunk size (500-1000 tokens) and overlap (100 tokens).
- **Metadata Tagging:** Attaches source file name, page numbers, and author tags for exact citation attribution.

### 2. High-Efficiency Vector Retrieval
- **Qdrant Integration:** Hierarchical Navigable Small World (HNSW) graph indexing providing sub-10ms retrieval across tens of thousands of vector embeddings.
- **Cosine Distance Scoring:** Computes top-$k$ nearest neighbors:
  $$\text{Cosine Similarity}(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2}$$

### 3. API Endpoints & Verification
- `POST /api/v1/documents/upload`: Multipart document ingestion and automatic indexing.
- `POST /api/v1/query`: Semantic query execution returning answer text, source snippet citations, and confidence scores.

---

## 📂 Repository Structure

```
rag-document-qa-application/
├── src/
│   └── rag_app/
│       ├── rag_engine.py            # Vector store retrieval, embeddings, and prompt chaining
│       └── __init__.py
├── tests/
│   └── test_rag.py                  # Unit tests verifying chunking, retrieval, and synthesis
├── BUILD_STATUS.md                  # Verification logs and test metrics
├── VERIFIED.md                      # Architecture verification badge
└── README.md                        # Documentation
```

---

## 🛠️ Setup & Execution

```bash
git clone https://github.com/taran-dev4u/rag-document-qa-application.git
cd rag-document-qa-application
pip install -r requirements.txt
pytest tests/
```

---

## 👨‍💻 Author
- **Author:** Taran Mamidala
