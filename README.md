# rag-document-qa-application

A modular hybrid Retrieval-Augmented Generation (RAG) service with Okapi BM25 lexical ranking, dense vector search, and Reciprocal Rank Fusion (RRF). Built with FastAPI and tested across Python 3.10–3.12.

## Features

- **Hybrid Retrieval:** Combines BM25 keyword matching with dense cosine similarity using Reciprocal Rank Fusion ($k=60$).
- **Configurable Chunkers:** Sliding window token chunker and character-based chunker with separator boundary preservation.
- **FastAPI Endpoints:** REST API for document ingestion, index clearing, and question answering with inline citation IDs.
- **Evaluation Utilities:** Information retrieval scoring covering Hit Rate@K, Mean Reciprocal Rank (MRR), Precision@K, and Recall@K.
- **Offline & CI Ready:** Zero external vector DB or API token dependencies needed for running the full test suite.

## Quickstart

### 1. Install Dependencies

```bash
git clone https://github.com/taran-dev4u/rag-document-qa-application.git
cd rag-document-qa-application
pip install -r requirements.txt
```

### 2. Run the API Server

```bash
uvicorn app.main:app --reload --port 8000
```

Interactive OpenAPI docs will be available at `http://localhost:8000/docs`.

### 3. Ingest Documents

```bash
curl -X POST "http://localhost:8000/documents/ingest" \
     -H "Content-Type: application/json" \
     -d '{
       "documents": [
         {
           "doc_id": "fastapi_overview",
           "text": "FastAPI is a Python web framework built on top of Starlette and Pydantic with native async support."
         },
         {
           "doc_id": "postgres_overview",
           "text": "PostgreSQL is an open-source object-relational database system known for reliability and ACID compliance."
         }
       ]
     }'
```

### 4. Query with Citations

```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "What libraries does FastAPI build on?",
       "top_k": 2
     }'
```

Response format:
```json
{
  "query": "What libraries does FastAPI build on?",
  "answer": "According to the documentation (1 sources cited), the answer is grounded in the retrieved passages.",
  "context_block_count": 1,
  "citations": [
    {
      "citation_id": 1,
      "chunk_id": "fastapi_overview_chunk_0",
      "doc_id": "fastapi_overview",
      "score": 0.0328,
      "text_snippet": "FastAPI is a Python web framework built on top of Starlette and Pydantic..."
    }
  ]
}
```

## Running Tests

```bash
pytest -v
```

## Docker

```bash
docker compose up --build
```
