from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root_endpoint():
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "online"


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "uptime" in data


def test_ingest_and_query_flow():
    # 1. Clear store
    client.post("/documents/clear")

    # 2. Ingest document
    payload = {
        "documents": [
            {
                "doc_id": "fastapi_guide",
                "text": "FastAPI is a modern, fast web framework for building APIs with Python based on standard Python type hints.",
                "metadata": {"type": "docs"},
            },
            {
                "doc_id": "postgres_guide",
                "text": "PostgreSQL is a powerful, open-source object-relational database system with over 35 years of active development.",
                "metadata": {"type": "docs"},
            },
        ],
        "chunk_size": 150,
        "chunk_overlap": 20,
    }
    ingest_resp = client.post("/documents/ingest", json=payload)
    assert ingest_resp.status_code == 200
    ingest_data = ingest_resp.json()
    assert ingest_data["documents_ingested"] == 2
    assert ingest_data["total_indexed_chunks"] >= 2

    # 3. Query
    query_resp = client.post(
        "/query",
        json={"query": "What is FastAPI built on?", "top_k": 2},
    )
    assert query_resp.status_code == 200
    q_data = query_resp.json()
    assert "citations" in q_data
    assert len(q_data["citations"]) > 0
    assert q_data["citations"][0]["doc_id"] == "fastapi_guide"


def test_query_empty_index():
    client.post("/documents/clear")
    resp = client.post("/query", json={"query": "test query", "top_k": 2})
    assert resp.status_code == 400
