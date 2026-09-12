import time
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from rag_engine import (
    CharacterChunker,
    DocumentChunk,
    HashEmbeddingProvider,
    HybridRetriever,
    InMemoryVectorStore,
    BM25Retriever,
    LLMResponseGenerator,
    MockLLMClient,
)
from .schemas import (
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)

start_time = time.time()

# Core engine instances
vector_store = InMemoryVectorStore()
lexical_retriever = BM25Retriever()
embedding_provider = HashEmbeddingProvider(dimension=128)
retriever = HybridRetriever(vector_store, lexical_retriever, embedding_provider)
generator = LLMResponseGenerator(MockLLMClient())
chunker = CharacterChunker(chunk_size=300, chunk_overlap=30)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Lifecycle startup hook
    yield
    # Cleanup hook
    vector_store.clear()
    lexical_retriever.clear()


app = FastAPI(
    title="Hybrid RAG Document QA API",
    version="0.1.0",
    description="FastAPI service for hybrid dense-lexical document question answering.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
def root():
    return {
        "service": "Hybrid RAG Document QA Engine",
        "status": "online",
        "docs_url": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
def health():
    elapsed = int(time.time() - start_time)
    uptime_str = f"{elapsed // 3600}h {(elapsed % 3600) // 60}m {elapsed % 60}s"
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        total_chunks=vector_store.total_chunks,
        uptime=uptime_str,
    )


@app.post("/documents/ingest", response_model=IngestResponse, tags=["Ingestion"])
def ingest_documents(payload: IngestRequest):
    if not payload.documents:
        raise HTTPException(status_code=400, detail="Document list cannot be empty")

    custom_chunker = CharacterChunker(
        chunk_size=payload.chunk_size,
        chunk_overlap=payload.chunk_overlap,
    )

    all_chunks: List[DocumentChunk] = []
    for doc in payload.documents:
        chunks = custom_chunker.chunk(doc.text, doc_id=doc.doc_id, metadata=doc.metadata)
        all_chunks.extend(chunks)

    retriever.index_chunks(all_chunks)

    return IngestResponse(
        status="success",
        documents_ingested=len(payload.documents),
        chunks_created=len(all_chunks),
        total_indexed_chunks=vector_store.total_chunks,
    )


@app.post("/query", response_model=QueryResponse, tags=["Retrieval & Generation"])
def answer_query(payload: QueryRequest):
    if vector_store.total_chunks == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents have been indexed yet. Ingest documents via /documents/ingest first.",
        )

    fused_chunks = retriever.search(
        query=payload.query,
        top_k=payload.top_k,
        dense_weight=payload.dense_weight,
        lexical_weight=payload.lexical_weight,
    )

    response = generator.generate_response(payload.query, fused_chunks)
    return QueryResponse(**response)


@app.post("/documents/clear", tags=["Maintenance"])
def clear_index():
    count = vector_store.total_chunks
    vector_store.clear()
    lexical_retriever.clear()
    return {"status": "cleared", "chunks_removed": count}
