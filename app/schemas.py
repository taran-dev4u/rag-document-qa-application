from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DocumentItem(BaseModel):
    doc_id: str = Field(..., description="Unique document identifier")
    text: str = Field(..., min_length=1, description="Raw text content of the document")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Custom metadata tags")


class IngestRequest(BaseModel):
    documents: List[DocumentItem]
    chunk_size: int = Field(default=300, ge=50, le=2000)
    chunk_overlap: int = Field(default=30, ge=0, le=500)


class IngestResponse(BaseModel):
    status: str
    documents_ingested: int
    chunks_created: int
    total_indexed_chunks: int


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2, description="Natural language search query")
    top_k: int = Field(default=3, ge=1, le=20)
    dense_weight: float = Field(default=1.0, ge=0.0, le=5.0)
    lexical_weight: float = Field(default=1.0, ge=0.0, le=5.0)


class CitationItem(BaseModel):
    citation_id: int
    chunk_id: str
    doc_id: str
    score: float
    text_snippet: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    context_block_count: int
    citations: List[CitationItem]


class HealthResponse(BaseModel):
    status: str
    version: str
    total_chunks: int
    uptime: str
