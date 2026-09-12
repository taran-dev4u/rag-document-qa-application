from typing import List, Protocol, Tuple

from .chunking import DocumentChunk


class LLMClient(Protocol):
    """Protocol for LLM generation clients."""

    def generate(self, prompt: str) -> str:
        ...


class MockLLMClient:
    """Deterministic, offline-friendly mock LLM generator for tests and dry-runs."""

    def generate(self, prompt: str) -> str:
        # Extracts citation IDs mentioned in the prompt context
        lines = prompt.split("\n")
        sources = [line for line in lines if line.startswith("[") and "]" in line]
        if not sources:
            return "Based on the provided documents, no sufficient information was found to answer the query."
        return f"According to the documentation ({len(sources)} sources cited), the answer is grounded in the retrieved passages."


class ContextBuilder:
    """Formats retrieved document chunks into clean, cited prompt context."""

    @staticmethod
    def build_context(scored_chunks: List[Tuple[DocumentChunk, float]]) -> str:
        if not scored_chunks:
            return "No relevant context found."

        context_blocks = []
        for idx, (chunk, _) in enumerate(scored_chunks, start=1):
            source = chunk.doc_id
            block = f"[{idx}] (Source: {source})\n{chunk.text}"
            context_blocks.append(block)

        return "\n\n".join(context_blocks)


class LLMResponseGenerator:
    """Assembles prompt context and generates grounded answers with source citations."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def generate_response(self, query: str, retrieved_chunks: List[Tuple[DocumentChunk, float]]) -> dict:
        context = ContextBuilder.build_context(retrieved_chunks)
        prompt = (
            "You are a factual technical assistant. Answer the question strictly using the retrieved context below. "
            "Cite sources using [1], [2], etc. If the answer cannot be determined from the context, say so.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )
        answer = self.llm_client.generate(prompt)

        citations = [
            {
                "citation_id": idx,
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "score": round(score, 4),
                "text_snippet": chunk.text[:120] + "..." if len(chunk.text) > 120 else chunk.text,
            }
            for idx, (chunk, score) in enumerate(retrieved_chunks, start=1)
        ]

        return {
            "query": query,
            "answer": answer,
            "context_block_count": len(retrieved_chunks),
            "citations": citations,
        }
