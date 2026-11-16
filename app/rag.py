"""The RAG pipeline: ingest documents, retrieve chunks, generate answers."""

from __future__ import annotations

from .config import settings
from .ingest import chunk_pdf
from .llm import Answer, generate_answer
from .store import RetrievedChunk, VectorStore


class RAGPipeline:
    def __init__(self, store: VectorStore | None = None, answer_fn=generate_answer):
        self.store = store or VectorStore()
        self._answer_fn = answer_fn

    def ingest_pdf(self, pdf_bytes: bytes, filename: str) -> int:
        # Re-uploading a file replaces its old chunks.
        self.store.delete_source(filename)
        chunks = chunk_pdf(
            pdf_bytes, filename, settings.chunk_size, settings.chunk_overlap
        )
        return self.store.add(chunks)

    def ask(self, question: str, k: int | None = None) -> tuple[Answer, list[RetrievedChunk]]:
        retrieved = self.store.search(question, k)
        return self._answer_fn(question, retrieved), retrieved
