"""Vector store backed by ChromaDB (local, persistent, free).

Embeddings use Chroma's built-in default model (all-MiniLM-L6-v2 via ONNX),
so no extra API key is needed. The model downloads once (~80 MB) on first use.
"""

from __future__ import annotations

from dataclasses import dataclass

import chromadb

from .config import settings
from .ingest import Chunk


@dataclass
class RetrievedChunk:
    text: str
    source: str
    page: int
    score: float  # cosine similarity, higher is better


class VectorStore:
    def __init__(self, path: str | None = None, collection: str | None = None):
        self._client = chromadb.PersistentClient(path=str(path or settings.chroma_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection or settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        self._collection.upsert(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source, "page": c.page} for c in chunks],
        )
        return len(chunks)

    def search(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        if self.count() == 0:
            return []
        res = self._collection.query(
            query_texts=[query],
            n_results=min(k or settings.top_k, self.count()),
        )
        results = []
        for text, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            results.append(
                RetrievedChunk(
                    text=text,
                    source=str(meta["source"]),
                    page=int(meta["page"]),
                    score=round(1 - float(dist), 4),
                )
            )
        return results

    def list_sources(self) -> list[str]:
        metas = self._collection.get(include=["metadatas"])["metadatas"] or []
        return sorted({str(m["source"]) for m in metas})

    def delete_source(self, source: str) -> None:
        self._collection.delete(where={"source": source})

    def count(self) -> int:
        return self._collection.count()
