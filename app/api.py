"""FastAPI backend for DocuChat.

Run:  uvicorn app.api:app --reload
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

from functools import lru_cache

import anthropic
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .config import settings
from .rag import RAGPipeline

app = FastAPI(
    title="DocuChat API",
    description="Upload PDFs and ask questions about them. Answers come with page-level citations.",
    version="0.1.0",
)


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int | None = Field(None, ge=1, le=20)


class CitationOut(BaseModel):
    source: str
    page: int
    quote: str


class ChunkOut(BaseModel):
    source: str
    page: int
    score: float
    text: str


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    retrieved: list[ChunkOut]
    model: str


@app.get("/health")
def health(pipeline: RAGPipeline = Depends(get_pipeline)):
    return {"status": "ok", "chunks": pipeline.store.count()}


@app.post("/documents")
async def upload_document(
    file: UploadFile = File(...), pipeline: RAGPipeline = Depends(get_pipeline)
):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")
    data = await file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"File is larger than {settings.max_upload_mb} MB.")
    try:
        n = pipeline.ingest_pdf(data, file.filename)
    except Exception as exc:  # pypdf raises a variety of errors on broken files
        raise HTTPException(422, f"Could not read PDF: {exc}") from exc
    if n == 0:
        raise HTTPException(
            422, "No text found in this PDF (it may be a scanned image without OCR)."
        )
    return {"filename": file.filename, "chunks": n}


@app.get("/documents")
def list_documents(pipeline: RAGPipeline = Depends(get_pipeline)):
    return {"documents": pipeline.store.list_sources()}


@app.delete("/documents/{filename}")
def delete_document(filename: str, pipeline: RAGPipeline = Depends(get_pipeline)):
    pipeline.store.delete_source(filename)
    return {"deleted": filename}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, pipeline: RAGPipeline = Depends(get_pipeline)):
    try:
        answer, retrieved = pipeline.ask(req.question, req.top_k)
    except anthropic.AuthenticationError as exc:
        raise HTTPException(500, "Claude API key is missing or invalid. Check your .env file.") from exc
    except anthropic.RateLimitError as exc:
        raise HTTPException(429, "Rate limited by the Claude API. Try again shortly.") from exc
    except anthropic.APIConnectionError as exc:
        raise HTTPException(503, "Could not reach the Claude API.") from exc
    except anthropic.APIStatusError as exc:
        raise HTTPException(502, f"Claude API error: {exc.message}") from exc

    return AskResponse(
        answer=answer.text,
        citations=[CitationOut(**vars(c)) for c in answer.citations],
        retrieved=[ChunkOut(**vars(c)) for c in retrieved],
        model=answer.model,
    )
