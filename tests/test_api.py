"""End-to-end API tests with a real ChromaDB store and a fake LLM (no API key needed)."""

import pytest
from fastapi.testclient import TestClient

from app.api import app, get_pipeline
from app.llm import Answer, Citation
from app.rag import RAGPipeline
from app.store import VectorStore


def fake_answer(question, chunks):
    top = chunks[0]
    return Answer(
        text=f"Based on {top.source}: {top.text}",
        citations=[Citation(source=top.source, page=top.page, quote=top.text[:40])],
        model="fake-model",
    )


@pytest.fixture
def client(tmp_path):
    pipeline = RAGPipeline(store=VectorStore(path=tmp_path / "chroma"), answer_fn=fake_answer)
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "chunks": 0}


def test_rejects_non_pdf(client):
    r = client.post("/documents", files={"file": ("notes.txt", b"hi", "text/plain")})
    assert r.status_code == 400


def test_upload_and_ask(client, sample_pdf):
    r = client.post("/documents", files={"file": ("policy.pdf", sample_pdf, "application/pdf")})
    assert r.status_code == 200, r.text
    assert r.json() == {"filename": "policy.pdf", "chunks": 2}
    assert client.get("/documents").json() == {"documents": ["policy.pdf"]}

    r = client.post("/ask", json={"question": "How many days do I have to get a refund?"})
    assert r.status_code == 200, r.text
    body = r.json()
    # retrieval should rank the refund page first
    assert body["retrieved"][0]["page"] == 1
    assert "30 days" in body["answer"]
    assert body["citations"][0]["source"] == "policy.pdf"


def test_reupload_replaces_chunks(client, sample_pdf):
    for _ in range(2):
        client.post("/documents", files={"file": ("policy.pdf", sample_pdf, "application/pdf")})
    assert client.get("/health").json()["chunks"] == 2


def test_delete_document(client, sample_pdf):
    client.post("/documents", files={"file": ("policy.pdf", sample_pdf, "application/pdf")})
    client.delete("/documents/policy.pdf")
    assert client.get("/documents").json() == {"documents": []}


def test_empty_question_rejected(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422
