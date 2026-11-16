# 📄 DocuChat: Chat with your PDFs (RAG + Claude)

Upload PDF documents and ask questions in plain language. DocuChat finds the
relevant passages with vector search and has Claude answer **only from those
passages**, with **page-level citations** so you can check every claim.

<!-- Add a screenshot or demo GIF here: ![demo](docs/demo.gif) -->

## Features

- 📥 **PDF ingestion:** text extraction per page, overlapping chunking that respects paragraph and sentence boundaries
- 🔎 **Semantic search:** local embeddings + ChromaDB (no extra API key, persists to disk)
- 🤖 **Grounded answers:** Claude's native citations API returns exact quotes mapped to file + page
- 🚫 **Says "I don't know":** answers are restricted to the uploaded documents
- ⚡ **FastAPI backend:** typed request/response models, auto-generated docs at `/docs`
- 💬 **Streamlit chat UI:** multi-file upload, source viewer, retrieved-chunk inspector
- 🐳 **Docker Compose:** one command to run API + UI
- ✅ **Tested:** unit tests for chunking, end-to-end API tests with a mocked LLM

## Architecture

```
            ┌──────────────┐   upload PDF    ┌──────────────────────────────┐
            │  Streamlit   │ ──────────────► │  FastAPI                     │
            │  chat UI     │ ◄────────────── │  /documents  /ask  /health   │
            └──────────────┘  answer+cites   └──────────────┬───────────────┘
                                                            │
                     ingest: pypdf → chunk (page-aware) → embed → ChromaDB
                     ask:    embed question → top-k chunks → Claude (citations on)
```

1. **Ingest:** `pypdf` extracts text per page; `split_text` cuts it into ~1000-char chunks with 200-char overlap.
2. **Index:** chunks are embedded (all-MiniLM-L6-v2) and stored in ChromaDB with `source` + `page` metadata.
3. **Retrieve:** the question is embedded and the top-k most similar chunks are fetched (cosine similarity).
4. **Generate:** each chunk is sent to Claude as a `document` block with citations enabled; the cited
   `document_index` is mapped back to the file and page.

## Tech stack

Python 3.12 · FastAPI · Claude API (`anthropic` SDK) · ChromaDB · pypdf · Streamlit · Docker · pytest

## Quick start

```bash
git clone https://github.com/<your-username>/docuchat.git
cd docuchat
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
copy .env.example .env          # macOS/Linux: cp .env.example .env
# then put your ANTHROPIC_API_KEY in .env
```

Run the API and the UI in two terminals:

```bash
uvicorn app.api:app --reload
streamlit run ui/streamlit_app.py
```

Open http://localhost:8501 for the chat UI, or http://localhost:8000/docs for the API.

### With Docker

```bash
docker compose up --build
```

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/documents` | Upload a PDF (multipart `file`), returns number of chunks indexed |
| `GET` | `/documents` | List indexed files |
| `DELETE` | `/documents/{filename}` | Remove a file from the index |
| `POST` | `/ask` | `{"question": "...", "top_k": 5}` → answer, citations, retrieved chunks |
| `GET` | `/health` | Status and chunk count |

## Tests

```bash
pytest -v
```

The API tests use a real ChromaDB store and a fake LLM, so they run without an API key.

## Configuration

All settings live in `.env` (see `.env.example`): `CLAUDE_MODEL`, `CLAUDE_EFFORT`,
`CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`, `MAX_UPLOAD_MB`.

## Roadmap

- [ ] Streaming answers
- [ ] Multi-turn chat memory
- [ ] OCR for scanned PDFs
- [ ] Hybrid search (BM25 + vectors) and re-ranking
- [ ] Evaluation set to measure answer accuracy

## Project structure

```
app/
  api.py      FastAPI routes
  config.py   settings from .env
  ingest.py   PDF text extraction + chunking
  store.py    ChromaDB vector store
  llm.py      Claude answer generation with citations
  rag.py      pipeline that ties it together
ui/
  streamlit_app.py
tests/
```
