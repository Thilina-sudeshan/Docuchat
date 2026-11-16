"""Answer generation with Claude, grounded in retrieved chunks.

Each retrieved chunk is sent as a `document` content block with citations
enabled, so Claude's answer comes back with exact quotes pointing at the
chunk (and therefore the file + page) they came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import anthropic

from .config import settings
from .store import RetrievedChunk

SYSTEM_PROMPT = """You are DocuChat, an assistant that answers questions using only the \
documents provided in the user's message.

- Base every statement on the documents and cite them.
- If the documents don't contain the answer, say so plainly instead of guessing.
- Keep answers concise: a short direct answer first, then supporting detail if useful.
- Answer in the same language the user asked in."""


@dataclass
class Citation:
    source: str
    page: int
    quote: str


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    model: str = ""
    refused: bool = False


class LLMConfigError(RuntimeError):
    """Claude credentials are missing, so no request can be sent."""


@lru_cache(maxsize=1)
def get_client() -> anthropic.Anthropic:
    # Reads ANTHROPIC_API_KEY (or an `ant auth login` profile) from the environment.
    return anthropic.Anthropic()


def _document_block(chunk: RetrievedChunk) -> dict:
    return {
        "type": "document",
        "source": {"type": "text", "media_type": "text/plain", "data": chunk.text},
        "title": f"{chunk.source} (page {chunk.page})",
        "citations": {"enabled": True},
    }


def generate_answer(question: str, chunks: list[RetrievedChunk]) -> Answer:
    if not chunks:
        return Answer(
            text="I couldn't find anything relevant in the uploaded documents. "
            "Try uploading a PDF first or rephrasing your question."
        )

    content = [_document_block(c) for c in chunks]
    content.append({"type": "text", "text": question})

    try:
        response = get_client().beta.messages.create(
            model=settings.claude_model,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            output_config={"effort": settings.claude_effort},
            # If a safety classifier declines, the API retries on a fallback model
            # inside the same call instead of returning an empty answer.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": content}],
        )
    except TypeError as exc:
        # The SDK raises TypeError when it finds no API key or auth profile.
        if "authentication" in str(exc).lower():
            raise LLMConfigError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env, add your key, "
                "and restart the API server."
            ) from exc
        raise

    if response.stop_reason == "refusal":
        return Answer(
            text="Sorry, I can't help with that request.",
            model=response.model,
            refused=True,
        )

    parts: list[str] = []
    citations: list[Citation] = []
    seen: set[tuple[str, int, str]] = set()
    for block in response.content:
        if block.type != "text":
            continue  # skip thinking / fallback marker blocks
        parts.append(block.text)
        for cit in getattr(block, "citations", None) or []:
            idx = getattr(cit, "document_index", None)
            if idx is None or not 0 <= idx < len(chunks):
                continue
            src = chunks[idx]
            key = (src.source, src.page, cit.cited_text)
            if key not in seen:
                seen.add(key)
                citations.append(
                    Citation(source=src.source, page=src.page, quote=cit.cited_text.strip())
                )

    return Answer(text="".join(parts).strip(), citations=citations, model=response.model)
