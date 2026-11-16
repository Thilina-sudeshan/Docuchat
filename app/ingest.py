"""PDF loading and chunking."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass

from pypdf import PdfReader


@dataclass
class Chunk:
    text: str
    source: str  # file name
    page: int  # 1-indexed page number
    chunk_index: int  # position of the chunk within its page

    @property
    def id(self) -> str:
        return f"{self.source}::p{self.page}::c{self.chunk_index}"


def extract_pages(pdf_bytes: bytes) -> list[str]:
    """Return the text of each page of a PDF (one string per page)."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return [_clean(page.extract_text() or "") for page in reader.pages]


def _clean(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks, preferring paragraph/sentence boundaries.

    Each chunk is at most ``chunk_size`` characters. Consecutive chunks share
    roughly ``overlap`` characters so an answer that straddles a boundary is
    still retrievable.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # Try to end on a natural boundary within the last 30% of the window.
            window_start = start + int(chunk_size * 0.7)
            for sep in ("\n\n", "\n", ". ", " "):
                cut = text.rfind(sep, window_start, end)
                if cut != -1:
                    end = cut + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_pdf(
    pdf_bytes: bytes, source: str, chunk_size: int = 1000, overlap: int = 200
) -> list[Chunk]:
    """Extract and chunk a PDF, keeping the page number of every chunk."""
    chunks: list[Chunk] = []
    for page_no, page_text in enumerate(extract_pages(pdf_bytes), start=1):
        for i, piece in enumerate(split_text(page_text, chunk_size, overlap)):
            chunks.append(Chunk(text=piece, source=source, page=page_no, chunk_index=i))
    return chunks
