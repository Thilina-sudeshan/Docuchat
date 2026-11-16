import pytest

from app.ingest import chunk_pdf, split_text


def test_short_text_is_single_chunk():
    assert split_text("hello world", chunk_size=100, overlap=10) == ["hello world"]


def test_empty_text_has_no_chunks():
    assert split_text("   ", chunk_size=100, overlap=10) == []


def test_chunks_respect_size_and_cover_text():
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    chunks = split_text(text, chunk_size=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)
    # every sentence ends up in at least one chunk
    for i in (0, 77, 199):
        assert any(f"Sentence number {i} " in c for c in chunks)


def test_chunks_overlap():
    text = "word " * 500
    chunks = split_text(text, chunk_size=200, overlap=50)
    assert len(chunks) >= 3


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, -1)])
def test_invalid_params(size, overlap):
    with pytest.raises(ValueError):
        split_text("abc", chunk_size=size, overlap=overlap)


def test_chunk_pdf_keeps_page_numbers(sample_pdf):
    chunks = chunk_pdf(sample_pdf, "policy.pdf")
    assert [c.page for c in chunks] == [1, 2]
    assert "refund" in chunks[0].text
    assert "Support hours" in chunks[1].text
    assert chunks[0].id == "policy.pdf::p1::c0"
