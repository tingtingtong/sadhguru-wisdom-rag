from ingestion.chunking import chunk_text


def test_chunk_text_basic():
    text = "Hello world. " * 200
    chunks = chunk_text(text)
    assert len(chunks) > 1
    assert all("text" in c for c in chunks)
    assert all("chunk_index" in c for c in chunks)


def test_chunk_text_with_metadata():
    text = "Yoga is the science of inner wellbeing. " * 50
    metadata = {"title": "Test", "url": "https://example.com"}
    chunks = chunk_text(text, metadata)
    assert all(c["title"] == "Test" for c in chunks)
    assert all(c["url"] == "https://example.com" for c in chunks)


def test_chunk_text_short():
    text = "Short text."
    chunks = chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0]["text"] == "Short text."
