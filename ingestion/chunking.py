from __future__ import annotations

import logfire
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings
from observability.instrumentation import trace


@trace("chunking")
def chunk_text(text: str, metadata: dict | None = None) -> list[dict]:
    """Split text into chunks with metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_text(text)
    result = []
    for i, chunk in enumerate(chunks):
        doc = {
            "text": chunk,
            "chunk_index": i,
            "total_chunks": len(chunks),
            **(metadata or {}),
        }
        result.append(doc)

    logfire.info(f"Chunked text into {len(result)} chunks")
    return result
