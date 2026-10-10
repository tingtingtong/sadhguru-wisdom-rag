from __future__ import annotations

import logfire

from ingestion.chunking import chunk_text
from ingestion.indexer import index_documents
from ingestion.quality import drop_shared_boilerplate_docs, filter_chunks
from observability.instrumentation import trace
from scraper.extractors import _clean_content
from scraper.models import ScrapedDocument
from scraper.storage import load_documents


@trace("ingestion_pipeline")
def run_ingestion(content_type: str | None = None) -> int:
    """Full ingestion pipeline: load -> chunk -> embed -> index."""
    with logfire.span("load_documents"):
        docs = load_documents(content_type)
        logfire.info(f"Loaded {len(docs)} documents")

    if not docs:
        logfire.warn("No documents to ingest")
        return 0

    print(f"  Loaded {len(docs)} documents")
    docs, dropped = drop_shared_boilerplate_docs(docs)
    print(f"  Dropped {dropped} documents with a generic shared description -> {len(docs)} remain")

    all_chunks = []
    for doc in docs:
        metadata = {
            "title": doc.title,
            "url": doc.url,
            "content_type": doc.content_type,
            "topics": doc.topics,
        }
        chunks = chunk_text(_clean_content(doc.content), metadata)
        all_chunks.extend(chunks)

    print(f"  Total chunks: {len(all_chunks)}")
    all_chunks, stats = filter_chunks(all_chunks)
    print(f"  After quality filter: {stats}")
    logfire.info(f"Chunk quality filter: {stats}")

    count = index_documents(all_chunks)
    logfire.info(f"Ingestion complete: {count} chunks indexed")
    return count


def ingest_documents(docs: list[ScrapedDocument]) -> int:
    """Ingest a list of ScrapedDocument objects directly."""
    all_chunks = []
    for doc in docs:
        metadata = {
            "title": doc.title,
            "url": doc.url,
            "content_type": doc.content_type,
            "topics": doc.topics,
        }
        chunks = chunk_text(_clean_content(doc.content), metadata)
        all_chunks.extend(chunks)

    all_chunks, _ = filter_chunks(all_chunks)
    return index_documents(all_chunks)
