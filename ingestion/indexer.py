from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import logfire
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Modifier,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from config.settings import settings
from ingestion.embeddings import embed_texts
from ingestion.quality import content_hash
from ingestion.sparse_encoding import encode_document
from observability.instrumentation import trace


@lru_cache(maxsize=4)
def _client(prefer_grpc: bool) -> QdrantClient:
    if settings.qdrant_url:
        return QdrantClient(
            url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=120, prefer_grpc=prefer_grpc
        )
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port, prefer_grpc=prefer_grpc)


def get_qdrant(prefer_grpc: bool | None = None) -> QdrantClient:
    """Shared client (one per transport).

    Reusing it matters: a fresh client costs ~0.5 s to build plus ~1.5 s for the first request's
    TLS handshake to the cloud cluster. gRPC also multiplexes parallel searches over one channel.
    """
    return _client(settings.qdrant_prefer_grpc if prefer_grpc is None else prefer_grpc)


@trace("ensure_collection")
def ensure_collection(client: QdrantClient) -> None:
    """Create Qdrant collection if it doesn't exist."""
    collections = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection not in collections:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config={
                "dense": VectorParams(size=settings.embedding_dim, distance=Distance.COSINE)
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(index=SparseIndexParams(), modifier=Modifier.IDF)
            },
        )
        logfire.info(f"Created collection: {settings.qdrant_collection}")


def _point_id(chunk: dict) -> str:
    """Deterministic id: re-ingesting the same chunk overwrites instead of duplicating."""
    key = f"{chunk.get('url', '')}|{chunk.get('chunk_index', 0)}|{content_hash(chunk['text'])}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def _upsert_with_retry(client: QdrantClient, points: list, attempts: int = 5) -> None:
    for attempt in range(attempts):
        try:
            client.upsert(collection_name=settings.qdrant_collection, points=points)
            return
        except Exception as e:  # timeouts / transient network errors on the cloud cluster
            if attempt == attempts - 1:
                raise
            wait = 2 ** attempt * 2
            logfire.warn(f"Upsert failed ({e!r}); retry {attempt + 1}/{attempts - 1} in {wait}s")
            time.sleep(wait)


@trace("index_documents")
def index_documents(chunks: list[dict], mega_batch_size: int = 500) -> int:
    """Index chunked documents into Qdrant with dense + sparse vectors.

    Processes in mega-batches: embed + upsert 500 chunks at a time so progress is saved
    incrementally. Resumable: chunks whose (deterministic) id already exists are skipped, so a
    re-run after a crash only embeds and uploads what is missing.
    """
    if not chunks:
        return 0

    # Bulk uploads over gRPC: REST sends each 3072-d vector as ~60 KB of JSON text (4x slower here)
    client = get_qdrant(prefer_grpc=settings.qdrant_grpc_ingest)
    ensure_collection(client)

    total_indexed = skipped = 0

    for mega_start in range(0, len(chunks), mega_batch_size):
        mega_batch = chunks[mega_start : mega_start + mega_batch_size]
        ids = [_point_id(c) for c in mega_batch]
        existing = {
            str(r.id)
            for r in client.retrieve(
                collection_name=settings.qdrant_collection, ids=ids, with_payload=False, with_vectors=False
            )
        }
        todo = [(pid, c) for pid, c in zip(ids, mega_batch) if pid not in existing]
        skipped += len(mega_batch) - len(todo)
        if not todo:
            total_indexed += len(mega_batch)
            print(f"  Indexed {total_indexed}/{len(chunks)} chunks (skipped {skipped} already present)")
            continue

        # Dense embeddings for the missing chunks
        dense_vectors = embed_texts([c["text"] for _, c in todo])

        points = []
        for (pid, chunk), dense in zip(todo, dense_vectors):
            indices, values = encode_document(chunk["text"])
            payload = {
                "text": chunk["text"],
                "title": chunk.get("title", ""),
                "url": chunk.get("url", ""),
                "content_type": chunk.get("content_type", "article"),
                "topics": chunk.get("topics", []),
                "chunk_index": chunk.get("chunk_index", 0),
                "total_chunks": chunk.get("total_chunks", 1),
            }
            point = PointStruct(id=pid, vector={"dense": dense}, payload=payload)
            if indices:
                point.vector["sparse"] = SparseVector(indices=indices, values=values)
            points.append(point)

        # Sequential 20-point requests to a remote cluster were the bottleneck (~140 chunks/min);
        # moderate batches sent concurrently (with retries) are ~4x faster.
        upsert_batch_size = 50
        batches = [points[s : s + upsert_batch_size] for s in range(0, len(points), upsert_batch_size)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda b: _upsert_with_retry(client, b), batches))

        total_indexed += len(mega_batch)
        print(f"  Indexed {total_indexed}/{len(chunks)} chunks (skipped {skipped} already present)", flush=True)

    logfire.info(f"Indexed {total_indexed} chunks into Qdrant ({skipped} already present)")
    return total_indexed
