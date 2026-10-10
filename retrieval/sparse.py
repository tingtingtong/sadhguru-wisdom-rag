from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector

from config.settings import settings
from ingestion.sparse_encoding import encode_query
from observability.instrumentation import trace


@trace("sparse_search")
def sparse_search(
    client: QdrantClient,
    query: str,
    top_k: int | None = None,
) -> list[dict]:
    """Sparse BM25-style search via Qdrant sparse vectors."""
    top_k = top_k or settings.retrieval_top_k

    indices, values = encode_query(query)
    if not indices:
        return []

    try:
        results = client.query_points(
            collection_name=settings.qdrant_collection,
            query=SparseVector(indices=indices, values=values),
            using="sparse",
            limit=top_k,
            with_payload=True,
        )
    except Exception:
        # Collection may not have sparse vectors configured
        return []

    return [
        {
            "id": str(r.id),
            "score": r.score,
            "text": r.payload.get("text", ""),
            "title": r.payload.get("title", ""),
            "url": r.payload.get("url", ""),
            "content_type": r.payload.get("content_type", "") or r.payload.get("source_type", ""),
            "topics": r.payload.get("topics", []),
        }
        for r in results.points
    ]
