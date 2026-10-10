from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from config.settings import settings
from ingestion.embeddings import embed_query
from observability.instrumentation import trace


@trace("dense_search")
def dense_search(
    client: QdrantClient,
    query: str,
    top_k: int | None = None,
    content_type: str | None = None,
) -> list[dict]:
    """Dense vector similarity search."""
    top_k = top_k or settings.retrieval_top_k
    query_vector = embed_query(query)

    query_filter = None
    if content_type:
        # Support both 'content_type' and 'source_type' field names
        query_filter = Filter(
            should=[
                FieldCondition(key="content_type", match=MatchValue(value=content_type)),
                FieldCondition(key="source_type", match=MatchValue(value=content_type)),
            ]
        )

    # Try named vector first, fall back to unnamed
    try:
        results = client.query_points(
            collection_name=settings.qdrant_collection,
            query=query_vector,
            using="dense",
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
    except Exception:
        results = client.query_points(
            collection_name=settings.qdrant_collection,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )

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
