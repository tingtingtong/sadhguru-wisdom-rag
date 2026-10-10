from __future__ import annotations

from fastapi import APIRouter

from api.schemas import SearchRequest, SearchResponse, SourceDocument
from retrieval.engine import retrieve

router = APIRouter(tags=["search"])


@router.post("/search")
async def search(request: SearchRequest) -> SearchResponse:
    """Retrieval-only endpoint (no LLM generation)."""
    results = retrieve(
        query=request.query,
        content_type=request.content_type,
        use_reranker=True,
        use_mmr=True,
    )

    sources = [
        SourceDocument(
            title=r.get("title", ""),
            url=r.get("url", ""),
            text=r.get("text", ""),
            content_type=r.get("content_type", ""),
            score=r.get("rerank_score") or r.get("rrf_score") or r.get("score"),
        )
        for r in results[: request.top_k]
    ]

    return SearchResponse(results=sources, query=request.query, total=len(sources))
