from __future__ import annotations

from fastapi import APIRouter, Depends
from qdrant_client import QdrantClient

from api.dependencies import get_qdrant_client, get_settings
from api.schemas import HealthResponse
from config.settings import Settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/health/ready")
async def health_ready(
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    import asyncio

    from retrieval.warmup import jev_status

    qdrant_status, points = "unknown", None
    try:
        client = get_qdrant_client()
        info = await asyncio.to_thread(client.get_collection, settings.qdrant_collection)
        points = info.points_count
        qdrant_status = "connected" if points else "empty"
    except Exception:
        qdrant_status = "disconnected"

    jev = await asyncio.to_thread(jev_status)
    # Jev being down is degraded, not down: reranking, the gate and the guardrail all have fallbacks
    ready = qdrant_status == "connected" and (jev in ("ok", "disabled"))
    return HealthResponse(
        status="ready" if ready else "degraded",
        qdrant=qdrant_status,
        llm_provider=settings.llm_provider,
        collection=settings.qdrant_collection,
        points=points,
        reranker=settings.reranker_provider,
        jev=jev,
    )
