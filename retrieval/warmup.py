"""Warm the network clients at startup so the first user request doesn't pay for them.

Measured cold costs: OpenAI client + first call ~8 s, Qdrant gRPC channel + first request ~3 s,
Jev client ~1 s. All are one-time per process.
"""
from __future__ import annotations

import time
from functools import lru_cache

import logfire

from config.settings import settings


def warm_up() -> None:
    try:
        from ingestion.embeddings import embed_query
        from ingestion.indexer import get_qdrant

        embed_query("warm up")
        get_qdrant().get_collection(settings.qdrant_collection)
        if settings.openrouter_api_key:
            from retrieval.reranker import _get_jev_client

            _get_jev_client()
        logfire.info("Warm-up complete")
    except Exception as e:  # never block startup on a warm-up failure
        logfire.warn(f"Warm-up failed (continuing): {e!r}")


@lru_cache(maxsize=2)
def _jev_status(minute_bucket: int) -> str:
    if not settings.openrouter_api_key:
        return "disabled"
    try:
        from guardrails.topic_classifier import _jev_in_scope

        return "ok" if _jev_in_scope("health check") is not None else "unavailable"
    except Exception:
        return "unavailable"


def jev_status() -> str:
    """'ok' | 'unavailable' | 'disabled' (no key). Cached for a minute so probes stay cheap."""
    return _jev_status(int(time.time() // 60))
