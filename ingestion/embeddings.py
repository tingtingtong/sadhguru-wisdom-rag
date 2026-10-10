from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np

from config.settings import settings
from observability.instrumentation import trace

_local_model = None
_openai_client = None


def _get_local_model():
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        _local_model = SentenceTransformer(settings.embedding_model)
    return _local_model


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        # Explicit timeouts: the default (10 min x retries) lets one dead connection stall a bulk ingest
        _openai_client = OpenAI(api_key=settings.openai_api_key, timeout=60.0, max_retries=3)
    return _openai_client


@trace("embed")
def embed_texts(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    """Generate dense embeddings for a list of texts (batched for API limits)."""
    if settings.embedding_provider == "openai":
        client = _get_openai_client()
        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

        def _embed(batch: list[str]) -> list[list[float]]:
            response = client.embeddings.create(model=settings.embedding_model, input=batch)
            return [item.embedding for item in response.data]

        # Batches are independent: embed them concurrently (map preserves order)
        with ThreadPoolExecutor(max_workers=5) as pool:
            results = list(pool.map(_embed, batches))
        if len(batches) > 1:
            print(f"  Embedded {len(texts)} chunks")
        return [emb for batch in results for emb in batch]

    model = _get_local_model()
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return embeddings.tolist()


@trace("embed_query")
def embed_query(text: str) -> list[float]:
    """Generate embedding for a single query (cached: guardrail and retrieval embed the same text)."""
    return list(_embed_query_cached(text))


@lru_cache(maxsize=256)
def _embed_query_cached(text: str) -> tuple[float, ...]:
    if settings.embedding_provider == "openai":
        client = _get_openai_client()
        response = client.embeddings.create(model=settings.embedding_model, input=[text])
        return tuple(response.data[0].embedding)

    model = _get_local_model()
    embedding = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
    return tuple(embedding.tolist())
