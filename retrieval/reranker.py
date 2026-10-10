from __future__ import annotations

import math

import httpx
import logfire

from config.settings import settings
from observability.instrumentation import trace

_ranker = None
_bge_model = None
_jev_client: httpx.Client | None = None

_JEV_PASSAGE_CHARS = 1500
_JEV_INSTRUCTIONS = (
    "Which passage most directly answers or is most relevant to the query? "
    "Judge only by the passage content."
)


# ── FlashRank ────────────────────────────────────────────────────────────────

def get_ranker():
    global _ranker
    if _ranker is None:
        from flashrank import Ranker
        _ranker = Ranker()
    return _ranker


def _score_flashrank(query: str, docs: list[dict]) -> list[float]:
    from flashrank import RerankRequest

    passages = [{"id": i, "text": d["text"]} for i, d in enumerate(docs)]
    ranked = get_ranker().rerank(RerankRequest(query=query, passages=passages))
    scores = [0.0] * len(docs)
    for item in ranked:
        scores[item["id"]] = float(item["score"])
    return scores


# ── BGE cross-encoder (local) ────────────────────────────────────────────────

def _get_bge():
    global _bge_model
    if _bge_model is None:
        from sentence_transformers import CrossEncoder
        _bge_model = CrossEncoder(settings.bge_model, max_length=512)
    return _bge_model


def _score_bge(query: str, docs: list[dict]) -> list[float]:
    raw = _get_bge().predict([(query, d["text"]) for d in docs])
    return [float(x) for x in raw]


# ── Jev (TypeSafe System One via OpenRouter) ─────────────────────────────────

def _get_jev_client() -> httpx.Client:
    global _jev_client
    if _jev_client is None:
        _jev_client = httpx.Client(timeout=settings.jev_timeout)
    return _jev_client


def _jev_choice(query: str, docs: list[dict]) -> list[float]:
    """One listwise Choice call: probability per candidate, summing to 1."""
    criteria = {f"d{i}": d["text"][:_JEV_PASSAGE_CHARS] for i, d in enumerate(docs)}
    payload = {
        "model": settings.jev_model,
        "state": f"Query: {query}",
        "questions": {
            "best": {
                "type": "choice",
                "instructions": _JEV_INSTRUCTIONS,
                "criteria": criteria,
            }
        },
    }
    resp = _get_jev_client().post(
        settings.jev_url,
        json=payload,
        headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
    )
    resp.raise_for_status()
    data = resp.json()
    answer = (data.get("answers") or data)["best"]
    probs = answer["probabilities"]
    return [float(probs.get(f"d{i}", 0.0)) for i in range(len(docs))]


def _score_jev(query: str, docs: list[dict]) -> list[float]:
    """Listwise ranking; pools over the batch size are ranked in rounds.

    Probabilities are relative to the pool they were judged in, so winners of
    each round are re-ranked against each other and only the final round's
    scores order the survivors.
    """
    if not settings.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    batch = max(2, settings.jev_batch_size)
    keep = max(1, settings.rerank_top_k)
    pool = list(range(len(docs)))
    scores = [0.0] * len(docs)

    while True:
        next_pool: list[int] = []
        chunks = [pool[i : i + batch] for i in range(0, len(pool), batch)]
        for chunk in chunks:
            probs = _jev_choice(query, [docs[i] for i in chunk])
            for i, p in zip(chunk, probs):
                scores[i] = p
            ranked = sorted(chunk, key=lambda i: scores[i], reverse=True)
            next_pool.extend(ranked[:keep] if len(chunks) > 1 else ranked)
        if len(chunks) == 1:
            return scores
        pool = next_pool


_SCORERS = {
    "flashrank": _score_flashrank,
    "bge": _score_bge,
    "jev": _score_jev,
}


@trace("rerank")
def rerank(query: str, documents: list[dict], top_k: int | None = None) -> list[dict]:
    """Rerank documents with the configured provider (settings.reranker_provider).

    Falls back to FlashRank if the provider fails (e.g. Jev API down, missing key).
    """
    top_k = top_k or settings.rerank_top_k
    if not documents:
        return []

    provider = settings.reranker_provider.lower()
    scorer = _SCORERS.get(provider)
    if scorer is None:
        raise ValueError(
            f"Unknown reranker_provider {provider!r}; expected one of {sorted(_SCORERS)}"
        )

    try:
        scores = scorer(query, documents)
    except Exception as e:
        if provider == "flashrank":
            raise
        logfire.warn(f"Reranker '{provider}' failed ({e!r}); falling back to flashrank")
        provider = "flashrank"
        scores = _score_flashrank(query, documents)

    order = sorted(range(len(documents)), key=lambda i: scores[i], reverse=True)
    results = []
    for i in order[:top_k]:
        doc = documents[i].copy()
        doc["rerank_score"] = scores[i] if math.isfinite(scores[i]) else 0.0
        doc["reranker"] = provider
        results.append(doc)
    return results
