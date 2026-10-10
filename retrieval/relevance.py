"""Relevance gate: decide whether the retrieved chunks can actually answer the question.

Reranker scores aren't comparable across providers (Jev's are relative probabilities within the
candidate pool; FlashRank's saturate near 1.0 even for absent topics), so the gate uses an
absolute signal per provider:
  - jev:       one `noul` call -> P(passages directly answer the question)
  - flashrank/bge: optional floor on the best rerank score (off by default: it blocked good questions)
"""
from __future__ import annotations

import logfire

from config.settings import settings
from retrieval.reranker import _get_jev_client

_JEV_ANSWERABLE = "The passages contain specific information that directly answers the question."
_PASSAGE_CHARS = 1200


def jev_answerability(query: str, docs: list[dict]) -> float:
    state = "Question: " + query + "\n\nPassages:\n" + "\n\n".join(
        f"[{i}] {d['text'][:_PASSAGE_CHARS]}" for i, d in enumerate(docs, 1)
    )
    resp = _get_jev_client().post(
        settings.jev_url,
        json={
            "model": settings.jev_model,
            "state": state,
            "questions": {"answerable": {"type": "noul", "instructions": _JEV_ANSWERABLE}},
        },
        headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
    )
    resp.raise_for_status()
    data = resp.json()
    return float((data.get("answers") or data)["answerable"]["noul"])


def _score_floor(docs: list[dict]) -> bool:
    if settings.relevance_min_score <= 0:  # disabled by default: see settings.relevance_min_score
        return True
    return max(d.get("rerank_score", 0.0) for d in docs) >= settings.relevance_min_score


def is_answerable(query: str, docs: list[dict]) -> bool:
    """True if `docs` look able to answer `query`. Fails open if the signal is unavailable."""
    if not docs:
        return False
    if not settings.relevance_gate:
        return True

    reranker = docs[0].get("reranker")
    if reranker == "jev" and settings.openrouter_api_key:
        try:
            p = jev_answerability(query, docs)
            logfire.info(f"Relevance gate (jev): P(answerable)={p:.3f} min={settings.jev_answerable_min}")
            return p >= settings.jev_answerable_min
        except Exception as e:
            logfire.warn(f"Jev answerability failed ({e!r}); letting results through")
            return True

    if reranker == "jev":  # relative scores, no key: nothing reliable to gate on
        return True
    ok = _score_floor(docs)
    logfire.info(f"Relevance gate ({reranker}): best score floor ok={ok}")
    return ok
