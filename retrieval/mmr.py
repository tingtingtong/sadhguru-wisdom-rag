from __future__ import annotations

import math

from config.settings import settings
from ingestion.sparse_encoding import tokenize
from observability.instrumentation import trace

# Words shorter than this carry little topical signal; keeps MMR's similarity about content.
_MIN_TOKEN = 4
_STOP = frozenset(
    "that this with from have your what when where which about there their would could should "
    "they them then than been being were also into more some such only other these those just very".split()
)


def _term_vector(text: str) -> dict[str, float]:
    vec: dict[str, float] = {}
    for tok in tokenize(text):
        if len(tok) >= _MIN_TOKEN and tok not in _STOP:
            vec[tok] = vec.get(tok, 0.0) + 1.0
    return vec


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = sum(w * b.get(t, 0.0) for t, w in a.items())
    na = math.sqrt(sum(w * w for w in a.values()))
    nb = math.sqrt(sum(w * w for w in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def _relevance(docs: list[dict]) -> list[float]:
    """Min-max normalised retrieval/rerank score per document."""
    raw = [d.get("rerank_score", d.get("rrf_score", d.get("score", 0.0))) or 0.0 for d in docs]
    lo, hi = min(raw), max(raw)
    return [(r - lo) / (hi - lo) if hi > lo else 1.0 for r in raw]


@trace("mmr_diversity")
def maximal_marginal_relevance(
    query: str,
    documents: list[dict],
    top_k: int | None = None,
    lambda_mult: float | None = None,
) -> list[dict]:
    """Select relevant but non-redundant documents (MMR).

    Relevance is the reranker's score; redundancy is lexical cosine similarity between chunks.
    It runs locally: the previous version re-embedded every candidate with OpenAI (~0.6 s/query)
    even though the chunks were already embedded in the index.
    """
    top_k = top_k or settings.final_top_k
    lambda_mult = lambda_mult or settings.mmr_lambda

    if len(documents) <= top_k:
        return documents

    relevance = _relevance(documents)
    vectors = [_term_vector(d["text"]) for d in documents]

    selected: list[int] = []
    remaining = list(range(len(documents)))
    while remaining and len(selected) < top_k:
        best, best_score = remaining[0], -math.inf
        for i in remaining:
            redundancy = max((_cosine(vectors[i], vectors[j]) for j in selected), default=0.0)
            score = lambda_mult * relevance[i] - (1 - lambda_mult) * redundancy
            if score > best_score:
                best, best_score = i, score
        selected.append(best)
        remaining.remove(best)

    return [documents[i] for i in selected]
