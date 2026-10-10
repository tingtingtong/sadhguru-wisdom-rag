"""Retrieval quality metrics: Context Precision, Recall, MRR, NDCG."""
from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass


@dataclass
class RetrievalMetrics:
    context_precision: float
    context_recall: float
    mrr: float
    ndcg: float
    hit_rate: float
    avg_rank_of_relevant: float
    total_retrieved: int
    relevant_retrieved: int


def _keyword_relevance(doc_text: str, keywords: list[str]) -> bool:
    """Check if a document is relevant based on keyword overlap."""
    text_lower = doc_text.lower()
    matches = sum(1 for kw in keywords if kw.lower() in text_lower)
    return matches >= max(1, len(keywords) // 3)


def context_precision(
    retrieved_docs: list[dict],
    relevant_keywords: list[str],
) -> float:
    """Fraction of retrieved docs that are relevant."""
    if not retrieved_docs or not relevant_keywords:
        return 0.0
    relevant_count = sum(
        1 for doc in retrieved_docs if _keyword_relevance(doc.get("text", ""), relevant_keywords)
    )
    return relevant_count / len(retrieved_docs)


def context_recall(
    retrieved_docs: list[dict],
    relevant_keywords: list[str],
) -> float:
    """Fraction of relevant keywords covered by at least one retrieved doc."""
    if not relevant_keywords:
        return 0.0
    all_text = " ".join(doc.get("text", "") for doc in retrieved_docs).lower()
    covered = sum(1 for kw in relevant_keywords if kw.lower() in all_text)
    return covered / len(relevant_keywords)


def mean_reciprocal_rank(
    retrieved_docs: list[dict],
    relevant_keywords: list[str],
) -> float:
    """MRR: reciprocal of the rank of the first relevant document."""
    for rank, doc in enumerate(retrieved_docs, 1):
        if _keyword_relevance(doc.get("text", ""), relevant_keywords):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved_docs: list[dict],
    relevant_keywords: list[str],
    k: int = 5,
) -> float:
    """Normalized Discounted Cumulative Gain at k."""
    docs = retrieved_docs[:k]
    if not docs or not relevant_keywords:
        return 0.0

    relevance_scores = []
    for doc in docs:
        text_lower = doc.get("text", "").lower()
        matches = sum(1 for kw in relevant_keywords if kw.lower() in text_lower)
        relevance_scores.append(min(matches, len(relevant_keywords)))

    # DCG
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevance_scores))

    # Ideal DCG
    ideal = sorted(relevance_scores, reverse=True)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))

    return dcg / idcg if idcg > 0 else 0.0


def compute_retrieval_metrics(
    retrieved_docs: list[dict],
    relevant_keywords: list[str],
) -> RetrievalMetrics:
    """Compute all retrieval metrics for a single query."""
    precision = context_precision(retrieved_docs, relevant_keywords)
    recall = context_recall(retrieved_docs, relevant_keywords)
    mrr = mean_reciprocal_rank(retrieved_docs, relevant_keywords)
    ndcg = ndcg_at_k(retrieved_docs, relevant_keywords)

    relevant_count = sum(
        1 for doc in retrieved_docs if _keyword_relevance(doc.get("text", ""), relevant_keywords)
    )

    # Average rank of relevant docs
    relevant_ranks = [
        rank
        for rank, doc in enumerate(retrieved_docs, 1)
        if _keyword_relevance(doc.get("text", ""), relevant_keywords)
    ]
    avg_rank = sum(relevant_ranks) / len(relevant_ranks) if relevant_ranks else 0.0

    return RetrievalMetrics(
        context_precision=round(precision, 4),
        context_recall=round(recall, 4),
        mrr=round(mrr, 4),
        ndcg=round(ndcg, 4),
        hit_rate=1.0 if relevant_count > 0 else 0.0,
        avg_rank_of_relevant=round(avg_rank, 2),
        total_retrieved=len(retrieved_docs),
        relevant_retrieved=relevant_count,
    )
