from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import logfire

from config.settings import settings
from ingestion.indexer import get_qdrant
from observability.instrumentation import trace
from retrieval.dense import dense_search
from retrieval.hybrid import reciprocal_rank_fusion
from retrieval.mmr import maximal_marginal_relevance
from retrieval.relevance import is_answerable
from retrieval.reranker import rerank
from retrieval.sparse import sparse_search


def _search_both(client, query: str, content_type: str | None) -> tuple[list[dict], list[dict]]:
    with ThreadPoolExecutor(max_workers=2) as pool:
        dense_future = pool.submit(dense_search, client, query, content_type=content_type)
        sparse_future = pool.submit(sparse_search, client, query)
        return dense_future.result(), sparse_future.result()


@trace("retrieval_engine")
def retrieve(
    query: str,
    content_type: str | None = None,
    use_reranker: bool = True,
    use_mmr: bool = True,
    use_relevance_gate: bool = True,
) -> list[dict]:
    """Full retrieval pipeline: dense + sparse -> RRF -> rerank -> MMR -> relevance gate."""
    client = get_qdrant()

    # Dense (embed -> search) and sparse (search only) are independent: run them concurrently
    with logfire.span("parallel_search"):
        try:
            dense_results, sparse_results = _search_both(client, query, content_type)
        except Exception as e:
            if not settings.qdrant_prefer_grpc:
                raise
            # gRPC (port 6334) is blocked on some hosts: drop to REST for the rest of the process
            logfire.warn(f"Qdrant gRPC search failed ({e!r}); falling back to REST")
            settings.qdrant_prefer_grpc = False
            dense_results, sparse_results = _search_both(get_qdrant(), query, content_type)

    logfire.info(f"Dense: {len(dense_results)}, Sparse: {len(sparse_results)} results")

    # Hybrid fusion
    fused = reciprocal_rank_fusion(dense_results, sparse_results)
    logfire.info(f"RRF fused: {len(fused)} results")

    if not fused:
        return []

    # Reranking
    if use_reranker:
        reranked = rerank(query, fused)
        logfire.info(f"Reranked: {len(reranked)} results")
    else:
        reranked = fused[: settings.rerank_top_k]

    # MMR diversity
    if use_mmr and len(reranked) > settings.final_top_k:
        final = maximal_marginal_relevance(query, reranked)
        logfire.info(f"MMR selected: {len(final)} results")
    else:
        final = reranked[: settings.final_top_k]

    # Relevance gate: better to say "not found" than to answer from unrelated chunks
    if use_reranker and use_relevance_gate and not is_answerable(query, final):
        logfire.info("Relevance gate: retrieved chunks cannot answer the query; returning none")
        return []

    return final
