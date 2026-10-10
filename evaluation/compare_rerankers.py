"""Compare reranker providers (flashrank / jev / bge) on the on-topic retrieval cases.

Usage: python -m evaluation.compare_rerankers [--providers flashrank jev bge] [--no-mmr]
Reuses the keyword-based retrieval metrics from evaluation.metrics.retrieval.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import asdict

os.environ.setdefault("LOGFIRE_IGNORE_NO_CONFIG", "1")

from config.settings import settings  # noqa: E402
from evaluation.dataset import get_on_topic_cases  # noqa: E402
from evaluation.metrics.retrieval import compute_retrieval_metrics  # noqa: E402
from retrieval.engine import retrieve  # noqa: E402

METRICS = ["context_precision", "context_recall", "mrr", "ndcg", "hit_rate"]


def run_provider(provider: str, use_mmr: bool) -> dict:
    settings.reranker_provider = provider
    rows, latencies, fallbacks = [], [], 0
    for case in get_on_topic_cases():
        start = time.perf_counter()
        try:
            results = retrieve(case.question, use_reranker=True, use_mmr=use_mmr, use_relevance_gate=False)
        except Exception as e:
            rows.append({"id": case.id, "error": str(e)})
            continue
        latencies.append((time.perf_counter() - start) * 1000)
        used = {d.get("reranker") for d in results}
        if used and used != {provider}:
            fallbacks += 1
        m = compute_retrieval_metrics(results, case.relevant_keywords)
        rows.append({"id": case.id, "metrics": asdict(m), "reranker_used": sorted(x for x in used if x)})
    ok = [r for r in rows if "metrics" in r]
    agg = {k: round(statistics.mean(r["metrics"][k] for r in ok), 4) for k in METRICS} if ok else {}
    return {
        "provider": provider, "cases": len(ok), "errors": len(rows) - len(ok),
        "fallback_cases": fallbacks, "aggregate": agg,
        "latency_ms_mean": round(statistics.mean(latencies), 1) if latencies else None,
        "latency_ms_p95": round(sorted(latencies)[int(len(latencies) * 0.95) - 1], 1) if latencies else None,
        "rows": rows,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--providers", nargs="+", default=["flashrank", "jev", "bge"])
    ap.add_argument("--no-mmr", action="store_true")
    ap.add_argument("--output", default="evaluation/reranker_comparison.json")
    args = ap.parse_args()

    results = []
    for p in args.providers:
        print(f"== {p} ==", flush=True)
        results.append(run_provider(p, use_mmr=not args.no_mmr))
        r = results[-1]
        print(json.dumps({k: v for k, v in r.items() if k != "rows"}), flush=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nprovider   " + " ".join(f"{m[:9]:>9}" for m in METRICS) + "  mean_ms  fallbacks")
    for r in results:
        a = r["aggregate"]
        print(f"{r['provider']:<10} " + " ".join(f"{a.get(m, 0):>9.3f}" for m in METRICS)
              + f" {r['latency_ms_mean']:>8} {r['fallback_cases']:>9}")
