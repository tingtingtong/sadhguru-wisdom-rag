"""Page-grounded retrieval evaluation.

Samples pages from the scraped corpus, has the LLM write a natural question answered by a passage
of each page, then checks whether retrieval returns that page. Ground truth is the source URL, so
no keyword lists or hand labelling are needed.

Usage:
    python -m evaluation.page_questions generate --n 60 --seed 7      # writes evaluation/page_questions.json
    python -m evaluation.page_questions run --providers flashrank jev # scores retrieval hit@k / MRR

Strict metric: another page may answer equally well, so hit@k is a conservative lower bound.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import time

os.environ.setdefault("LOGFIRE_IGNORE_NO_CONFIG", "1")

DATASET = "evaluation/page_questions.json"
SYSTEM = (
    "You write evaluation questions for a question-answering system about Sadhguru's teachings. "
    "Given a passage, write ONE natural question a curious person might ask that this passage answers. "
    "Do not mention 'the passage', 'the text' or the title, and do not copy long phrases. "
    "Reply with only the question."
)


def _norm(url: str) -> str:
    return url.split("#")[0].split("?")[0].rstrip("/")


def _sample_docs(n: int, seed: int) -> list:
    from scraper.storage import load_documents

    rng = random.Random(seed)
    docs = [d for d in load_documents(None) if len(d.content) >= 1200]
    by_type: dict[str, list] = {}
    for d in docs:
        by_type.setdefault("article" if d.content_type == "article" else d.content_type, []).append(d)
    quota = {"article": 0.7, "video_transcript": 0.3}
    picked = []
    for t, share in quota.items():
        pool = by_type.get(t, [])
        picked += rng.sample(pool, min(len(pool), round(n * share)))
    rng.shuffle(picked)
    return picked


async def generate(n: int, seed: int, append: bool = False) -> None:
    from generation.gateway import get_llm_provider

    rng = random.Random(seed)
    provider = get_llm_provider()
    items = json.load(open(DATASET, encoding="utf-8")) if append and os.path.exists(DATASET) else []
    have = {_norm(i["url"]) for i in items}
    for d in _sample_docs(n, seed):
        if _norm(d.url) in have:
            continue
        start = rng.randint(150, max(151, len(d.content) - 900))
        passage = d.content[start : start + 900]
        try:
            q = (await provider.generate(SYSTEM, passage)).strip().strip('"')
        except Exception as e:
            print("skip (LLM error):", repr(e)[:80])
            continue
        if 15 <= len(q) <= 300:
            items.append({"question": q, "url": d.url, "title": d.title, "content_type": d.content_type})
            print(f"  [{len(items)}] {q[:100]}")
    with open(DATASET, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)
    print(f"wrote {len(items)} questions to {DATASET}")


def run(providers: list[str], k: int, gate: bool, path: str = DATASET) -> list[dict]:
    from config.settings import settings
    from retrieval.engine import retrieve

    items = json.load(open(path, encoding="utf-8"))
    out = []
    for prov in providers:
        settings.reranker_provider = prov
        ranks, blocked, lat = [], 0, []
        for it in items:
            t = time.perf_counter()
            try:
                res = retrieve(it["question"], use_relevance_gate=gate)
            except Exception as e:
                print("  error:", repr(e)[:80])
                continue
            lat.append((time.perf_counter() - t) * 1000)
            if not res:
                blocked += 1
            urls = [_norm(r.get("url", "")) for r in res]
            ranks.append(urls.index(_norm(it["url"])) + 1 if _norm(it["url"]) in urls else None)
        n = len(ranks)
        row = {
            "provider": prov, "n": n, "gate": gate,
            f"hit@{k}": round(sum(r is not None and r <= k for r in ranks) / n, 3),
            "hit@1": round(sum(r == 1 for r in ranks) / n, 3),
            "mrr": round(sum(1 / r for r in ranks if r) / n, 3),
            "gate_blocked": blocked,
            "latency_ms_mean": round(statistics.mean(lat)),
        }
        print(json.dumps(row), flush=True)
        out.append(row)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--n", type=int, default=60)
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--append", action="store_true", help="add to the existing dataset (skips pages already covered)")
    r = sub.add_parser("run")
    r.add_argument("--providers", nargs="+", default=["flashrank", "jev"])
    r.add_argument("--k", type=int, default=5)
    r.add_argument("--gate", action="store_true", help="apply the relevance gate (blocked = miss)")
    r.add_argument("--dataset", default=DATASET)
    a = ap.parse_args()
    if a.cmd == "generate":
        asyncio.run(generate(a.n, a.seed, a.append))
    else:
        run(a.providers, a.k, a.gate, a.dataset)
