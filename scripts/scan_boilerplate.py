"""Read-only scan of the Qdrant collection for navigation/boilerplate chunks and duplicates.

Usage: python -m scripts.scan_boilerplate [--out evaluation/boilerplate_scan.json]
Never writes to Qdrant.
"""
from __future__ import annotations

import argparse
import collections
import json

from config.settings import settings
from ingestion.indexer import get_qdrant
from ingestion.quality import classify_chunk as classify, content_hash



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="evaluation/boilerplate_scan.json")
    args = ap.parse_args()

    client = get_qdrant()
    total = client.get_collection(settings.qdrant_collection).points_count
    kinds = collections.Counter()
    by_type = collections.defaultdict(collections.Counter)
    hashes = collections.Counter()
    bad_ids, examples = [], {}
    seen, off = 0, None
    while True:
        batch, off = client.scroll(
            settings.qdrant_collection, limit=1000, offset=off,
            with_payload=True, with_vectors=False,
        )
        for p in batch:
            text = p.payload.get("text", "")
            ctype = p.payload.get("content_type", "?")
            kind = classify(text)
            hashes[content_hash(text)] += 1
            by_type[ctype]["total"] += 1
            if kind:
                kinds[kind] += 1
                by_type[ctype][kind] += 1
                bad_ids.append(str(p.id))
                examples.setdefault(kind, text[:200])
        seen += len(batch)
        print(f"scanned {seen}/{total}", flush=True)
        if off is None:
            break

    dup_extra = sum(c - 1 for c in hashes.values() if c > 1)
    flagged = len(bad_ids)
    report = {
        "total_points": seen,
        "flagged_boilerplate": flagged,
        "flagged_pct": round(100 * flagged / seen, 1),
        "by_kind": dict(kinds),
        "by_content_type": {k: dict(v) for k, v in by_type.items()},
        "exact_duplicate_extra_copies": dup_extra,
        "distinct_texts": len(hashes),
        "top_duplicated_texts": sorted(hashes.values(), reverse=True)[:5],
        "examples": examples,
        "flagged_ids": bad_ids,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "flagged_ids"}, indent=1))


if __name__ == "__main__":
    main()
