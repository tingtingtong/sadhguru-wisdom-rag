"""Delete superseded Qdrant collections. DRY RUN unless --confirm is given.

Usage:
    python -m scripts.delete_collections                       # list what would be deleted
    python -m scripts.delete_collections --confirm             # delete sadhguru_wisdom and sadhguru_wisdom_v2

Run this only AFTER every consumer (local .env, Streamlit Cloud secrets, Docker env) points at the
current collection (settings.qdrant_collection): deletion is irreversible and a deployed app that
still points at an old collection will start failing immediately. Refuses to delete the active one.
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("LOGFIRE_IGNORE_NO_CONFIG", "1")

SUPERSEDED = ["sadhguru_wisdom", "sadhguru_wisdom_v2"]  # 68% menu boilerplate / early rebuild


def main() -> None:
    from config.settings import settings
    from ingestion.indexer import get_qdrant

    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", action="store_true", help="actually delete (default: dry run)")
    ap.add_argument("--names", nargs="*", default=SUPERSEDED)
    args = ap.parse_args()

    client = get_qdrant()
    existing = {c.name for c in client.get_collections().collections}
    active = settings.qdrant_collection
    print(f"active collection: {active} ({client.get_collection(active).points_count} points)")
    for name in args.names:
        if name == active:
            print(f"  {name}: REFUSING (it is the active collection)")
        elif name not in existing:
            print(f"  {name}: not found")
        else:
            n = client.get_collection(name).points_count
            if args.confirm:
                client.delete_collection(name)
                print(f"  {name}: DELETED ({n} points)")
            else:
                print(f"  {name}: would delete ({n} points)   [dry run; pass --confirm]")


if __name__ == "__main__":
    main()
