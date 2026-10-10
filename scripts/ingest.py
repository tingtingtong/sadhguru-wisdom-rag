"""CLI: Ingest scraped documents into Qdrant.

Usage: python scripts/ingest.py [--type article]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.pipeline import run_ingestion


def main(content_type: str | None) -> None:
    print(f"Starting ingestion pipeline (type={content_type or 'all'})...")
    count = run_ingestion(content_type)
    print(f"Done! {count} chunks indexed into Qdrant.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into Qdrant")
    parser.add_argument("--type", default=None, choices=["article", "quote", "poem"])
    args = parser.parse_args()

    main(args.type)
