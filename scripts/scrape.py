"""CLI: Scrape Sadhguru wisdom content from isha.sadhguru.org.

Usage:
    python scripts/scrape.py                          # Incremental, all types, up to 500 pages
    python scripts/scrape.py --max-pages 100           # Limit pages
    python scripts/scrape.py --mode browser            # Use browser scroll (slower, finds more)
    python scripts/scrape.py --full                    # Re-scrape everything
    python scripts/scrape.py --retry-failed            # Only retry previously failed URLs
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scraper.crawler import crawl_urls, fetch_urls
from scraper.sitemap import discover_urls, discover_urls_fast, discover_urls_sitemap
from scraper.storage import (
    get_scraped_urls,
    load_failed_urls,
    save_document,
    save_failed_urls,
    update_manifest_batch,
)

from scraper.storage import DATA_DIR

FAILED_URLS_PATH = DATA_DIR / "failed_urls.json"


async def main(
    max_pages: int,
    mode: str,
    scroll_rounds: int,
    concurrency: int,
    full: bool = False,
    retry_failed: bool = False,
    types: tuple[str, ...] = ("article", "video", "audio"),
) -> None:
    print(f"=== Sadhguru Wisdom Scraper ===")
    incremental = not full
    print(f"Mode: {mode} | Max pages: {max_pages} | Concurrency: {concurrency} | Incremental: {incremental}")
    print()

    if retry_failed:
        # Only retry previously failed URLs
        urls = load_failed_urls()
        if not urls:
            print("No failed URLs to retry.")
            return
        print(f"Retrying {len(urls)} previously failed URLs...")
    else:
        # Phase 1: Discover URLs
        print("[1/3] Discovering URLs...")
        if mode == "sitemap":
            urls = await discover_urls_sitemap(types)
            urls = urls[:max_pages]
        elif mode == "fast":
            urls = await discover_urls_fast(max_pages=max_pages)
        else:
            urls = await discover_urls(max_pages=max_pages, scroll_rounds=scroll_rounds)

        if not urls:
            print("No URLs found. The site may require different crawling approach.")
            return

        # Filter out already-scraped URLs in incremental mode
        if incremental:
            already_scraped = get_scraped_urls()
            new_urls = [u for u in urls if u not in already_scraped]
            skipped = len(urls) - len(new_urls)
            print(f"  Already scraped: {skipped} | New to scrape: {len(new_urls)}")
            urls = new_urls

        if not urls:
            print("\nAll URLs already scraped. Nothing to do.")
            print("Use --full to re-scrape everything, or --retry-failed to retry failed URLs.")
            return

    print(f"\n[2/3] Crawling {len(urls)} pages (concurrency={concurrency})...")
    # Fetch + save in batches so a crash or interruption never loses more than one batch
    batch_size = 400 if mode == "sitemap" else len(urls)
    documents, failed_urls = [], []
    for start in range(0, len(urls), batch_size):
        batch = urls[start : start + batch_size]
        if mode == "sitemap":
            docs, failed = await fetch_urls(batch, concurrency=concurrency)
        else:
            docs, failed = await crawl_urls(batch, concurrency=concurrency)
        pairs = [(doc, save_document(doc)) for doc in docs]
        if pairs:
            update_manifest_batch(pairs)
        documents.extend(docs)
        failed_urls.extend(failed)
        print(f"  batch {start // batch_size + 1}: saved {len(docs)}, failed {len(failed)} "
              f"(running total {len(documents)} saved / {len(failed_urls)} failed)", flush=True)
    print(f"Extracted {len(documents)} documents, {len(failed_urls)} failed")

    # Save failed URLs for retry
    if failed_urls:
        # Merge with existing failures (remove any that succeeded this time)
        succeeded_urls = {doc.url for doc in documents}
        existing_failed = set(load_failed_urls())
        all_failed = (existing_failed | set(failed_urls)) - succeeded_urls
        save_failed_urls(sorted(all_failed))
        print(f"\n  {len(all_failed)} failed URLs saved to {FAILED_URLS_PATH}")
        print(f"  Run with --retry-failed to retry them")
    else:
        # Clear failed list if all succeeded
        save_failed_urls([])

    # Also save as JSON for easy re-indexing
    json_path = DATA_DIR / "corpus.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)

    # In incremental mode, merge with existing corpus
    existing_docs = []
    if incremental and json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            existing_docs = json.load(f)

    existing_urls = {d["url"] for d in existing_docs}
    for doc in documents:
        if doc.url not in existing_urls:
            existing_docs.append({
                "title": doc.title,
                "url": doc.url,
                "text": doc.content,
                "content_type": doc.content_type,
                "topics": doc.topics,
            })

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(existing_docs, f, indent=2, ensure_ascii=False)

    print(f"\n=== Done! ===")
    print(f"  {len(documents)} new documents scraped")
    print(f"  {len(existing_docs)} total documents in corpus")
    print(f"  JSON corpus: {json_path}")
    if documents:
        print(f"  Content types: {dict((t, sum(1 for d in documents if d.content_type == t)) for t in set(d.content_type for d in documents))}")
    if failed_urls:
        print(f"  Failed: {len(failed_urls)} URLs (use --retry-failed)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape Sadhguru wisdom content")
    parser.add_argument("--max-pages", type=int, default=500, help="Max pages to scrape")
    parser.add_argument("--mode", choices=["fast", "browser", "sitemap"], default="browser",
                        help="fast=httpx snowball, browser=scroll+JS rendering, sitemap=all URLs from sitemaps via plain HTTP")
    parser.add_argument("--types", default="article,video,audio",
                        help="sitemap mode: comma list of article,video,audio")
    parser.add_argument("--scroll-rounds", type=int, default=30,
                        help="Scroll attempts per listing page (browser mode)")
    parser.add_argument("--concurrency", type=int, default=3,
                        help="Parallel browser tabs for crawling")
    parser.add_argument("--full", action="store_true",
                        help="Re-scrape everything (ignore manifest). Default is incremental.")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Only retry previously failed URLs.")
    args = parser.parse_args()

    asyncio.run(main(args.max_pages, args.mode, args.scroll_rounds, args.concurrency, args.full, args.retry_failed,
                     tuple(t.strip() for t in args.types.split(","))))
