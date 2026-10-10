from __future__ import annotations

import asyncio

import logfire
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

from scraper.extractors import extract_article, extract_from_next_data
from scraper.models import ScrapedDocument


async def crawl_urls(urls: list[str], concurrency: int = 3) -> tuple[list[ScrapedDocument], list[str]]:
    """Crawl a list of URLs and extract documents. Returns (documents, failed_urls)."""
    documents: list[ScrapedDocument] = []
    failed: list[str] = []
    browser_config = BrowserConfig(headless=True)
    crawl_config = CrawlerRunConfig(
        wait_until="networkidle",
        page_timeout=30000,
    )

    semaphore = asyncio.Semaphore(concurrency)
    total = len(urls)
    completed = 0

    async with AsyncWebCrawler(config=browser_config) as crawler:

        async def _crawl_one(url: str) -> ScrapedDocument | None:
            nonlocal completed
            async with semaphore:
                with logfire.span("crawl_page", url=url):
                    try:
                        result = await crawler.arun(url=url, config=crawl_config)
                        completed += 1
                        if result.success:
                            doc = extract_article(
                                url=url,
                                markdown=result.markdown or "",
                                html=result.html or "",
                            )
                            if doc:
                                print(f"  [{completed}/{total}] OK: {doc.title[:60]}")
                                logfire.info(f"Extracted: {doc.title}")
                                return doc
                            else:
                                print(f"  [{completed}/{total}] SKIP (too short): {url}")
                        else:
                            print(f"  [{completed}/{total}] FAIL: {url}")
                    except Exception as e:
                        completed += 1
                        print(f"  [{completed}/{total}] ERROR: {url} - {e}")
                        logfire.error(f"Failed to crawl {url}: {e}")
                    return None

        tasks = [_crawl_one(url) for url in urls]
        results = await asyncio.gather(*tasks)

        for i, doc in enumerate(results):
            if doc:
                documents.append(doc)
            else:
                failed.append(urls[i])

    logfire.info(f"Crawled {len(documents)} documents from {len(urls)} URLs ({len(failed)} failed)")
    if failed:
        print(f"\n  Failed URLs ({len(failed)}):")
        for f_url in failed[:10]:
            print(f"    {f_url}")
        if len(failed) > 10:
            print(f"    ... and {len(failed) - 10} more")

    return documents, failed


async def fetch_urls(
    urls: list[str], concurrency: int = 6, retries: int = 2
) -> tuple[list[ScrapedDocument], list[str]]:
    """Fast path: plain HTTP + HTML->markdown (pages are server-rendered, no browser needed).

    ~1.3 s/page vs 10-17 s with a browser. Returns (documents, failed_urls); pages that load but
    have no usable body are counted as failed so they show up for review.
    """
    import httpx
    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    generator = DefaultMarkdownGenerator()
    semaphore = asyncio.Semaphore(concurrency)
    total, done = len(urls), 0
    documents: list[ScrapedDocument] = []
    failed: list[str] = []

    async with httpx.AsyncClient(headers=headers, timeout=30, follow_redirects=True) as client:

        async def _one(url: str) -> ScrapedDocument | None:
            nonlocal done
            async with semaphore:
                html = None
                for attempt in range(retries + 1):
                    try:
                        r = await client.get(url)
                        if r.status_code == 200:
                            html = r.text
                            break
                        if r.status_code in (404, 410):
                            break
                    except httpx.HTTPError:
                        pass
                    await asyncio.sleep(1.5 * (attempt + 1))
            done += 1
            if done % 100 == 0 or done == total:
                print(f"  [{done}/{total}] fetched", flush=True)
            if html is None:
                return None
            try:
                doc = extract_from_next_data(url, html)  # structured JSON: no menus / widgets / footer
                if doc:
                    return doc
                md = await asyncio.to_thread(generator.generate_markdown, input_html=html, base_url=url)
                return extract_article(url=url, markdown=getattr(md, "raw_markdown", str(md)), html=html)
            except Exception as e:  # one malformed page must not sink the batch
                logfire.warn(f"Extraction failed for {url}: {e!r}")
                return None

        results = await asyncio.gather(*[_one(u) for u in urls])

    for url, doc in zip(urls, results):
        if doc:
            documents.append(doc)
        else:
            failed.append(url)
    logfire.info(f"Fetched {len(documents)} documents from {len(urls)} URLs ({len(failed)} failed)")
    return documents, failed
