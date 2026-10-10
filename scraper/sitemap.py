from __future__ import annotations

import asyncio
import re

import logfire
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

WISDOM_BASE = "https://isha.sadhguru.org/en/wisdom"

# All content type listing pages
LISTING_URLS = [
    f"{WISDOM_BASE}/type/article",
    f"{WISDOM_BASE}/type/video",
    f"{WISDOM_BASE}/type/quotes",
    f"{WISDOM_BASE}/type/poem",
    f"{WISDOM_BASE}/type/audio",
]

# JS to scroll down and click "Load More" repeatedly
SCROLL_AND_LOAD_JS = """
async () => {
    const delay = ms => new Promise(r => setTimeout(r, ms));
    let lastHeight = 0;
    let attempts = 0;
    const maxAttempts = %d;

    while (attempts < maxAttempts) {
        // Scroll to bottom
        window.scrollTo(0, document.body.scrollHeight);
        await delay(1500);

        // Click any "Load More" / "Show More" buttons
        const buttons = document.querySelectorAll(
            'button, a.load-more, [class*="load-more"], [class*="show-more"], [data-action="load-more"]'
        );
        for (const btn of buttons) {
            const text = btn.textContent.toLowerCase().trim();
            if (text.includes('load more') || text.includes('show more') || text.includes('view more')) {
                btn.click();
                await delay(2000);
            }
        }

        const newHeight = document.body.scrollHeight;
        if (newHeight === lastHeight) {
            attempts++;
        } else {
            attempts = 0;
        }
        lastHeight = newHeight;
    }
}
"""


async def discover_urls(max_pages: int = 500, scroll_rounds: int = 30) -> list[str]:
    """Crawl listing pages with JS scroll to discover all article URLs."""
    all_urls: list[str] = []
    browser_config = BrowserConfig(headless=True)

    js_code = SCROLL_AND_LOAD_JS % scroll_rounds

    crawl_config = CrawlerRunConfig(
        wait_until="networkidle",
        page_timeout=120000,
        js_code=js_code,
    )

    with logfire.span("discover_urls", max_pages=max_pages):
        async with AsyncWebCrawler(config=browser_config) as crawler:
            for listing_url in LISTING_URLS:
                with logfire.span("crawl_listing", url=listing_url):
                    print(f"  Scrolling {listing_url} ...")
                    try:
                        result = await crawler.arun(url=listing_url, config=crawl_config)
                        if result.success and result.html:
                            urls = _extract_article_urls(result.html)
                            new_count = len([u for u in urls if u not in all_urls])
                            all_urls.extend(u for u in urls if u not in all_urls)
                            print(f"    Found {len(urls)} links ({new_count} new) from {listing_url}")
                            logfire.info(f"Found {len(urls)} URLs from {listing_url}")
                        else:
                            print(f"    Failed: success={result.success}")
                    except Exception as e:
                        print(f"    Error: {e}")

                if len(all_urls) >= max_pages:
                    break

    unique = list(dict.fromkeys(all_urls))[:max_pages]
    print(f"  Total unique URLs discovered: {len(unique)}")
    logfire.info(f"Total unique URLs discovered: {len(unique)}")
    return unique


async def discover_urls_fast(max_pages: int = 500) -> list[str]:
    """
    Multi-phase fast discovery using httpx — no browser needed.
    Phase 1: Crawl listing pages for initial seeds
    Phase 2: Follow related links from each content page (snowball crawl)
    Falls back to browser-based discovery if too few found.
    """
    import httpx

    all_urls: set[str] = set()
    visited: set[str] = set()
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    # Phase 1: Seed from listing pages
    print("  Phase 1: Listing pages...")
    to_visit = list(LISTING_URLS) + [f"{WISDOM_BASE}/topics"]

    async with httpx.AsyncClient(follow_redirects=True, timeout=15, headers=headers) as client:
        for page_url in to_visit:
            if page_url in visited:
                continue
            visited.add(page_url)

            try:
                r = await client.get(page_url)
                if r.status_code != 200:
                    continue

                links = _extract_article_urls(r.text)
                new = [l for l in links if l not in all_urls]
                all_urls.update(new)
                if new:
                    print(f"    {page_url.split('/')[-1]}: +{len(new)} links (total: {len(all_urls)})")

                # Find topic sub-pages
                topic_links = re.findall(r'href="(/en/wisdom/topic/[^"]+)"', r.text)
                for tl in topic_links:
                    full = f"https://isha.sadhguru.org{tl}"
                    if full not in visited:
                        to_visit.append(full)

            except Exception:
                pass

            if len(all_urls) >= max_pages:
                break

        # Phase 2: Snowball — visit each discovered page to find related links
        print(f"  Phase 2: Snowball crawl from {len(all_urls)} seeds...")
        snowball_queue = list(all_urls)
        round_num = 0

        while snowball_queue and len(all_urls) < max_pages:
            round_num += 1
            batch = snowball_queue[:20]
            snowball_queue = snowball_queue[20:]
            new_this_round = 0

            tasks = []
            for url in batch:
                if url in visited:
                    continue
                visited.add(url)
                tasks.append(client.get(url))

            try:
                responses = await asyncio.gather(*tasks, return_exceptions=True)
            except Exception:
                continue

            for resp in responses:
                if isinstance(resp, Exception):
                    continue
                if resp.status_code != 200:
                    continue
                links = _extract_article_urls(resp.text)
                for link in links:
                    if link not in all_urls:
                        all_urls.add(link)
                        snowball_queue.append(link)
                        new_this_round += 1

            if new_this_round:
                print(f"    Round {round_num}: +{new_this_round} new (total: {len(all_urls)})")

            if new_this_round == 0 and not snowball_queue:
                break

    result = sorted(all_urls)[:max_pages]
    print(f"  Discovery complete: {len(result)} URLs")

    # If too few, fall back to browser-based scroll discovery
    if len(result) < 30:
        print("  Too few URLs from fast mode, falling back to browser scroll...")
        result = await discover_urls(max_pages=max_pages)

    return result


SITEMAP_BASE = "https://isha.sadhguru.org/english-wisdom-{kind}-sitemap.xml"
# The "article" sitemap also lists /sadhguru-spot/ and /poem/ pages
SITEMAP_KINDS = ("article", "video", "audio")


async def discover_urls_sitemap(kinds: tuple[str, ...] = SITEMAP_KINDS) -> list[str]:
    """Discover every content URL from the site's published sitemaps (no crawling needed)."""
    import httpx

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    urls: list[str] = []
    async with httpx.AsyncClient(follow_redirects=True, timeout=60, headers=headers) as client:
        for kind in kinds:
            r = await client.get(SITEMAP_BASE.format(kind=kind))
            r.raise_for_status()
            found = re.findall(r"<loc>([^<]+)</loc>", r.text)
            found = [u.split("#")[0].split("?")[0] for u in found if "/en/wisdom/" in u]
            print(f"  sitemap '{kind}': {len(found)} URLs")
            urls.extend(found)
    unique = list(dict.fromkeys(urls))
    print(f"  Total unique URLs from sitemaps: {len(unique)}")
    return unique


def _extract_article_urls(html: str) -> list[str]:
    """Extract wisdom content URLs from HTML."""
    pattern = r'href="(/en/wisdom/(?:article|video|quotes|poem|audio)/[^"]+)"'
    matches = re.findall(pattern, html)
    full_urls = []
    for m in matches:
        # Skip listing/type/topic pages, keep only content pages
        if "/type/" in m or "/topic/" in m:
            continue
        # Strip fragment identifiers (#) to avoid duplicates
        m = m.split("#")[0]
        url = f"https://isha.sadhguru.org{m}"
        if url not in full_urls:
            full_urls.append(url)
    return full_urls


if __name__ == "__main__":
    urls = asyncio.run(discover_urls_fast(max_pages=500))
    for u in urls:
        print(u)
