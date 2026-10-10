from __future__ import annotations

import re

from ingestion.quality import link_ratio
from scraper.models import ScrapedDocument


_NEXT_DATA = re.compile(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.S)
_SHORTCODE = re.compile(
    r"\[/?(?:SadhguruImage|SadhguruSignature|powerpress|poem|caption|embed|video|audio|gallery|youtube)\b[^\]]*\]", re.I
)
_POEM_TITLE = re.compile(r'\[poem title="([^"]*)"\]', re.I)


def _field(value) -> str:
    """Next.js fields come as [{'value': ...}] lists, bare strings, or plain values."""
    if isinstance(value, list) and value and isinstance(value[0], dict):
        value = value[0].get("value", "")
    return value if isinstance(value, str) else ""


def _html_to_text(fragment: str) -> str:
    """Body HTML -> paragraphs of plain text (links keep their anchor text; shortcodes dropped)."""
    from bs4 import BeautifulSoup

    fragment = _POEM_TITLE.sub("", fragment)
    fragment = _SHORTCODE.sub("", fragment)
    soup = BeautifulSoup(fragment, "html.parser")
    for br in soup.find_all("br"):
        br.replace_with("\n")
    blocks = [p.get_text() for p in soup.find_all(["p", "li", "h1", "h2", "h3", "h4", "blockquote"])]
    text = "\n".join(blocks) if blocks else soup.get_text()
    lines = [re.sub(r"[ \t ]+", " ", ln).strip() for ln in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def extract_from_next_data(url: str, html: str) -> ScrapedDocument | None:
    """Preferred extractor: read the page's structured JSON (title, body, topics, date).

    Bypasses site menus, the newsletter widget and footers entirely. Returns None when the page
    has no usable body so the caller can fall back to the markdown path.
    """
    import html as htmllib
    import json

    m = _NEXT_DATA.search(html)
    if not m:
        return None
    try:
        page = json.loads(m.group(1))["props"]["pageProps"]["pageData"]
    except (ValueError, KeyError, TypeError):
        return None
    if not isinstance(page, dict):  # some URLs (redirects, removed pages) have pageData = null
        return None

    title = htmllib.unescape(_field(page.get("title"))).strip()
    body_html = "\n".join(_field([b]) for b in (page.get("body") or []) if isinstance(b, dict))
    body = _html_to_text(body_html)
    summary = _html_to_text(_field(page.get("summary")))
    # Many video/audio pages are just a YouTube/podcast embed with a one-line summary: keep those
    # as short title+summary entries (useful for pointing at the right media), but nothing thinner.
    if not title or len(body) + (len(summary) if summary[:80] not in body else 0) < 60:
        return None

    parts = [f"# {title}"]
    if summary and summary[:80] not in body:
        parts.append(summary)
    if body:
        parts.append(body)
    note = _html_to_text(_field(page.get("editorsNote")))
    if note:
        parts.append(note)
    content = "\n\n".join(parts)

    topics: list[str] = []
    for entry in page.get("topics") or []:
        for t in (entry or {}).get("value") or []:
            if isinstance(t, dict) and t.get("title"):
                topics.append(t["title"])

    return ScrapedDocument(
        title=title,
        url=url,
        content=content,
        content_type=_detect_content_type(url),
        topics=list(dict.fromkeys(topics))[:10] or _extract_topics(content, title),
        date=_field(page.get("createdAt"))[:10],
        excerpt=summary[:300],
    )


def extract_article(url: str, markdown: str, html: str) -> ScrapedDocument | None:
    """Extract article content from crawled markdown."""
    if not markdown or len(markdown.strip()) < 100:
        return None

    title = _extract_title(markdown, html)
    content = _clean_content(markdown)
    content_type = _detect_content_type(url)
    topics = _extract_topics(content, title)

    if len(content.strip()) < 100:
        return None

    return ScrapedDocument(
        title=title,
        url=url,
        content=content,
        content_type=content_type,
        topics=topics,
    )


def _extract_title(markdown: str, html: str) -> str:
    # Try HTML h1 first (most reliable)
    html_match = re.search(r"<h1[^>]*>([^<]+)</h1>", html, re.IGNORECASE)
    if html_match:
        title = html_match.group(1).strip()
        if len(title) > 5:
            return title

    # Try markdown h1
    md_match = re.search(r"^#\s+(.+)$", markdown, re.MULTILINE)
    if md_match:
        title = md_match.group(1).strip()
        if len(title) > 5 and not title.startswith(("[", "http", "*", "<")):
            return title

    # Try og:title or title tag
    og_match = re.search(r'<meta\s+property="og:title"\s+content="([^"]+)"', html, re.IGNORECASE)
    if og_match:
        return og_match.group(1).strip()

    title_match = re.search(r"<title>([^<]+)</title>", html, re.IGNORECASE)
    if title_match:
        title = title_match.group(1).strip()
        # Remove site suffix
        title = re.sub(r"\s*[|–-]\s*(?:Sadhguru|Isha).*$", "", title)
        if title:
            return title

    # Fallback: first non-nav line
    lines = markdown.strip().split("\n")
    for line in lines[:10]:
        line = line.strip()
        if len(line) > 10 and not line.startswith(("http", "![", "[", "<", "*", "Subscribe")):
            return line[:200]

    return "Untitled"


# Bullet / link-only lines, e.g. "* [Text](url)" or "[![](img) Volunteer](url)" (site menus)
_LINK_LINE = re.compile(r"^\s*(?:[*+\-]\s+)?(?:!?\[.*?\]\([^)]*\)\s*)+$")

_H1 = re.compile(r"^#\s+\S")
# Where the article body ends and the related-content / newsletter / contact footer begins
_FOOTER = re.compile(r"^\s*(?:Related Tags|Related Content)\s*$|^\s*#{1,6}\s*Subscribe\b|©\s*1999")


def trim_page_chrome(markdown: str) -> str:
    """Keep only the page body: from the first '# Title' heading to the related/footer block.

    isha.sadhguru.org pages put ~250 lines of site menu before the title and a related-content,
    newsletter and contact footer after the body. Pages without an H1 (poems) start after the
    first standalone 'Subscribe' line. Idempotent.
    """
    lines = markdown.split("\n")
    start = next((i for i, ln in enumerate(lines) if _H1.match(ln)), None)
    if start is None:
        start = next((i + 1 for i, ln in enumerate(lines) if ln.strip() == "Subscribe"), 0)
    end = next((i for i in range(start, len(lines)) if _FOOTER.search(lines[i])), len(lines))
    return "\n".join(ln for ln in lines[start:end] if ln.strip() != "Subscribe")


def _clean_content(markdown: str) -> str:
    lines = trim_page_chrome(markdown).split("\n")
    cleaned = []
    skip_patterns = [
        r"^\s*\[.*\]\(.*\)\s*$",  # standalone links
        r"^!\[.*\]\(.*\)$",  # images
        r"^\s*Share\s*$",
        r"^\s*Related\s+Articles?\s*$",
        r"^\s*Tags?\s*:",
        r"^\s*Subscribe\s*$",
        r"^\s*Sign\s+up\s*$",
        r"^\s*Follow\s+us\s*$",
        r"^\s*Get weekly updates on the latest blogs",  # newsletter widget ...
        r"^\s*Country(?:India|United States)",  # ... and its country dropdown
    ]

    for line in lines:
        if len(line) < 800 and _LINK_LINE.match(line):
            continue
        if any(re.match(p, line, re.IGNORECASE) for p in skip_patterns):
            continue
        if link_ratio(line) >= 0.6:
            continue
        cleaned.append(line)

    return "\n".join(cleaned).strip()


def _detect_content_type(url: str) -> str:
    if "/quotes/" in url or "/quote/" in url:
        return "quote"
    if "/poems/" in url or "/poem/" in url:
        return "poem"
    if "/video/" in url:
        return "video_transcript"
    if "/audio/" in url:
        return "audio"
    return "article"  # includes /article/ and /sadhguru-spot/


def _extract_topics(content: str, title: str) -> list[str]:
    from config.constants import SADHGURU_TOPICS

    text = f"{title} {content}".lower()
    found = []
    for topic in SADHGURU_TOPICS:
        if topic.lower() in text:
            found.append(topic)
    return found[:10]
