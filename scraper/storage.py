from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import yaml

from scraper.models import ScrapedDocument

DATA_DIR = Path(os.environ.get("SCRAPE_DATA_DIR", "data/raw"))
MANIFEST_PATH = DATA_DIR / "manifest.json"
FAILED_URLS_PATH = DATA_DIR / "failed_urls.json"


def save_document(doc: ScrapedDocument) -> Path:
    """Save document as markdown with YAML frontmatter."""
    type_dir = DATA_DIR / f"{doc.content_type}s"
    type_dir.mkdir(parents=True, exist_ok=True)

    slug = _slugify(doc.title)
    # URL hash keeps same-titled pages (e.g. an audio and an article) from overwriting each other
    url_hash = hashlib.md5(doc.url.split("#")[0].encode()).hexdigest()[:8]
    filepath = type_dir / f"{slug}-{url_hash}.md"

    frontmatter = {
        "title": doc.title,
        "url": doc.url,
        "content_type": doc.content_type,
        "topics": doc.topics,
        "date": doc.date,
    }

    content = f"---\n{yaml.dump(frontmatter, default_flow_style=False)}---\n\n{doc.content}"
    filepath.write_text(content, encoding="utf-8")
    return filepath


def load_documents(content_type: str | None = None) -> list[ScrapedDocument]:
    """Load saved markdown documents."""
    docs = []
    search_dir = DATA_DIR / f"{content_type}s" if content_type else DATA_DIR

    if not search_dir.exists():
        return docs

    for md_file in search_dir.rglob("*.md"):
        text = md_file.read_text(encoding="utf-8")
        doc = _parse_markdown(text)
        if doc:
            docs.append(doc)

    return docs


def _parse_markdown(text: str) -> ScrapedDocument | None:
    match = re.match(r"^---\n(.*?)\n---\n\n(.*)", text, re.DOTALL)
    if not match:
        return None

    try:
        meta = yaml.safe_load(match.group(1))
        content = match.group(2)
        return ScrapedDocument(
            title=meta.get("title", "Untitled"),
            url=meta.get("url", ""),
            content=content,
            content_type=meta.get("content_type", "article"),
            topics=meta.get("topics", []),
            date=meta.get("date", ""),
        )
    except Exception:
        return None


def load_manifest() -> dict[str, dict]:
    """Load the scrape manifest (URL -> metadata mapping)."""
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_manifest(manifest: dict[str, dict]) -> None:
    """Save the scrape manifest."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def update_manifest_batch(docs: list[tuple[ScrapedDocument, Path]]) -> None:
    """Add multiple scraped documents to the manifest in one write."""
    manifest = load_manifest()
    now = datetime.now(timezone.utc).isoformat()
    for doc, filepath in docs:
        manifest[doc.url] = {
            "title": doc.title,
            "file": str(filepath),
            "content_type": doc.content_type,
            "scraped_at": now,
        }
    save_manifest(manifest)


def get_scraped_urls() -> set[str]:
    """Return set of already-scraped URLs (normalized, no fragments)."""
    return {url.split("#")[0] for url in load_manifest().keys()}


def load_failed_urls() -> list[str]:
    """Load list of previously failed URLs."""
    if FAILED_URLS_PATH.exists():
        with open(FAILED_URLS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_failed_urls(urls: list[str]) -> None:
    """Save list of failed URLs for retry."""
    FAILED_URLS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FAILED_URLS_PATH, "w", encoding="utf-8") as f:
        json.dump(urls, f, indent=2, ensure_ascii=False)


def _slugify(title: str) -> str:
    slug = title.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    return slug[:80]
