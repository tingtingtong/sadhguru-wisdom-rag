"""Chunk quality filter: drops navigation/boilerplate chunks and exact duplicates before indexing."""
from __future__ import annotations

import hashlib
import re

LINK = re.compile(r"\[[^\]]*\]\([^)]*\)")
MIN_CHUNK_CHARS = 80


def link_ratio(text: str) -> float:
    return sum(len(m) for m in LINK.findall(text)) / max(len(text), 1)


def classify_chunk(text: str) -> str | None:
    """Return why a chunk is boilerplate ('link_heavy' | 'nav_list' | 'too_short'), else None."""
    t = text.strip()
    n_links = len(LINK.findall(t))
    ratio = link_ratio(t)
    if ratio >= 0.5 or (n_links >= 5 and ratio >= 0.3):
        return "link_heavy"
    if n_links >= 3 and t.startswith(("* [", "- [", "[")):
        return "nav_list"
    if len(t) < MIN_CHUNK_CHARS:
        return "too_short"
    return None


def content_hash(text: str) -> str:
    return hashlib.md5(re.sub(r"\s+", " ", text).strip().lower().encode()).hexdigest()


def drop_shared_boilerplate_docs(docs: list, min_shared: int = 5, max_chars: int = 700) -> tuple[list, int]:
    """Drop short documents whose body text is shared by `min_shared`+ pages.

    Many audio/video pages carry only a generic blurb as their description (e.g. one sentence
    reused by 1,000+ podcast pages), so the only unique text is the title. Indexing them floods
    retrieval with near-identical, information-free chunks.
    """
    def key(d) -> str:
        parts = d.content.split("\n", 1)
        return content_hash(parts[1] if len(parts) > 1 else "")[:16]

    short = [d for d in docs if len(d.content) < max_chars]
    counts: dict[str, int] = {}
    for d in short:
        counts[key(d)] = counts.get(key(d), 0) + 1
    drop = {id(d) for d in short if counts[key(d)] >= min_shared}
    return [d for d in docs if id(d) not in drop], len(drop)


def filter_chunks(chunks: list[dict], seen: set[str] | None = None) -> tuple[list[dict], dict]:
    """Drop boilerplate and exact-duplicate chunks. Pass a shared `seen` set to dedupe across calls."""
    seen = set() if seen is None else seen
    kept: list[dict] = []
    stats = {"input": len(chunks), "link_heavy": 0, "nav_list": 0, "too_short": 0, "duplicate": 0}
    for c in chunks:
        kind = classify_chunk(c["text"])
        if kind:
            stats[kind] += 1
            continue
        h = content_hash(c["text"])
        if h in seen:
            stats["duplicate"] += 1
            continue
        seen.add(h)
        kept.append(c)
    stats["kept"] = len(kept)
    return kept, stats
