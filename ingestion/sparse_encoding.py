"""Shared BM25-style sparse encoding for indexing and querying.

Both sides must use the same vocabulary mapping: a term's index is a stable hash
(crc32, unlike Python's per-process salted hash()). Documents store the BM25 term-frequency
component; Qdrant applies IDF at query time (collection created with Modifier.IDF); queries
use weight 1.0 per distinct term, so the dot product is the BM25 score.
"""
from __future__ import annotations

import re
import zlib
from collections import Counter

_TOKEN = re.compile(r"\w+", re.UNICODE)
K1 = 1.2
B = 0.75
AVG_DOC_LEN = 90.0  # approx tokens in a 512-char chunk


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def term_index(token: str) -> int:
    return zlib.crc32(token.encode("utf-8"))  # fits Qdrant's u32 sparse index


def encode_document(text: str) -> tuple[list[int], list[float]]:
    tokens = tokenize(text)
    if not tokens:
        return [], []
    norm = K1 * (1 - B + B * len(tokens) / AVG_DOC_LEN)
    weights: dict[int, float] = {}
    for tok, tf in Counter(tokens).items():
        idx = term_index(tok)
        weights[idx] = weights.get(idx, 0.0) + tf * (K1 + 1) / (tf + norm)
    return list(weights), list(weights.values())


def encode_query(text: str) -> tuple[list[int], list[float]]:
    indices = sorted({term_index(t) for t in tokenize(text)})
    return indices, [1.0] * len(indices)
