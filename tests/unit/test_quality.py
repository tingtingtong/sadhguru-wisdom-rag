import uuid

from ingestion.quality import classify_chunk, filter_chunks
from scraper.extractors import _clean_content

NAV = (
    "* [Ecstasy of Enlightenment with Sadhguru](https://isha.sadhguru.org/in/en/events/special-events/x)\n"
    "      * [anumArga - Project Samskriti US Tour](https://isha.sadhguru.org/in/en/events/y)\n"
    "      * [Saptarishi Arati](https://isha.sadhguru.org/in/en/events/z)"
)
PROSE = (
    "Meditation is not something you do; it is something you become. When the mind is "
    "still, a different dimension of life opens up, and you see that [Sadhguru](https://isha.sadhguru.org) "
    "has often called this inner engineering rather than belief."
)


def test_classify():
    assert classify_chunk(NAV) == "link_heavy"
    assert classify_chunk("* [Upcoming") == "too_short"
    assert classify_chunk("Get weekly updates on the latest blogs via newsletters right in your mailbox.") == "too_short"
    assert classify_chunk(PROSE) is None


def test_filter_chunks_drops_boilerplate_and_duplicates():
    chunks = [{"text": NAV}, {"text": PROSE}, {"text": PROSE.upper()}, {"text": "  " + PROSE + " "}]
    kept, stats = filter_chunks(chunks)
    assert len(kept) == 1 and kept[0]["text"] == PROSE
    assert stats["link_heavy"] == 1 and stats["duplicate"] == 2


def test_filter_chunks_dedupes_across_calls():
    seen: set[str] = set()
    assert len(filter_chunks([{"text": PROSE}], seen)[0]) == 1
    assert len(filter_chunks([{"text": PROSE}], seen)[0]) == 0


def test_clean_content_strips_menu_lines_keeps_prose():
    md = (
        "* [Man](https://isha.sadhguru.org/man)\n"
        "    * [![](https://static.sadhguru.org/i.svg) Volunteer](https://isha.sadhguru.org/volunteer)\n"
        "[Home](https://isha.sadhguru.org) [Shop](https://shop.example.com)\n"
        + PROSE
    )
    out = _clean_content(md)
    assert "Volunteer" not in out and "[Man]" not in out and "Shop" not in out
    assert "Meditation is not something you do" in out


def test_point_id_is_deterministic():
    ns = uuid.NAMESPACE_URL
    from ingestion.quality import content_hash
    a = uuid.uuid5(ns, f"u|0|{content_hash(PROSE)}")
    assert a == uuid.uuid5(ns, f"u|0|{content_hash('  ' + PROSE.upper())}")


def test_sparse_encoding_shared_vocabulary():
    from ingestion.sparse_encoding import encode_document, encode_query

    d_idx, d_val = encode_document("Karma is memory, the residual impact of past action.")
    q_idx, q_val = encode_query("what is karma memory")
    overlap = set(d_idx) & set(q_idx)
    assert len(overlap) == 3  # is, karma, memory
    assert encode_document("Karma is memory")[0] == encode_document("karma IS memory")[0]
    assert all(v > 0 for v in d_val) and q_val == [1.0] * len(q_idx)
    assert encode_document("") == ([], [])


def test_point_id_helper_matches_content():
    from ingestion.indexer import _point_id

    a = {"url": "u", "chunk_index": 3, "text": "Karma is memory."}
    assert _point_id(a) == _point_id({**a, "text": "  karma IS memory. "})
    assert _point_id(a) != _point_id({**a, "chunk_index": 4})
