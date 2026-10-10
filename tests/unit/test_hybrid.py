from retrieval.hybrid import reciprocal_rank_fusion


def test_rrf_basic():
    dense = [
        {"id": "a", "score": 0.9, "text": "doc a"},
        {"id": "b", "score": 0.8, "text": "doc b"},
    ]
    sparse = [
        {"id": "b", "score": 0.7, "text": "doc b"},
        {"id": "c", "score": 0.6, "text": "doc c"},
    ]
    results = reciprocal_rank_fusion(dense, sparse, k=60)
    assert len(results) == 3
    # b should be highest since it appears in both
    assert results[0]["id"] == "b"
    assert "rrf_score" in results[0]


def test_rrf_empty():
    results = reciprocal_rank_fusion([], [])
    assert results == []
