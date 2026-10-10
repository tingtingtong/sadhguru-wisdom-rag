import httpx
import pytest

from config.settings import settings
from retrieval import reranker

DOCS = [{"id": f"c{i}", "text": f"passage {i}"} for i in range(5)]


@pytest.fixture(autouse=True)
def _restore(monkeypatch):
    for name in ("reranker_provider", "openrouter_api_key", "jev_batch_size", "rerank_top_k"):
        monkeypatch.setattr(settings, name, getattr(settings, name))
    monkeypatch.setitem(reranker._SCORERS, "flashrank", lambda q, d: [0.1 * i for i in range(len(d))])


def test_empty_docs():
    assert reranker.rerank("q", []) == []


def test_unknown_provider(monkeypatch):
    monkeypatch.setattr(settings, "reranker_provider", "nope")
    with pytest.raises(ValueError):
        reranker.rerank("q", DOCS)


def test_flashrank_orders_and_truncates(monkeypatch):
    monkeypatch.setattr(settings, "reranker_provider", "flashrank")
    out = reranker.rerank("q", DOCS, top_k=2)
    assert [d["id"] for d in out] == ["c4", "c3"]
    assert out[0]["reranker"] == "flashrank" and out[0]["rerank_score"] == pytest.approx(0.4)


def test_jev_listwise(monkeypatch):
    monkeypatch.setattr(settings, "reranker_provider", "jev")
    monkeypatch.setattr(settings, "openrouter_api_key", "k")
    seen = {}

    def fake_post(self, url, json, headers):
        seen["payload"], seen["headers"] = json, headers
        probs = {"d0": 0.1, "d1": 0.6, "d2": 0.1, "d3": 0.1, "d4": 0.1}
        body = {"answers": {"best": {"type": "choice", "probabilities": probs}}}
        return httpx.Response(200, json=body, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    out = reranker.rerank("what is joy", DOCS, top_k=3)
    assert out[0]["id"] == "c1" and out[0]["reranker"] == "jev"
    assert seen["headers"]["Authorization"] == "Bearer k"
    assert set(seen["payload"]["questions"]["best"]["criteria"]) == {f"d{i}" for i in range(5)}


def test_jev_rounds_when_over_batch(monkeypatch):
    monkeypatch.setattr(settings, "reranker_provider", "jev")
    monkeypatch.setattr(settings, "openrouter_api_key", "k")
    monkeypatch.setattr(settings, "jev_batch_size", 2)
    monkeypatch.setattr(settings, "rerank_top_k", 1)
    calls = []

    def fake_choice(query, docs):
        calls.append(len(docs))
        # prefer the highest-numbered passage
        w = [int(d["text"].split()[1]) + 1 for d in docs]
        return [x / sum(w) for x in w]

    monkeypatch.setattr(reranker, "_jev_choice", fake_choice)
    out = reranker.rerank("q", DOCS, top_k=1)
    assert out[0]["id"] == "c4"
    assert len(calls) > 1


def test_jev_falls_back_without_key(monkeypatch):
    monkeypatch.setattr(settings, "reranker_provider", "jev")
    monkeypatch.setattr(settings, "openrouter_api_key", "")
    out = reranker.rerank("q", DOCS, top_k=1)
    assert out[0]["reranker"] == "flashrank"
