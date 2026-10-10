from types import SimpleNamespace

from fastapi.testclient import TestClient

from config.settings import settings


def test_retrieval_falls_back_to_rest_when_grpc_fails(monkeypatch):
    from retrieval import engine

    calls = []

    def fake_search(client, query, content_type):
        calls.append(settings.qdrant_prefer_grpc)
        if settings.qdrant_prefer_grpc:
            raise RuntimeError("StatusCode.UNAVAILABLE: failed to connect to port 6334")
        return [], []

    monkeypatch.setattr(settings, "qdrant_prefer_grpc", True)
    monkeypatch.setattr(engine, "get_qdrant", lambda *a, **k: object())
    monkeypatch.setattr(engine, "_search_both", fake_search)
    assert engine.retrieve("anything") == []
    assert calls == [True, False] and settings.qdrant_prefer_grpc is False


def test_retrieval_does_not_mask_errors_on_rest(monkeypatch):
    import pytest

    from retrieval import engine

    monkeypatch.setattr(settings, "qdrant_prefer_grpc", False)
    monkeypatch.setattr(engine, "get_qdrant", lambda *a, **k: object())
    monkeypatch.setattr(engine, "_search_both", lambda *a: (_ for _ in ()).throw(RuntimeError("down")))
    with pytest.raises(RuntimeError):
        engine.retrieve("anything")


def _client(monkeypatch, points, jev):
    from api.main import create_app
    from api.routes import health
    from retrieval import warmup

    fake = SimpleNamespace(get_collection=lambda name: SimpleNamespace(points_count=points))
    monkeypatch.setattr(health, "get_qdrant_client", lambda: fake)
    monkeypatch.setattr(warmup, "jev_status", lambda: jev)
    return TestClient(create_app())


def test_health_ready_reports_collection_and_jev(monkeypatch):
    j = _client(monkeypatch, 36637, "ok").get("/health/ready").json()
    assert j["status"] == "ready" and j["qdrant"] == "connected" and j["points"] == 36637
    assert j["collection"] == settings.qdrant_collection and j["jev"] == "ok"


def test_health_ready_degraded_when_empty_or_jev_down(monkeypatch):
    assert _client(monkeypatch, 0, "ok").get("/health/ready").json()["status"] == "degraded"
    assert _client(monkeypatch, 10, "unavailable").get("/health/ready").json()["status"] == "degraded"
    assert _client(monkeypatch, 10, "disabled").get("/health/ready").json()["status"] == "ready"
