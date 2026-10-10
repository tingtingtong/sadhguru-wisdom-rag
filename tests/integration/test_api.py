def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_topics(client):
    response = client.get("/topics")
    assert response.status_code == 200
    data = response.json()
    assert "topics" in data
    assert data["total"] > 0
    assert "yoga" in data["topics"]
    assert "meditation" in data["topics"]


def test_query_validation_empty(client):
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 422
