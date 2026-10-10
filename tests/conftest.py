import os

import pytest
from fastapi.testclient import TestClient

# Ensure test env
os.environ.setdefault("LLM_PROVIDER", "groq")
os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("LOGFIRE_TOKEN", "")


@pytest.fixture
def client():
    from api.main import create_app

    app = create_app()
    return TestClient(app)
