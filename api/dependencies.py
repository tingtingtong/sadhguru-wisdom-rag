from __future__ import annotations

from functools import lru_cache

from qdrant_client import QdrantClient

from config.settings import Settings, settings
from generation.gateway import get_llm_provider
from generation.providers.base import LLMProvider
from ingestion.indexer import get_qdrant


def get_settings() -> Settings:
    return settings


def get_qdrant_client() -> QdrantClient:
    return get_qdrant()


def get_provider() -> LLMProvider:
    return get_llm_provider()
