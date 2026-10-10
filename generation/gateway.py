from __future__ import annotations

import logfire

from config.settings import settings
from generation.providers.base import LLMProvider
from generation.providers.groq_provider import GroqProvider

_provider: LLMProvider | None = None


def get_llm_provider() -> LLMProvider:
    global _provider
    if _provider is None:
        if settings.llm_provider == "groq":
            _provider = GroqProvider()
        elif settings.llm_provider == "ollama":
            from generation.providers.ollama_provider import OllamaProvider
            _provider = OllamaProvider()
        else:
            raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")
        logfire.info(f"LLM provider initialized: {_provider.name()}")
    return _provider
