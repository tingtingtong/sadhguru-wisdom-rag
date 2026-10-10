from __future__ import annotations

import logfire
from ollama import AsyncClient

from config.settings import settings
from observability.instrumentation import trace


class OllamaProvider:
    def __init__(self) -> None:
        self._client = AsyncClient(host=settings.ollama_base_url)
        self._model = settings.ollama_model

    def name(self) -> str:
        return "ollama"

    @trace("llm_generate")
    async def generate(self, system_prompt: str, user_message: str) -> str:
        with logfire.span("ollama_api_call", model=self._model):
            response = await self._client.chat(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )
            return response.message.content or ""
