from __future__ import annotations

import asyncio
import re

import logfire
from groq import AsyncGroq

from config.settings import settings
from generation.errors import LLMBusyError
from observability.instrumentation import trace


_RETRY_HINT = re.compile(r"try again in ([0-9.]+)\s*(ms|s)", re.I)


def _retry_after_seconds(error: Exception) -> float | None:
    """Parse Groq's 'Please try again in 1.02s' hint from a 429, with a small safety margin."""
    m = _RETRY_HINT.search(str(error))
    if not m:
        return None
    value = float(m.group(1)) / (1000 if m.group(2).lower() == "ms" else 1)
    return value + 0.25


def _is_rate_limit(error: Exception) -> bool:
    return getattr(error, "status_code", None) == 429 or "429" in str(error)[:40]


class GroqProvider:
    def __init__(self) -> None:
        # Few SDK retries: on a rate limit, failing over to the fallback model beats sleeping through backoff
        self._client = AsyncGroq(api_key=settings.groq_api_key, max_retries=1, timeout=30.0)
        self._primary_model = settings.groq_primary_model
        self._fallback_model = settings.groq_fallback_model

    def name(self) -> str:
        return "groq"

    @trace("llm_generate")
    async def generate(self, system_prompt: str, user_message: str) -> str:
        try:
            return await self._call(self._primary_model, system_prompt, user_message)
        except Exception as e:
            logfire.warn(f"Primary model failed ({self._primary_model}): {e}, trying fallback")
            primary_error = e
        try:
            return await self._call(self._fallback_model, system_prompt, user_message)
        except Exception as e:
            wait = _retry_after_seconds(primary_error) or _retry_after_seconds(e)
            if wait is None or wait > settings.groq_max_rate_limit_wait_s:
                if _is_rate_limit(primary_error) or _is_rate_limit(e):
                    raise LLMBusyError("all models rate limited") from e
                raise
            # Both models are rate limited (free-tier tokens/minute): wait the hinted time, retry once
            logfire.warn(f"Both models rate limited; waiting {wait:.1f}s then retrying the primary")
            await asyncio.sleep(wait)
            return await self._call(self._primary_model, system_prompt, user_message)

    async def _call(self, model: str, system_prompt: str, user_message: str) -> str:
        with logfire.span("groq_api_call", model=model):
            response = await self._client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=settings.groq_temperature,
                max_tokens=2048,
            )
            return response.choices[0].message.content or ""
