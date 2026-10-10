import asyncio

from config.settings import settings
from generation.providers.groq_provider import GroqProvider


def test_falls_back_to_second_model_on_primary_failure(monkeypatch):
    calls = []

    async def fake_call(self, model, system_prompt, user_message):
        calls.append(model)
        if model == settings.groq_primary_model:
            raise RuntimeError("429 rate limited")
        return "ok from fallback"

    monkeypatch.setattr(GroqProvider, "_call", fake_call)
    out = asyncio.run(GroqProvider().generate("sys", "hi"))
    assert out == "ok from fallback"
    assert calls == [settings.groq_primary_model, settings.groq_fallback_model]


def test_fallback_model_is_not_the_primary():
    assert settings.groq_fallback_model != settings.groq_primary_model


def test_waits_out_short_rate_limit_then_retries_primary(monkeypatch):
    from generation.providers import groq_provider

    sleeps, calls = [], []

    async def fake_sleep(s):
        sleeps.append(s)

    async def fake_call(self, model, system_prompt, user_message):
        calls.append(model)
        if len(calls) < 3:
            raise RuntimeError("Error code: 429 - Rate limit reached. Please try again in 1.02s. Need more tokens?")
        return "finally ok"

    monkeypatch.setattr(groq_provider.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(GroqProvider, "_call", fake_call)
    assert asyncio.run(GroqProvider().generate("sys", "hi")) == "finally ok"
    assert calls == [settings.groq_primary_model, settings.groq_fallback_model, settings.groq_primary_model]
    assert 1.0 < sleeps[0] < 2.0


def test_does_not_wait_when_hint_is_long_or_missing(monkeypatch):
    import pytest

    async def fake_call(self, model, system_prompt, user_message):
        raise RuntimeError("Error code: 429 - Please try again in 600s")

    from generation.errors import LLMBusyError

    monkeypatch.setattr(GroqProvider, "_call", fake_call)
    with pytest.raises(LLMBusyError):  # too long to wait out: surface "busy", don't hold the request
        asyncio.run(GroqProvider().generate("sys", "hi"))


def test_retry_hint_parsing():
    from generation.providers.groq_provider import _retry_after_seconds

    assert abs(_retry_after_seconds(RuntimeError("try again in 250ms")) - 0.5) < 1e-6
    assert _retry_after_seconds(RuntimeError("something else")) is None


def test_raises_busy_error_when_all_models_are_rate_limited_for_long(monkeypatch):
    import pytest

    from generation.errors import LLMBusyError

    async def fake_call(self, model, system_prompt, user_message):
        raise RuntimeError("Error code: 429 - tokens per day. Please try again in 167s")

    monkeypatch.setattr(GroqProvider, "_call", fake_call)
    with pytest.raises(LLMBusyError):
        asyncio.run(GroqProvider().generate("sys", "hi"))


def test_other_errors_are_not_disguised_as_busy(monkeypatch):
    import pytest

    from generation.errors import LLMBusyError

    async def fake_call(self, model, system_prompt, user_message):
        raise RuntimeError("Error code: 404 - model_not_found")

    monkeypatch.setattr(GroqProvider, "_call", fake_call)
    with pytest.raises(RuntimeError) as e:
        asyncio.run(GroqProvider().generate("sys", "hi"))
    assert not isinstance(e.value, LLMBusyError)
