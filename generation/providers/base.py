from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    async def generate(self, system_prompt: str, user_message: str) -> str: ...

    def name(self) -> str: ...
