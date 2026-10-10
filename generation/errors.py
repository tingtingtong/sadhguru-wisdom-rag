from __future__ import annotations

BUSY_MESSAGE = "I'm receiving a lot of questions right now. Please try again in a minute."
GENERIC_ERROR_MESSAGE = "Something went wrong while preparing your answer. Please try again."


class LLMBusyError(Exception):
    """Every configured model is rate limited and the wait would be too long to hold the request."""
