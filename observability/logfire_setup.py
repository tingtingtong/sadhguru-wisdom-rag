import logfire
from fastapi import FastAPI

from config.settings import settings


def setup_logfire(app: FastAPI) -> None:
    logfire.configure(token=settings.logfire_token or None, send_to_logfire="if-token-present")
    logfire.instrument_fastapi(app)
