from __future__ import annotations

import logfire
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


def setup_error_handler(app: FastAPI) -> None:
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logfire.error(f"Unhandled exception: {exc}", _exc_info=exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "An internal error occurred. Please try again later."},
        )
