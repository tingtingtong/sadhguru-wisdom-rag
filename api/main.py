from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.middleware.cors import setup_cors
from api.middleware.error_handler import setup_error_handler
from api.middleware.rate_limiter import setup_rate_limiter
from api.routes import health, query, search, topics
from observability.logfire_setup import setup_logfire


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio

    from retrieval.warmup import warm_up

    await asyncio.to_thread(warm_up)  # first request shouldn't pay for client set-up
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        lifespan=lifespan,
        title="Sadhguru Wisdom RAG",
        description="Enterprise RAG application for Sadhguru's teachings",
        version="0.1.0",
    )

    # Middleware
    setup_cors(app)
    setup_rate_limiter(app)
    setup_error_handler(app)

    # Observability
    setup_logfire(app)

    # Routes
    app.include_router(health.router)
    app.include_router(query.router)
    app.include_router(search.router)
    app.include_router(topics.router)

    return app
