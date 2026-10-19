"""Application factory. Run with `uvicorn rag_engine.api.main:create_app --factory`."""

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager

from fastapi import FastAPI

from rag_engine import __version__
from rag_engine.adapters.probes import PostgresProbe, QdrantProbe, RedisProbe
from rag_engine.api.middleware import RequestContextMiddleware
from rag_engine.api.routers import health, search
from rag_engine.api.wiring import LazyRetriever
from rag_engine.core.config import Settings, get_settings
from rag_engine.core.ports import HealthProbe, Retriever
from rag_engine.observability.logging import configure_logging


def create_app(
    settings: Settings | None = None,
    probes: Sequence[HealthProbe] | None = None,
    retriever: Retriever | None = None,
) -> FastAPI:
    """Build the app. Settings, probes and the retriever can be injected, as the tests do."""
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    if probes is None:
        probes = [
            QdrantProbe(settings.qdrant_url),
            PostgresProbe(settings.database_url.get_secret_value()),
            RedisProbe(settings.redis_url),
        ]
    lazy = None if retriever is not None else LazyRetriever(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        if lazy is not None:
            await lazy.aclose()

    app = FastAPI(title="Hybrid RAG Production Engine", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.probes = tuple(probes)
    app.state.retriever = retriever or lazy
    app.add_middleware(RequestContextMiddleware)
    app.include_router(health.router)
    app.include_router(search.router)
    return app
