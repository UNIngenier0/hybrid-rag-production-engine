"""Application factory. Run with `uvicorn rag_engine.api.main:create_app --factory`."""

from collections.abc import Sequence

from fastapi import FastAPI

from rag_engine import __version__
from rag_engine.adapters.probes import PostgresProbe, QdrantProbe, RedisProbe
from rag_engine.api.middleware import RequestContextMiddleware
from rag_engine.api.routers import health
from rag_engine.core.config import Settings, get_settings
from rag_engine.core.ports import HealthProbe
from rag_engine.observability.logging import configure_logging


def create_app(
    settings: Settings | None = None, probes: Sequence[HealthProbe] | None = None
) -> FastAPI:
    """Build the app. Settings and probes can be injected, which is how the tests use it."""
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    if probes is None:
        probes = [
            QdrantProbe(settings.qdrant_url),
            PostgresProbe(settings.database_url.get_secret_value()),
            RedisProbe(settings.redis_url),
        ]

    app = FastAPI(title="Hybrid RAG Production Engine", version=__version__)
    app.state.settings = settings
    app.state.probes = tuple(probes)
    app.add_middleware(RequestContextMiddleware)
    app.include_router(health.router)
    return app
