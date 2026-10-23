"""Application factory. Run with `uvicorn rag_engine.api.main:create_app --factory`."""

import logging
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI

from rag_engine import __version__
from rag_engine.adapters.anthropic_llm import AnthropicClient
from rag_engine.adapters.probes import PostgresProbe, QdrantProbe, RedisProbe
from rag_engine.adapters.redis_store import RedisRateLimiter, RedisResponseCache, connect
from rag_engine.api.middleware import RequestContextMiddleware
from rag_engine.api.routers import demo, health, metrics, query, search
from rag_engine.api.security import build_api_keys, rate_limited
from rag_engine.api.wiring import LazyRetriever, build_query_service
from rag_engine.core.config import Settings, get_settings
from rag_engine.core.ports import HealthProbe, RateLimiter, ResponseCache, Retriever
from rag_engine.generation.query import QueryService
from rag_engine.observability.logging import configure_logging
from rag_engine.observability.metrics import Metrics


def create_app(
    settings: Settings | None = None,
    probes: Sequence[HealthProbe] | None = None,
    retriever: Retriever | None = None,
    query_service: QueryService | None = None,
    rate_limiter: RateLimiter | None = None,
    response_cache: ResponseCache | None = None,
) -> FastAPI:
    """Build the app. Settings, probes, the retriever, the query service, the rate limiter and the
    answer cache can be injected, as the tests do."""
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    api_keys = build_api_keys(settings)  # a malformed API_KEYS stops the service right here
    if api_keys is None:
        logging.getLogger(__name__).warning(
            "API_KEYS is not set: /v1/* is open to anyone who can reach it (development only)"
        )
    redis = None
    if rate_limiter is None or response_cache is None:
        uses_limits = settings.rate_limit_search_per_minute or settings.rate_limit_query_per_minute
        if uses_limits or settings.query_cache_ttl_seconds:
            redis = connect(settings.redis_url)
            rate_limiter = rate_limiter or RedisRateLimiter(redis)
            response_cache = response_cache or RedisResponseCache(redis)

    if probes is None:
        probes = [
            QdrantProbe(settings.qdrant_url),
            PostgresProbe(settings.database_url.get_secret_value()),
            RedisProbe(settings.redis_url),
        ]
    lazy = None if retriever is not None else LazyRetriever(settings)
    active_retriever = retriever or lazy
    assert active_retriever is not None
    clients: list[AnthropicClient] = []
    if query_service is None:
        built = build_query_service(settings, active_retriever)
        if built is not None:
            query_service, clients = built

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        for client in clients:
            await client.aclose()
        if redis is not None:
            await redis.aclose()
        if lazy is not None:
            await lazy.aclose()

    app = FastAPI(title="Hybrid RAG Production Engine", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.probes = tuple(probes)
    app.state.retriever = active_retriever
    app.state.query_service = query_service
    app.state.api_keys = api_keys
    app.state.metrics = Metrics()
    app.state.rate_limiter = rate_limiter
    app.state.response_cache = response_cache
    app.add_middleware(RequestContextMiddleware)
    app.include_router(health.router)
    app.include_router(metrics.router)
    app.include_router(demo.router)
    # Authentication comes with the limit: `rate_limited` depends on the client lookup.
    app.include_router(search.router, dependencies=[Depends(rate_limited("search"))])
    app.include_router(query.router, dependencies=[Depends(rate_limited("query"))])
    return app
