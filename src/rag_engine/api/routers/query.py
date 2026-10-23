"""Answers: a cited answer to a question, with each citation checked against its passage."""

import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import ValidationError

from rag_engine.api.dependencies import get_app_settings, get_query_service
from rag_engine.api.query_cache import cache_key
from rag_engine.api.schemas import QueryRequest, QueryResponse, Usage
from rag_engine.core.config import Settings
from rag_engine.core.errors import BudgetExceededError, LLMError, RetrievalUnavailableError
from rag_engine.core.models import SearchFilters
from rag_engine.core.ports import ResponseCache
from rag_engine.generation.query import QueryService
from rag_engine.observability.metrics import Metrics

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["query"])


@router.post(
    "/query",
    responses={
        401: {"description": "A valid API key is required"},
        429: {"description": "The client's rate limit was exceeded"},
        502: {"description": "The language model could not answer"},
        503: {"description": "Retrieval or answer generation is not available"},
    },
)
async def query(
    request: QueryRequest,
    http: Request,
    service: Annotated[QueryService, Depends(get_query_service)],
    settings: Annotated[Settings, Depends(get_app_settings)],
) -> QueryResponse:
    """Retrieve the best passages, write an answer that cites them, and verify each citation.

    The answer comes back twice, clean and with its `[n]` markers, next to the statements it was
    split into and what was found for each: verified against the cited passage, only partly
    supported, unsupported, or not cited at all. An identical question asked again within
    `QUERY_CACHE_TTL_SECONDS` is answered from the cache (`cached: true`).
    """
    metrics: Metrics = http.app.state.metrics
    cache: ResponseCache | None = http.app.state.response_cache
    verify = settings.verify_citations if request.verify is None else request.verify
    caching = cache is not None and request.cache and settings.query_cache_ttl_seconds > 0
    key = cache_key(settings, request, verify) if caching else None

    if cache is not None and key is not None:
        hit = await _lookup(cache, key, metrics)
        if hit is not None:
            return hit

    filters = SearchFilters(
        source_types=tuple(request.filters.source_types), doc_ids=tuple(request.filters.doc_ids)
    )
    try:
        result = await service.query(
            request.query,
            mode=request.mode,
            chunks=request.chunks,
            filters=filters,
            verify=verify,
        )
    except RetrievalUnavailableError as error:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "retrieval unavailable") from error
    except BudgetExceededError as error:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "the model budget is spent"
        ) from error
    except LLMError as error:
        # The cause is in the server logs; the client only learns that the model failed.
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "the language model failed") from error

    response = QueryResponse.from_result(result)
    _record(metrics, response, result.timings_ms)
    # An answer whose citation check could not run is not worth keeping: ask again, get it checked.
    if cache is not None and key is not None and response.verification.error is None:
        try:
            await cache.set(key, response.model_dump_json(), settings.query_cache_ttl_seconds)
        except Exception:
            metrics.cache.labels("error").inc()
            logger.warning("could not store the answer in the cache", exc_info=True)
    return response


async def _lookup(cache: ResponseCache, key: str, metrics: Metrics) -> QueryResponse | None:
    started = time.perf_counter()
    try:
        stored = await cache.get(key)
        stored_response = QueryResponse.model_validate_json(stored) if stored else None
    except (ValidationError, ValueError):
        logger.warning("ignoring an unreadable cached answer", exc_info=True)
        stored_response = None
    except Exception:
        # Redis is down or slow: answer from the pipeline, as if there were no cache.
        metrics.cache.labels("error").inc()
        logger.warning("query cache unavailable", exc_info=True)
        return None
    if stored_response is None:
        metrics.cache.labels("miss").inc()
        return None
    metrics.cache.labels("hit").inc()
    lookup_ms = round((time.perf_counter() - started) * 1000, 1)
    return stored_response.model_copy(
        update={
            "cached": True,
            "timings_ms": {"total": lookup_ms},
            "usage": Usage(
                model=stored_response.usage.model, input_tokens=0, output_tokens=0, cost_usd=0.0
            ),
        }
    )


def _record(metrics: Metrics, response: QueryResponse, timings_ms: dict[str, float]) -> None:
    if response.cached is False:
        metrics.queries.labels(response.status).inc()
        for stage, milliseconds in timings_ms.items():
            metrics.query_stage_seconds.labels(stage).observe(milliseconds / 1000)
        usage = response.usage
        metrics.query_tokens.labels("input").inc(usage.input_tokens)
        metrics.query_tokens.labels("output").inc(usage.output_tokens)
        if usage.cost_usd:
            metrics.query_cost.inc(usage.cost_usd)
