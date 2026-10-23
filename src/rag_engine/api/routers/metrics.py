"""Prometheus scrape endpoint."""

import hmac
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from rag_engine.core.config import Settings
from rag_engine.observability.metrics import CONTENT_TYPE, Metrics

router = APIRouter(tags=["metrics"])
_bearer = HTTPBearer(auto_error=False)


@router.get("/metrics", include_in_schema=False)
async def metrics(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> Response:
    """Counters and histograms in the Prometheus text format. Protected by METRICS_TOKEN when it
    is set; otherwise keep the port off the public network."""
    settings: Settings = request.app.state.settings
    if settings.metrics_token is not None:
        presented = credentials.credentials if credentials else ""
        expected = settings.metrics_token.get_secret_value()
        if not hmac.compare_digest(presented.encode(), expected.encode()):
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                "a metrics token is required",
                headers={"WWW-Authenticate": "Bearer"},
            )
    collected: Metrics = request.app.state.metrics
    return Response(collected.render(), media_type=CONTENT_TYPE)
