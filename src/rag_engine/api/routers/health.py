"""Liveness and readiness endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, Response, status

from rag_engine.api.dependencies import get_app_settings, get_probes
from rag_engine.api.schemas import LivenessResponse, ReadinessResponse
from rag_engine.core.config import Settings
from rag_engine.core.health import run_probes
from rag_engine.core.ports import HealthProbe

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live")
async def live() -> LivenessResponse:
    """The process is up. Checks no dependencies, so it never fails because of one."""
    return LivenessResponse(status="ok")


@router.get("/ready", responses={503: {"model": ReadinessResponse}})
async def ready(
    response: Response,
    probes: Annotated[tuple[HealthProbe, ...], Depends(get_probes)],
    settings: Annotated[Settings, Depends(get_app_settings)],
) -> ReadinessResponse:
    """Every dependency answered in time. Returns 503 with per-dependency detail otherwise."""
    report = await run_probes(probes, settings.health_check_timeout_seconds)
    if not report.ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse.from_report(report)
