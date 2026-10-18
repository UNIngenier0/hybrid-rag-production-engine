"""Request and response models for the HTTP API."""

from typing import Literal

from pydantic import BaseModel

from rag_engine.core.health import ReadinessReport


class LivenessResponse(BaseModel):
    status: Literal["ok"]


class ProbeStatus(BaseModel):
    ok: bool
    latency_ms: float
    error: str | None = None


class ReadinessResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    checks: dict[str, ProbeStatus]

    @classmethod
    def from_report(cls, report: ReadinessReport) -> "ReadinessResponse":
        return cls(
            status="ready" if report.ready else "not_ready",
            checks={
                result.name: ProbeStatus(
                    ok=result.ok, latency_ms=result.latency_ms, error=result.error
                )
                for result in report.results
            },
        )
