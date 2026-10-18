"""Readiness aggregation: run every dependency probe concurrently under a time limit."""

import asyncio
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass

from rag_engine.core.ports import HealthProbe

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProbeResult:
    name: str
    ok: bool
    latency_ms: float
    # Short, safe-to-expose error label (an exception class name), never the exception text.
    error: str | None = None


@dataclass(frozen=True)
class ReadinessReport:
    results: tuple[ProbeResult, ...]

    @property
    def ready(self) -> bool:
        return all(result.ok for result in self.results)


async def run_probes(probes: Sequence[HealthProbe], timeout_seconds: float) -> ReadinessReport:
    results = await asyncio.gather(*(_run_probe(probe, timeout_seconds) for probe in probes))
    return ReadinessReport(results=tuple(results))


async def _run_probe(probe: HealthProbe, timeout_seconds: float) -> ProbeResult:
    started = time.perf_counter()
    error: str | None = None
    try:
        async with asyncio.timeout(timeout_seconds):
            await probe.check()
    except TimeoutError:
        error = "timeout"
    except Exception as exc:
        # The full message goes to the logs only: it may name hosts or other internals.
        error = type(exc).__name__
        logger.warning("health probe failed", extra={"probe": probe.name, "detail": str(exc)[:200]})
    latency_ms = round((time.perf_counter() - started) * 1000, 1)
    return ProbeResult(name=probe.name, ok=error is None, latency_ms=latency_ms, error=error)
