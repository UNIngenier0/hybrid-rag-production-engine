import asyncio
import time
from collections.abc import Callable

import httpx

from rag_engine.core.ports import HealthProbe

MakeClient = Callable[..., httpx.AsyncClient]


class FakeProbe:
    def __init__(
        self, name: str, *, error: Exception | None = None, delay_seconds: float = 0.0
    ) -> None:
        self.name = name
        self._error = error
        self._delay_seconds = delay_seconds

    async def check(self) -> None:
        if self._delay_seconds:
            await asyncio.sleep(self._delay_seconds)
        if self._error:
            raise self._error


class RendezvousProbe:
    """Succeeds only once its counterpart has started, which requires concurrent execution."""

    def __init__(self, name: str, mine: asyncio.Event, other: asyncio.Event) -> None:
        self.name = name
        self._mine = mine
        self._other = other

    async def check(self) -> None:
        self._mine.set()
        await self._other.wait()


async def test_liveness_does_not_depend_on_the_probes(make_client: MakeClient) -> None:
    failing = FakeProbe("qdrant", error=ConnectionError("down"))

    async with make_client(failing) as client:
        response = await client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_ready_when_every_probe_succeeds(make_client: MakeClient) -> None:
    probes: list[HealthProbe] = [FakeProbe("qdrant"), FakeProbe("postgres"), FakeProbe("redis")]

    async with make_client(*probes) as client:
        response = await client.get("/health/ready")

    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "ready"
    assert set(body["checks"]) == {"qdrant", "postgres", "redis"}
    assert all(check["ok"] and check["error"] is None for check in body["checks"].values())


async def test_not_ready_names_the_failing_dependency(make_client: MakeClient) -> None:
    probes: list[HealthProbe] = [
        FakeProbe("qdrant"),
        FakeProbe("postgres", error=ConnectionRefusedError("refused")),
    ]

    async with make_client(*probes) as client:
        response = await client.get("/health/ready")

    body = response.json()
    assert response.status_code == 503
    assert body["status"] == "not_ready"
    assert body["checks"]["qdrant"]["ok"] is True
    assert body["checks"]["postgres"] == {
        "ok": False,
        "latency_ms": body["checks"]["postgres"]["latency_ms"],
        "error": "ConnectionRefusedError",
    }


async def test_slow_dependency_is_reported_as_timeout(make_client: MakeClient) -> None:
    slow = FakeProbe("redis", delay_seconds=5)

    async with make_client(slow, timeout_seconds=0.05) as client:
        started = time.perf_counter()
        response = await client.get("/health/ready")
        elapsed = time.perf_counter() - started

    assert response.status_code == 503
    assert response.json()["checks"]["redis"]["error"] == "timeout"
    assert elapsed < 1.0


async def test_error_text_is_never_exposed(make_client: MakeClient) -> None:
    leaky = FakeProbe("postgres", error=RuntimeError("password=hunter2 host=10.0.0.5"))

    async with make_client(leaky) as client:
        response = await client.get("/health/ready")

    assert "hunter2" not in response.text
    assert "10.0.0.5" not in response.text
    assert response.json()["checks"]["postgres"]["error"] == "RuntimeError"


async def test_probes_run_concurrently(make_client: MakeClient) -> None:
    first, second = asyncio.Event(), asyncio.Event()
    probes: list[HealthProbe] = [
        RendezvousProbe("qdrant", first, second),
        RendezvousProbe("redis", second, first),
    ]

    async with make_client(*probes, timeout_seconds=0.5) as client:
        response = await client.get("/health/ready")

    # Run one after the other, the first probe would wait forever for the second and time out.
    assert response.status_code == 200
