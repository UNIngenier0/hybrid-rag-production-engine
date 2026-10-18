import httpx
import pytest

from rag_engine.adapters.probes import QdrantProbe


async def test_succeeds_when_qdrant_is_ready() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/readyz"
        return httpx.Response(200, text="all shards are ready")

    probe = QdrantProbe("http://qdrant:6333", transport=httpx.MockTransport(handler))

    await probe.check()


async def test_fails_when_qdrant_is_not_ready() -> None:
    probe = QdrantProbe(
        "http://qdrant:6333", transport=httpx.MockTransport(lambda _: httpx.Response(503))
    )

    with pytest.raises(httpx.HTTPStatusError):
        await probe.check()


async def test_fails_when_qdrant_is_unreachable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("unreachable", request=request)

    probe = QdrantProbe("http://qdrant:6333", transport=httpx.MockTransport(handler))

    with pytest.raises(httpx.ConnectError):
        await probe.check()
