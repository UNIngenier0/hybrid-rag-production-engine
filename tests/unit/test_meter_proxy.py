import importlib.util
import json
import sys
from pathlib import Path

import httpx

SCRIPT = Path(__file__).parents[2] / "scripts" / "benchmark" / "meter_proxy.py"
spec = importlib.util.spec_from_file_location("meter_proxy", SCRIPT)
assert spec and spec.loader
meter_proxy = importlib.util.module_from_spec(spec)
sys.modules["meter_proxy"] = meter_proxy
spec.loader.exec_module(meter_proxy)

JSON_REPLY = {
    "model": "claude-haiku-5-5",
    "content": [{"type": "text", "text": "hi"}],
    "usage": {"input_tokens": 1000, "output_tokens": 200},
}


def _event(name: str, data: dict[str, object]) -> str:
    return f"event: {name}\ndata: {json.dumps(data)}\n\n"


STREAM = (
    _event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "model": "claude-haiku-5-5",
                "usage": {"input_tokens": 500, "output_tokens": 1},
            },
        },
    )
    + _event("content_block_delta", {"type": "content_block_delta", "delta": {"text": "hi"}})
    + _event("message_delta", {"type": "message_delta", "usage": {"output_tokens": 40}})
)


class Chunks(httpx.AsyncByteStream):
    """A reply that arrives in pieces, as a real streamed response does."""

    def __init__(self, text: str) -> None:
        self.parts = [text[i : i + 64].encode() for i in range(0, len(text), 64)]

    async def __aiter__(self):  # type: ignore[no-untyped-def]
        for part in self.parts:
            yield part


def make_app(handler, budget=None):  # type: ignore[no-untyped-def]
    return meter_proxy.create_app("sk-ant-real-key", budget, transport=httpx.MockTransport(handler))


def client_for(app) -> httpx.AsyncClient:  # type: ignore[no-untyped-def]
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy")


async def test_a_json_reply_is_forwarded_with_the_real_key_and_counted() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=JSON_REPLY)

    app = make_app(handler)
    async with client_for(app) as client:
        reply = await client.post(
            "/v1/messages", json={"model": "claude-haiku-5-5"}, headers={"x-api-key": "dummy"}
        )
        usage = (await client.get("/__usage")).json()

    assert reply.json() == JSON_REPLY
    assert seen[0].headers["x-api-key"] == "sk-ant-real-key"  # the tool never needs the real one
    assert usage["calls"] == 1
    assert usage["by_model"]["claude-haiku-5-5"]["input_tokens"] == 1000
    assert abs(usage["spent_usd"] - (1000 * 0.10 + 200 * 0.50) / 1e6) < 1e-9


async def test_a_streamed_reply_is_relayed_and_counted() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, stream=Chunks(STREAM), headers={"content-type": "text/event-stream"}
        )

    app = make_app(handler)
    async with client_for(app) as client:
        reply = await client.post("/v1/messages", json={"stream": True})
        usage = (await client.get("/__usage")).json()

    assert reply.text == STREAM
    entry = usage["by_model"]["claude-haiku-5-5"]
    assert (entry["input_tokens"], entry["output_tokens"]) == (500, 40)


async def test_errors_from_upstream_pass_through_and_cost_nothing() -> None:
    app = make_app(
        lambda request: httpx.Response(429, json={"error": {"type": "rate_limit_error"}})
    )

    async with client_for(app) as client:
        reply = await client.post("/v1/messages", json={})
        usage = (await client.get("/__usage")).json()

    assert reply.status_code == 429
    assert usage["calls"] == 0


async def test_once_the_budget_is_spent_new_calls_are_refused_without_reaching_the_api() -> None:
    forwarded: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        forwarded.append(1)
        return httpx.Response(200, json=JSON_REPLY)

    app = make_app(handler, budget=0.0001)  # one call costs $0.0002
    async with client_for(app) as client:
        first = await client.post("/v1/messages", json={})
        second = await client.post("/v1/messages", json={})
        usage = (await client.get("/__usage")).json()

    assert first.status_code == 200
    assert second.status_code == 400
    assert "budget" in second.json()["error"]["message"]
    assert len(forwarded) == 1
    assert usage["refused_after_budget"] == 1


async def test_a_model_without_a_price_is_counted_in_tokens_only() -> None:
    reply = JSON_REPLY | {"model": "claude-future-9"}
    app = make_app(lambda request: httpx.Response(200, json=reply))

    async with client_for(app) as client:
        await client.post("/v1/messages", json={})
        usage = (await client.get("/__usage")).json()

    assert usage["spent_usd"] == 0
    assert usage["by_model"]["claude-future-9"]["input_tokens"] == 1000


async def test_a_compressed_stream_reaches_the_client_readable_and_is_counted() -> None:
    import gzip

    compressed = gzip.compress(STREAM.encode())

    class Compressed(httpx.AsyncByteStream):
        async def __aiter__(self):  # type: ignore[no-untyped-def]
            for i in range(0, len(compressed), 40):
                yield compressed[i : i + 40]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            stream=Compressed(),
            headers={"content-type": "text/event-stream", "content-encoding": "gzip"},
        )

    app = make_app(handler)
    async with client_for(app) as client:
        reply = await client.post("/v1/messages", json={"stream": True})
        usage = (await client.get("/__usage")).json()

    assert reply.text == STREAM
    assert usage["by_model"]["claude-haiku-5-5"]["output_tokens"] == 40
