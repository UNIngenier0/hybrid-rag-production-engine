import json

import httpx
import pytest

from rag_engine.adapters.anthropic_llm import AnthropicClient
from rag_engine.core.errors import LLMError

KEY = "sk-ant-test-secret-key"
OK = {
    "model": "claude-haiku-5-5",
    "content": [{"type": "text", "text": "Hello "}, {"type": "text", "text": "there"}],
    "usage": {"input_tokens": 12, "output_tokens": 5},
}


def client(handler, sleeps: list[float] | None = None, **options) -> AnthropicClient:  # type: ignore[no-untyped-def]
    async def sleep(seconds: float) -> None:
        if sleeps is not None:
            sleeps.append(seconds)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://api.test")
    return AnthropicClient(KEY, "claude-haiku-5-5", http=http, sleep=sleep, **options)


async def test_a_reply_carries_the_text_the_model_and_the_tokens() -> None:
    async with client(lambda request: httpx.Response(200, json=OK)) as llm:
        reply = await llm.complete("be brief", "hi")

    assert reply.text == "Hello there"
    assert reply.model == "claude-haiku-5-5"
    assert (reply.usage.input_tokens, reply.usage.output_tokens) == (12, 5)


async def test_cached_input_still_counts_as_input() -> None:
    payload = OK | {
        "usage": {
            "input_tokens": 10,
            "cache_creation_input_tokens": 20,
            "cache_read_input_tokens": 30,
            "output_tokens": 4,
        }
    }
    async with client(lambda request: httpx.Response(200, json=payload)) as llm:
        reply = await llm.complete("s", "u")

    assert reply.usage.input_tokens == 60


async def test_the_request_has_the_key_the_version_and_the_prompt() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=OK)

    async with client(handler) as llm:
        await llm.complete("the system prompt", "the user prompt", max_tokens=77)

    request = seen[0]
    body = json.loads(request.content)
    assert request.url.path == "/v1/messages"
    assert request.headers["x-api-key"] == KEY
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert body == {
        "model": "claude-haiku-5-5",
        "max_tokens": 77,
        "system": "the system prompt",
        "messages": [{"role": "user", "content": "the user prompt"}],
    }


async def test_a_temperature_is_sent_only_when_asked_for() -> None:
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=OK)

    async with client(handler) as llm:
        await llm.complete("s", "u", temperature=0.0)
        await llm.complete("s", "u")

    assert bodies[0]["temperature"] == 0.0
    assert "temperature" not in bodies[1]


async def test_rate_limits_are_retried_after_the_pause_the_server_asks_for() -> None:
    replies = iter(
        [httpx.Response(429, headers={"retry-after": "7"}), httpx.Response(200, json=OK)]
    )
    sleeps: list[float] = []

    async with client(lambda request: next(replies), sleeps) as llm:
        reply = await llm.complete("s", "u")

    assert reply.text == "Hello there"
    assert sleeps == [7.0]


async def test_overload_is_retried_with_a_growing_pause() -> None:
    replies = iter([httpx.Response(529), httpx.Response(529), httpx.Response(200, json=OK)])
    sleeps: list[float] = []

    async with client(lambda request: next(replies), sleeps) as llm:
        await llm.complete("s", "u")

    assert sleeps == [1.0, 2.0]


async def test_gives_up_when_the_retries_run_out() -> None:
    async with client(lambda request: httpx.Response(503), max_retries=2) as llm:
        with pytest.raises(LLMError, match="503"):
            await llm.complete("s", "u")


async def test_a_rejected_request_is_not_retried_and_never_echoes_the_key() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(
            401, json={"error": {"type": "authentication_error", "message": "invalid x-api-key"}}
        )

    async with client(handler) as llm:
        with pytest.raises(LLMError) as raised:
            await llm.complete("s", "u")

    assert len(calls) == 1
    assert "authentication_error" in str(raised.value)
    assert KEY not in str(raised.value)


async def test_network_errors_are_retried() -> None:
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) == 1:
            raise httpx.ConnectError("boom")
        return httpx.Response(200, json=OK)

    async with client(handler) as llm:
        reply = await llm.complete("s", "u")

    assert reply.text == "Hello there"
    assert len(attempts) == 2
