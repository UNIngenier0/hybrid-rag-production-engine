"""Anthropic Messages API client over httpx, with retries for the errors that pass on their own."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Self

import httpx

from rag_engine.core.errors import LLMError
from rag_engine.core.models import LLMReply, LLMUsage

logger = logging.getLogger(__name__)

API_URL = "https://api.anthropic.com"
API_VERSION = "2023-06-01"
# Rate limits, overload and server errors are worth another attempt; anything else is a mistake
# in the request or the credentials, and retrying would only repeat it.
_RETRYABLE = frozenset({408, 409, 429, 500, 502, 503, 504, 529})


class AnthropicClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str | None = None,
        http: httpx.AsyncClient | None = None,
        max_retries: int = 6,
        timeout_seconds: float = 180.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.model = model
        self._http = http or httpx.AsyncClient(
            base_url=base_url or API_URL, timeout=timeout_seconds
        )
        self._headers = {
            "x-api-key": api_key,
            "anthropic-version": API_VERSION,
            "content-type": "application/json",
        }
        self._max_retries = max_retries
        self._sleep = sleep

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    async def complete(
        self, system: str, user: str, *, max_tokens: int = 1024, temperature: float | None = None
    ) -> LLMReply:
        body: dict[str, object] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if temperature is not None:
            body["temperature"] = temperature
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._http.post("/v1/messages", json=body, headers=self._headers)
            except httpx.TransportError as error:
                if attempt == self._max_retries:
                    raise LLMError(
                        f"could not reach the Anthropic API: {type(error).__name__}"
                    ) from error
                await self._sleep(_backoff(attempt, None))
                continue
            if response.status_code == httpx.codes.OK:
                return _reply(response.json(), self.model)
            if response.status_code in _RETRYABLE and attempt < self._max_retries:
                logger.warning(
                    "retrying a model call",
                    extra={"status": response.status_code, "attempt": attempt},
                )
                await self._sleep(_backoff(attempt, response.headers.get("retry-after")))
                continue
            raise LLMError(_describe(response))
        raise AssertionError("unreachable")  # pragma: no cover


def _backoff(attempt: int, retry_after: str | None) -> float:
    if retry_after:
        try:
            return min(float(retry_after), 120.0)
        except ValueError:
            pass
    return float(min(2**attempt, 60))


def _reply(payload: dict[str, Any], model: str) -> LLMReply:
    text = "".join(
        str(block.get("text", ""))
        for block in payload.get("content") or []
        if isinstance(block, dict) and block.get("type") == "text"
    )
    usage = payload.get("usage") or {}
    # Cached input still counts as input for the purpose of knowing how much text was read.
    input_tokens = sum(
        int(usage.get(field, 0))
        for field in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
    )
    return LLMReply(
        text=text,
        usage=LLMUsage(input_tokens, int(usage.get("output_tokens", 0))),
        model=str(payload.get("model", model)),
    )


def _describe(response: httpx.Response) -> str:
    try:
        error = response.json().get("error", {})
        detail = f"{error.get('type', 'error')}: {error.get('message', '')}"
    except ValueError:
        detail = response.text
    return f"the Anthropic API answered {response.status_code} ({detail[:200]})"
