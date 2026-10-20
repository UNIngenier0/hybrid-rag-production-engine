"""A metering proxy for the Anthropic API.

The benchmark's own evaluator and agent call the API on their own, with no cost report and no cap.
Pointing them at this proxy (ANTHROPIC_BASE_URL) makes every call pass through here: it forwards
the request with the real key (which therefore never reaches the tool), adds up the tokens and
the cost per model, and refuses new calls once the budget is spent.

    python scripts/benchmark/meter_proxy.py            # listens on :8080
    GET /__usage                                       # what has been spent so far

The key comes from ANTHROPIC_API_KEY, the budget in dollars from METER_BUDGET_USD.
"""

import json
import os
from collections import defaultdict
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from rag_engine.core.models import LLMUsage
from rag_engine.generation.cost import cost_usd

UPSTREAM = "https://api.anthropic.com"
_DROPPED_HEADERS = {"host", "content-length", "x-api-key", "connection", "accept-encoding"}


class Tally:
    def __init__(self, budget_usd: float | None) -> None:
        self.budget_usd = budget_usd
        self.calls = 0
        self.refused = 0
        self.by_model: dict[str, dict[str, float]] = defaultdict(
            lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0, "usd": 0.0}
        )

    @property
    def spent_usd(self) -> float:
        return sum(entry["usd"] for entry in self.by_model.values())

    @property
    def exhausted(self) -> bool:
        return self.budget_usd is not None and self.spent_usd >= self.budget_usd

    def add(self, model: str, usage: LLMUsage) -> None:
        entry = self.by_model[model]
        entry["calls"] += 1
        entry["input_tokens"] += usage.input_tokens
        entry["output_tokens"] += usage.output_tokens
        try:
            entry["usd"] += cost_usd(model, usage)
        except KeyError:  # an unpriced model is counted in tokens but not in dollars
            entry["unpriced"] = 1
        self.calls += 1

    def report(self) -> dict[str, Any]:
        return {
            "budget_usd": self.budget_usd,
            "spent_usd": round(self.spent_usd, 6),
            "calls": self.calls,
            "refused_after_budget": self.refused,
            "by_model": self.by_model,
        }


def usage_of(payload: dict[str, Any]) -> LLMUsage:
    usage = payload.get("usage") or {}
    input_tokens = sum(
        int(usage.get(key, 0))
        for key in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
    )
    return LLMUsage(input_tokens, int(usage.get("output_tokens", 0)))


def usage_of_stream(lines: list[str]) -> tuple[str, LLMUsage]:
    """Model and tokens of a streamed reply: input from `message_start`, output from the last
    `message_delta` (which carries the running total)."""
    model, input_tokens, output_tokens = "", 0, 0
    for line in lines:
        if not line.startswith("data:"):
            continue
        try:
            event = json.loads(line[5:])
        except ValueError:
            continue
        if event.get("type") == "message_start":
            message = event.get("message", {})
            model = message.get("model", model)
            input_tokens = usage_of(message).input_tokens
            output_tokens = max(output_tokens, usage_of(message).output_tokens)
        elif event.get("type") == "message_delta":
            output_tokens = max(output_tokens, int(event.get("usage", {}).get("output_tokens", 0)))
    return model, LLMUsage(input_tokens, output_tokens)


def create_app(
    api_key: str, budget_usd: float | None, transport: httpx.AsyncBaseTransport | None = None
) -> FastAPI:
    app = FastAPI()
    tally = Tally(budget_usd)
    app.state.tally = tally
    client = httpx.AsyncClient(base_url=UPSTREAM, timeout=600.0, transport=transport)

    @app.get("/__usage")
    async def usage() -> dict[str, Any]:
        return tally.report()

    @app.api_route("/{path:path}", methods=["POST", "GET"])
    async def forward(path: str, request: Request) -> Response:
        if tally.exhausted:
            tally.refused += 1
            return JSONResponse(
                status_code=400,
                content={
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": f"metering proxy: the budget of ${tally.budget_usd} is spent",
                    },
                },
            )
        body = await request.body()
        headers = {k: v for k, v in request.headers.items() if k.lower() not in _DROPPED_HEADERS}
        headers["x-api-key"] = api_key
        upstream = await client.send(
            client.build_request(request.method, f"/{path}", content=body, headers=headers),
            stream=True,
        )
        passthrough = {
            k: v
            for k, v in upstream.headers.items()
            if k.lower() not in {"content-length", "content-encoding", "transfer-encoding"}
        }
        if "text/event-stream" in upstream.headers.get("content-type", ""):

            async def relay() -> Any:
                raw = bytearray()  # events can be cut anywhere between two network chunks
                async for chunk in upstream.aiter_bytes():  # decoded: the client asked for plain
                    raw.extend(chunk)
                    yield chunk
                await upstream.aclose()
                model, usage = usage_of_stream(raw.decode("utf-8", "ignore").splitlines())
                if model:
                    tally.add(model, usage)

            return StreamingResponse(relay(), status_code=upstream.status_code, headers=passthrough)
        content = await upstream.aread()
        await upstream.aclose()
        if upstream.status_code == 200:
            try:
                payload = json.loads(content)
                if "usage" in payload:
                    tally.add(str(payload.get("model", "")), usage_of(payload))
            except ValueError:
                pass
        return Response(content, status_code=upstream.status_code, headers=passthrough)

    return app


if __name__ == "__main__":
    import uvicorn

    budget = os.environ.get("METER_BUDGET_USD")
    uvicorn.run(
        create_app(os.environ["ANTHROPIC_API_KEY"], float(budget) if budget else None),
        host="0.0.0.0",
        port=8080,
        log_level="warning",
    )
