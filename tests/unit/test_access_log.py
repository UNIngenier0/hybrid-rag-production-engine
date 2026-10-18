import logging
from collections.abc import Callable

import httpx
import pytest

MakeClient = Callable[..., httpx.AsyncClient]


async def test_each_request_is_logged_once(
    make_client: MakeClient, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.DEBUG, logger="rag_engine.access"):
        async with make_client() as client:
            await client.get("/health/live")

    records = [r for r in caplog.records if r.name == "rag_engine.access"]
    assert len(records) == 1
    assert records[0].http_method == "GET"
    assert records[0].http_path == "/health/live"
    assert records[0].http_status == 200


async def test_health_checks_are_logged_below_info(
    make_client: MakeClient, caplog: pytest.LogCaptureFixture
) -> None:
    # Orchestrators poll these every few seconds, so they must not flood the INFO stream.
    with caplog.at_level(logging.DEBUG, logger="rag_engine.access"):
        async with make_client() as client:
            await client.get("/health/live")

    assert [r.levelno for r in caplog.records if r.name == "rag_engine.access"] == [logging.DEBUG]
