import re
from collections.abc import Callable

import httpx

MakeClient = Callable[..., httpx.AsyncClient]

GENERATED_ID = re.compile(r"[0-9a-f]{32}")


async def test_an_id_is_generated_when_the_client_sends_none(make_client: MakeClient) -> None:
    async with make_client() as client:
        response = await client.get("/health/live")

    assert GENERATED_ID.fullmatch(response.headers["X-Request-ID"])


async def test_a_well_formed_client_id_is_echoed(make_client: MakeClient) -> None:
    async with make_client() as client:
        response = await client.get("/health/live", headers={"X-Request-ID": "trace-abc.123_x"})

    assert response.headers["X-Request-ID"] == "trace-abc.123_x"


async def test_a_malformed_client_id_is_replaced(make_client: MakeClient) -> None:
    async with make_client() as client:
        response = await client.get("/health/live", headers={"X-Request-ID": "not valid!"})

    assert GENERATED_ID.fullmatch(response.headers["X-Request-ID"])


async def test_an_oversized_client_id_is_replaced(make_client: MakeClient) -> None:
    async with make_client() as client:
        response = await client.get("/health/live", headers={"X-Request-ID": "a" * 65})

    assert GENERATED_ID.fullmatch(response.headers["X-Request-ID"])
