import logging

import httpx
import pytest
from pydantic import SecretStr

from fakes import FakeRateLimiter, FakeRetriever, make_hit
from rag_engine.api.main import create_app
from rag_engine.api.security import ApiKeys
from rag_engine.core.config import Settings

KEY_A = "alpha-secret-key-0001"
KEY_B = "bravo-secret-key-0002"


def app_client(settings: Settings, **overrides: object) -> httpx.AsyncClient:
    keyed = settings.model_copy(update=overrides)
    app = create_app(keyed, probes=[], retriever=FakeRetriever([make_hit("a")]))
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def keyed_client(
    settings: Settings, text: str = f"alpha:{KEY_A},bravo:{KEY_B}"
) -> httpx.AsyncClient:
    return app_client(settings, api_keys=SecretStr(text))


SEARCH = {"query": "upload limit"}


def test_keys_are_read_as_name_and_secret_pairs() -> None:
    keys = ApiKeys.parse(f"alpha:{KEY_A}, bravo:{KEY_B} ,")

    assert len(keys) == 2
    assert keys.authenticate(KEY_A) == "alpha"
    assert keys.authenticate(KEY_B) == "bravo"
    assert keys.authenticate("not-a-key-at-all-xxx") is None
    assert keys.authenticate("") is None


@pytest.mark.parametrize(
    "text, message",
    [
        ("", "no key"),
        (KEY_A, "name:secret"),
        (f"bad name:{KEY_A}", "name:secret"),
        ("alpha:short", "shorter than"),
        (f"alpha:{KEY_A},alpha:{KEY_B}", "repeats"),
        (f"alpha:{KEY_A},bravo:{KEY_A}", "repeats"),
    ],
)
def test_a_malformed_or_weak_key_list_is_refused(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ApiKeys.parse(text)


def test_the_service_does_not_start_with_a_malformed_key_list(settings: Settings) -> None:
    with pytest.raises(ValueError):
        create_app(settings.model_copy(update={"api_keys": SecretStr("alpha:short")}), probes=[])


async def test_without_configured_keys_the_api_is_open(settings: Settings) -> None:
    async with app_client(settings) as http:
        response = await http.post("/v1/search", json=SEARCH)

    assert response.status_code == 200


async def test_a_request_without_a_key_is_refused_and_told_how_to_authenticate(
    settings: Settings,
) -> None:
    async with keyed_client(settings) as http:
        response = await http.post("/v1/search", json=SEARCH)

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert "X-API-Key" in response.json()["detail"]


async def test_a_wrong_key_is_refused(settings: Settings) -> None:
    async with keyed_client(settings) as http:
        response = await http.post("/v1/search", json=SEARCH, headers={"X-API-Key": "wrong"})

    assert response.status_code == 401


async def test_a_valid_key_works_in_either_header(settings: Settings) -> None:
    async with keyed_client(settings) as http:
        by_header = await http.post("/v1/search", json=SEARCH, headers={"X-API-Key": KEY_A})
        by_bearer = await http.post(
            "/v1/search", json=SEARCH, headers={"Authorization": f"Bearer {KEY_B}"}
        )

    assert (by_header.status_code, by_bearer.status_code) == (200, 200)


async def test_the_query_endpoint_needs_a_key_too(settings: Settings) -> None:
    async with keyed_client(settings) as http:
        response = await http.post("/v1/query", json=SEARCH)

    assert response.status_code == 401


async def test_health_checks_stay_open_so_orchestrators_can_probe_the_service(
    settings: Settings,
) -> None:
    async with keyed_client(settings) as http:
        live = await http.get("/health/live")
        ready = await http.get("/health/ready")

    assert (live.status_code, ready.status_code) == (200, 200)


async def test_a_rejected_key_is_never_written_to_the_logs(
    settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    async with keyed_client(settings) as http:
        await http.post("/v1/search", json=SEARCH, headers={"X-API-Key": "wrong-secret-value-123"})

    assert "wrong-secret-value-123" not in caplog.text
    assert KEY_A not in caplog.text


async def test_refusals_are_counted_by_reason(settings: Settings) -> None:
    async with keyed_client(settings) as http:
        await http.post("/v1/search", json=SEARCH)
        await http.post("/v1/search", json=SEARCH, headers={"X-API-Key": "wrong"})
        text = (await http.get("/metrics")).text

    assert 'rag_auth_failures_total{reason="missing"} 1.0' in text
    assert 'rag_auth_failures_total{reason="invalid"} 1.0' in text


async def test_limits_are_counted_per_client_not_per_caller_address(settings: Settings) -> None:
    limiter = FakeRateLimiter()
    keyed = settings.model_copy(
        update={
            "api_keys": SecretStr(f"alpha:{KEY_A},bravo:{KEY_B}"),
            "rate_limit_search_per_minute": 2,
        }
    )
    app = create_app(
        keyed, probes=[], retriever=FakeRetriever([make_hit("a")]), rate_limiter=limiter
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as http:
        for _ in range(3):
            last = await http.post("/v1/search", json=SEARCH, headers={"X-API-Key": KEY_A})
        other = await http.post("/v1/search", json=SEARCH, headers={"X-API-Key": KEY_B})

    assert last.status_code == 429 and other.status_code == 200
    assert set(limiter.counts) == {"search:key:alpha", "search:key:bravo"}
