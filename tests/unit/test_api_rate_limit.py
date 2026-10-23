import httpx

from fakes import FakeRateLimiter, FakeRetriever, make_hit
from rag_engine.api.main import create_app
from rag_engine.core.config import Settings

SEARCH = {"query": "upload limit"}


def client(settings: Settings, limiter: FakeRateLimiter, **limits: int) -> httpx.AsyncClient:
    limited = settings.model_copy(update=limits)
    app = create_app(
        limited, probes=[], retriever=FakeRetriever([make_hit("a")]), rate_limiter=limiter
    )
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_requests_within_the_limit_report_what_is_left(settings: Settings) -> None:
    async with client(settings, FakeRateLimiter(), rate_limit_search_per_minute=3) as http:
        first = await http.post("/v1/search", json=SEARCH)
        second = await http.post("/v1/search", json=SEARCH)

    assert first.status_code == 200
    assert first.headers["x-ratelimit-limit"] == "3"
    assert first.headers["x-ratelimit-remaining"] == "2"
    assert second.headers["x-ratelimit-remaining"] == "1"


async def test_the_request_over_the_limit_is_refused_with_when_to_retry(
    settings: Settings,
) -> None:
    async with client(settings, FakeRateLimiter(), rate_limit_search_per_minute=2) as http:
        responses = [await http.post("/v1/search", json=SEARCH) for _ in range(3)]

    assert [r.status_code for r in responses] == [200, 200, 429]
    refused = responses[2]
    assert refused.headers["retry-after"] == "60"
    assert refused.headers["x-ratelimit-remaining"] == "0"
    assert "2 search requests per minute" in refused.json()["detail"]


async def test_search_and_query_have_separate_limits(settings: Settings) -> None:
    limiter = FakeRateLimiter()
    async with client(
        settings, limiter, rate_limit_search_per_minute=1, rate_limit_query_per_minute=1
    ) as http:
        await http.post("/v1/search", json=SEARCH)
        search_again = await http.post("/v1/search", json=SEARCH)
        # No model key in the tests: the query reaches its 503 only if the limit let it through.
        query = await http.post("/v1/query", json=SEARCH)

    assert search_again.status_code == 429
    assert query.status_code == 503
    assert {key.split(":")[0] for key in limiter.counts} == {"search", "query"}


async def test_a_limit_of_zero_turns_the_limit_off(settings: Settings) -> None:
    limiter = FakeRateLimiter()
    async with client(settings, limiter, rate_limit_search_per_minute=0) as http:
        responses = [await http.post("/v1/search", json=SEARCH) for _ in range(3)]

    assert [r.status_code for r in responses] == [200, 200, 200]
    assert limiter.calls == []


async def test_if_the_limiter_is_down_requests_go_through_and_the_failure_is_counted(
    settings: Settings,
) -> None:
    broken = FakeRateLimiter(error=ConnectionError("redis is down"))
    async with client(settings, broken, rate_limit_search_per_minute=1) as http:
        responses = [await http.post("/v1/search", json=SEARCH) for _ in range(2)]
        text = (await http.get("/metrics")).text

    assert [r.status_code for r in responses] == [200, 200]
    assert "rag_rate_limit_errors_total 2.0" in text


async def test_anonymous_callers_are_counted_by_the_socket_address_not_a_header(
    settings: Settings,
) -> None:
    limiter = FakeRateLimiter()
    async with client(settings, limiter, rate_limit_search_per_minute=1) as http:
        await http.post("/v1/search", json=SEARCH, headers={"X-Forwarded-For": "1.1.1.1"})
        spoofed = await http.post("/v1/search", json=SEARCH, headers={"X-Forwarded-For": "2.2.2.2"})

    assert spoofed.status_code == 429
    assert len(limiter.counts) == 1


async def test_refusals_are_counted_by_group_and_client_name(settings: Settings) -> None:
    async with client(settings, FakeRateLimiter(), rate_limit_search_per_minute=1) as http:
        await http.post("/v1/search", json=SEARCH)
        await http.post("/v1/search", json=SEARCH)
        text = (await http.get("/metrics")).text

    assert 'rag_rate_limited_total{client="anonymous",group="search"} 1.0' in text
