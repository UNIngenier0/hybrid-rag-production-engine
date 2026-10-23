import httpx
from pydantic import SecretStr

from fakes import FakeRetriever, make_hit
from rag_engine.api.main import create_app
from rag_engine.core.config import Settings


def client(settings: Settings, **changes: object) -> httpx.AsyncClient:
    configured = settings.model_copy(update=changes)
    app = create_app(configured, probes=[], retriever=FakeRetriever([make_hit("a")]))
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_metrics_are_served_in_the_prometheus_text_format(settings: Settings) -> None:
    async with client(settings) as http:
        response = await http.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "# TYPE rag_http_requests_total counter" in response.text


async def test_requests_are_counted_by_route_template_method_and_status(settings: Settings) -> None:
    async with client(settings) as http:
        await http.post("/v1/search", json={"query": "upload limit"})
        await http.post("/v1/search", json={"query": ""})  # invalid: 422
        await http.get("/health/live")
        text = (await http.get("/metrics")).text

    assert 'rag_http_requests_total{method="POST",route="/v1/search",status="200"} 1.0' in text
    assert 'rag_http_requests_total{method="POST",route="/v1/search",status="422"} 1.0' in text
    assert 'rag_http_requests_total{method="GET",route="/health/live",status="200"} 1.0' in text
    assert 'rag_http_request_duration_seconds_count{method="POST",route="/v1/search"} 2.0' in text


async def test_unknown_paths_share_one_label_so_scanners_cannot_grow_the_metrics(
    settings: Settings,
) -> None:
    async with client(settings) as http:
        for path in ("/wp-admin", "/.env", "/a/b/c"):
            await http.get(path)
        text = (await http.get("/metrics")).text

    assert 'route="unmatched",status="404"} 3.0' in text
    assert "wp-admin" not in text


async def test_the_metrics_token_protects_the_endpoint_when_it_is_set(settings: Settings) -> None:
    async with client(settings, metrics_token=SecretStr("scrape-token-123")) as http:
        missing = await http.get("/metrics")
        wrong = await http.get("/metrics", headers={"Authorization": "Bearer nope"})
        right = await http.get("/metrics", headers={"Authorization": "Bearer scrape-token-123"})

    assert (missing.status_code, wrong.status_code, right.status_code) == (401, 401, 200)


async def test_each_app_has_its_own_counters(settings: Settings) -> None:
    async with client(settings) as first:
        await first.get("/health/live")
    async with client(settings) as second:
        text = (await second.get("/metrics")).text

    assert 'route="/health/live"' not in text


async def test_the_metrics_endpoint_is_not_in_the_public_api_description(
    settings: Settings,
) -> None:
    async with client(settings) as http:
        schema = (await http.get("/openapi.json")).json()

    assert "/metrics" not in schema["paths"]
    assert "/v1/search" in schema["paths"]
