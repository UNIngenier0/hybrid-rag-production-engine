"""Runs against the real docker compose stack: `docker compose up --build --wait` first."""

import os

import httpx
import pytest

pytestmark = pytest.mark.integration

API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000")


def test_api_is_alive() -> None:
    response = httpx.get(f"{API_URL}/health/live", timeout=5)

    assert response.status_code == 200


def test_api_reaches_every_data_service() -> None:
    response = httpx.get(f"{API_URL}/health/ready", timeout=10)

    body = response.json()
    assert response.status_code == 200, body
    assert body["status"] == "ready"
    assert set(body["checks"]) == {"qdrant", "postgres", "redis"}
    assert all(check["ok"] for check in body["checks"].values())


def test_metrics_are_exposed_in_the_prometheus_format() -> None:
    response = httpx.get(f"{API_URL}/metrics", timeout=5)

    if response.status_code == 401:
        pytest.skip("METRICS_TOKEN is set on this stack")
    assert response.status_code == 200
    assert "rag_http_requests_total" in response.text
    assert 'route="/health/live"' in response.text


def test_requests_to_the_api_are_authenticated_when_keys_are_configured() -> None:
    response = httpx.post(f"{API_URL}/v1/search", json={"query": "x"}, timeout=10)

    # Open (development): the request reaches retrieval. Otherwise it asks for a key.
    assert response.status_code in {200, 401, 503}
    if response.status_code == 401:
        assert response.headers["www-authenticate"] == "Bearer"


def test_the_demo_page_is_served() -> None:
    response = httpx.get(f"{API_URL}/demo", timeout=5)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "default-src 'none'" in response.headers["content-security-policy"]
