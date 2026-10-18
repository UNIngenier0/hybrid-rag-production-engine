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
