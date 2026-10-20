"""Answers against the real stack. They need the benchmark index and a model key, so they skip on
a stack without either (CI starts one with no data and no key)."""

import os

import httpx
import pytest

pytestmark = pytest.mark.integration

API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000")
QUESTION = "What are the default size limits for file uploads and total request size?"
STATUSES = {
    "verified",
    "partially_supported",
    "unsupported",
    "ungrounded",
    "unchecked",
    "unresolved",
    "uncited",
    "no_claim",
}


def ask(**body: object) -> httpx.Response:
    return httpx.post(f"{API_URL}/v1/query", json={"query": QUESTION, **body}, timeout=300)


@pytest.fixture(scope="module", autouse=True)
def answers_are_available() -> None:
    try:
        response = ask(verify=False)
    except httpx.HTTPError:
        pytest.skip("the API is not reachable")
    if response.status_code == 503:
        pytest.skip(f"no answers on this stack: {response.json()['detail']}")
    assert response.status_code == 200, response.text


def test_the_answer_comes_with_citations_and_a_verdict_for_every_statement() -> None:
    body = ask().json()

    assert body["answer"].strip()
    assert body["verification"]["method"] == "judge"
    assert body["claims"]
    assert {claim["status"] for claim in body["claims"]} <= STATUSES
    for citation in body["citations"]:
        assert citation["doc_id"] and all(0 <= i < len(body["claims"]) for i in citation["claims"])
    assert set(body["document_ids"]) == {c["doc_id"] for c in body["citations"]}
    assert body["usage"]["input_tokens"] > 0
    assert {"generation", "verification", "total"} <= set(body["timings_ms"])


def test_every_statement_that_is_verified_comes_with_evidence_taken_from_its_passage() -> None:
    claims = ask().json()["claims"]

    for claim in (c for c in claims if c["status"] == "verified"):
        assert claim["evidence"], claim


def test_without_the_judge_nothing_is_called_verified() -> None:
    body = ask(verify=False).json()

    assert body["verification"]["method"] == "values"
    assert all(claim["status"] != "verified" for claim in body["claims"])
