"""Search against the real stack. They need the benchmark index, so they skip on an empty stack
(CI starts one with no data) and run on a machine where the index has been loaded."""

import os

import httpx
import pytest

pytestmark = pytest.mark.integration

API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000")
QUERY = "default wait time of the tool-call batching invoker"


def search(**body: object) -> httpx.Response:
    return httpx.post(
        f"{API_URL}/v1/search", json={"query": QUERY, "limit": 5, **body}, timeout=180
    )


@pytest.fixture(scope="module", autouse=True)
def index_is_loaded() -> None:
    try:
        response = search(mode="bm25")
    except httpx.HTTPError:
        pytest.skip("the API is not reachable")
    if response.status_code != 200 or not response.json()["hits"]:
        pytest.skip("the stack has no index loaded (see docs/setup.md)")


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
def test_every_retrieval_mode_returns_ranked_chunks(mode: str) -> None:
    response = search(mode=mode)

    body = response.json()
    assert response.status_code == 200, body
    hits = body["hits"]
    assert 0 < len(hits) <= 5
    assert [hit["rank"] for hit in hits] == list(range(1, len(hits) + 1))
    assert all(hit["doc_id"] and hit["text"] for hit in hits)
    assert body["document_ids"] == list(dict.fromkeys(hit["doc_id"] for hit in hits))
    assert body["timings_ms"]["total"] > 0


def test_hybrid_reports_where_each_retriever_ranked_a_chunk() -> None:
    hits = search(mode="hybrid", limit=30).json()["hits"]

    assert any(hit["dense_rank"] is not None for hit in hits)
    assert any(hit["bm25_rank"] is not None for hit in hits)


def test_a_source_filter_is_applied_inside_the_index() -> None:
    hits = search(mode="hybrid", filters={"source_types": ["github"]}).json()["hits"]

    assert hits
    assert {hit["source_type"] for hit in hits} == {"github"}


def test_context_is_the_passage_around_the_chunk() -> None:
    hits = search(mode="hybrid", include_context=True).json()["hits"]

    assert all(hit["context"] and hit["text"].strip()[:40] in hit["context"] for hit in hits)


def test_reranking_orders_by_the_cross_encoder_when_one_is_configured() -> None:
    response = search(mode="hybrid+rerank")

    if response.status_code == 503:
        pytest.skip("no reranker configured (set RERANKER_MODEL)")
    scores = [hit["score"] for hit in response.json()["hits"]]
    assert response.status_code == 200
    assert scores == sorted(scores, reverse=True)


def test_an_empty_query_is_rejected_before_any_search() -> None:
    response = httpx.post(f"{API_URL}/v1/search", json={"query": "  "}, timeout=30)

    assert response.status_code == 422


def test_requests_are_counted_against_the_limit_in_the_stacks_redis() -> None:
    first = search(mode="bm25")
    second = search(mode="bm25")

    if first.status_code == 401:
        pytest.skip("this stack asks for an API key")
    if "x-ratelimit-limit" not in first.headers:
        pytest.skip("the search limit is off on this stack")
    # Two requests of the same caller, counted by one Redis: the second sees one less left.
    assert (
        int(second.headers["x-ratelimit-remaining"])
        == int(first.headers["x-ratelimit-remaining"]) - 1
    )
