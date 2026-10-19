import httpx

from fakes import FakeEmbedder, FakePassageStore, FakeSearcher, FakeSparseEncoder, make_hit
from rag_engine.api.main import create_app
from rag_engine.core.config import Settings
from rag_engine.core.errors import RetrievalUnavailableError
from rag_engine.core.models import RetrievalMode, RetrievalResult, SearchFilters
from rag_engine.retrieval.retriever import HybridRetriever


class RecordingRetriever:
    """A retriever that answers with fixed hits and remembers how it was called."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[dict[str, object]] = []

    async def retrieve(
        self,
        query: str,
        *,
        mode: RetrievalMode = "hybrid",
        filters: SearchFilters | None = None,
        limit: int | None = None,
        include_context: bool = False,
    ) -> RetrievalResult:
        if self.error:
            raise self.error
        self.calls.append(
            {
                "query": query,
                "mode": mode,
                "filters": filters,
                "limit": limit,
                "include_context": include_context,
            }
        )
        hits = [
            make_hit("p1", doc_id="d1", score=0.9, dense_rank=1, section="Intro"),
            make_hit("p2", doc_id="d1", score=0.5, bm25_rank=3),
            make_hit("p3", doc_id="d2", score=0.4, context="parent text"),
        ]
        return RetrievalResult(query=query, mode=mode, hits=hits, timings_ms={"total": 1.5})


def client_for(settings: Settings, retriever: RecordingRetriever) -> httpx.AsyncClient:
    app = create_app(settings, probes=[], retriever=retriever)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_search_returns_ranked_hits_and_distinct_documents(settings: Settings) -> None:
    async with client_for(settings, RecordingRetriever()) as client:
        response = await client.post("/v1/search", json={"query": "rotate the key"})

    body = response.json()
    assert response.status_code == 200
    assert body["mode"] == "hybrid"
    assert body["document_ids"] == ["d1", "d2"]
    assert body["timings_ms"] == {"total": 1.5}
    assert [hit["rank"] for hit in body["hits"]] == [1, 2, 3]
    first = body["hits"][0]
    assert first["chunk_id"] == "p1"
    assert first["section"] == "Intro"
    assert first["dense_rank"] == 1
    assert first["bm25_rank"] is None
    assert body["hits"][2]["context"] == "parent text"


async def test_the_request_options_reach_the_retriever(settings: Settings) -> None:
    retriever = RecordingRetriever()

    async with client_for(settings, retriever) as client:
        await client.post(
            "/v1/search",
            json={
                "query": "  rotate the key  ",
                "mode": "bm25",
                "limit": 5,
                "filters": {"source_types": ["slack", "jira"], "doc_ids": ["d9"]},
                "include_context": True,
            },
        )

    assert retriever.calls == [
        {
            "query": "rotate the key",
            "mode": "bm25",
            "filters": SearchFilters(source_types=("slack", "jira"), doc_ids=("d9",)),
            "limit": 5,
            "include_context": True,
        }
    ]


async def test_defaults_are_hybrid_with_ten_results_and_no_filters(settings: Settings) -> None:
    retriever = RecordingRetriever()

    async with client_for(settings, retriever) as client:
        await client.post("/v1/search", json={"query": "q"})

    assert retriever.calls == [
        {
            "query": "q",
            "mode": "hybrid",
            "filters": SearchFilters(),
            "limit": 10,
            "include_context": False,
        }
    ]


async def test_invalid_requests_are_rejected_before_retrieval(settings: Settings) -> None:
    retriever = RecordingRetriever()
    invalid = [
        {},
        {"query": ""},
        {"query": "   "},
        {"query": "x" * 2001},
        {"query": "q", "mode": "semantic"},
        {"query": "q", "limit": 0},
        {"query": "q", "limit": 101},
        {"query": "q", "filters": {"source_types": ["s"] * 21}},
    ]

    async with client_for(settings, retriever) as client:
        statuses = [(await client.post("/v1/search", json=body)).status_code for body in invalid]

    assert statuses == [422] * len(invalid)
    assert retriever.calls == []


async def test_unavailable_retrieval_is_a_503_without_internals(settings: Settings) -> None:
    retriever = RecordingRetriever(error=RetrievalUnavailableError("ConnectionError: qdrant:6333"))

    async with client_for(settings, retriever) as client:
        response = await client.post("/v1/search", json={"query": "q"})

    assert response.status_code == 503
    assert response.json() == {"detail": "retrieval unavailable"}
    assert "qdrant" not in response.text


async def test_the_api_serves_a_real_retriever_end_to_end(settings: Settings) -> None:
    searcher = FakeSearcher(
        dense=[make_hit("a", doc_id="d1"), make_hit("b", doc_id="d2")],
        sparse=[make_hit("b", doc_id="d2")],
    )
    retriever = HybridRetriever(
        searcher, FakeEmbedder(), FakeSparseEncoder(), passages=FakePassageStore()
    )
    app = create_app(settings, probes=[], retriever=retriever)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/search", json={"query": "q", "include_context": True, "limit": 5}
        )

    body = response.json()
    assert response.status_code == 200
    assert body["document_ids"] == ["d2", "d1"]
    assert body["hits"][0]["context"] == "d2[0:40]"
