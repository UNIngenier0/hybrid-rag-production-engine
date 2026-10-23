"""A slow or unreachable backend is "retrieval unavailable" (a 503), never a crash (a 500)."""

from unittest.mock import AsyncMock

import asyncpg
import grpc
import httpx
import pytest
from grpc.aio import AioRpcError
from grpc.aio import Metadata as GrpcMetadata
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from fakes import FakeEmbedder, FakeLLM, FakePassageStore, FakeSparseEncoder
from rag_engine.adapters.postgres_documents import PostgresDocumentStore
from rag_engine.adapters.qdrant_index import QdrantChunkIndex
from rag_engine.api.main import create_app
from rag_engine.core.config import Settings
from rag_engine.core.errors import RetrievalUnavailableError
from rag_engine.core.models import PassageSpan, SparseVector
from rag_engine.generation.answer import AnswerGenerator
from rag_engine.generation.query import QueryService
from rag_engine.generation.verify import CitationVerifier
from rag_engine.retrieval.retriever import HybridRetriever


def deadline_exceeded() -> AioRpcError:
    return AioRpcError(
        grpc.StatusCode.DEADLINE_EXCEEDED, GrpcMetadata(), GrpcMetadata(), "Deadline Exceeded"
    )


FAILURES = [
    deadline_exceeded(),
    ResponseHandlingException(httpx.ReadTimeout("timed out")),
    UnexpectedResponse(503, "Service Unavailable", b"", {}),
    ConnectionRefusedError("qdrant is down"),
    TimeoutError(),
]


@pytest.mark.parametrize("failure", FAILURES, ids=lambda f: type(f).__name__)
async def test_a_failed_dense_or_sparse_search_is_reported_as_unavailable(
    failure: Exception,
) -> None:
    client = AsyncMock()
    client.query_points.side_effect = failure
    index = QdrantChunkIndex(client, "chunks")

    with pytest.raises(RetrievalUnavailableError, match="qdrant"):
        await index.search_dense([0.1, 0.2], limit=5)
    with pytest.raises(RetrievalUnavailableError, match="qdrant"):
        await index.search_sparse(SparseVector(indices=[1], values=[1.0]), limit=5)


async def test_a_programming_error_is_not_disguised_as_an_outage() -> None:
    client = AsyncMock()
    client.query_points.side_effect = KeyError("bug")

    with pytest.raises(KeyError):
        await QdrantChunkIndex(client, "chunks").search_dense([0.1], limit=5)


@pytest.mark.parametrize(
    "failure",
    [asyncpg.PostgresConnectionError("gone"), asyncpg.InterfaceError("closed"), TimeoutError()],
    ids=lambda f: type(f).__name__,
)
async def test_a_failed_passage_lookup_is_reported_as_unavailable(failure: Exception) -> None:
    pool = AsyncMock()
    pool.fetch.side_effect = failure
    store = PostgresDocumentStore.__new__(PostgresDocumentStore)
    store._pool = pool  # type: ignore[attr-defined]

    with pytest.raises(RetrievalUnavailableError, match="postgres"):
        await store.passages([PassageSpan("d", "sha", 0, 10)])


async def test_the_endpoints_answer_503_when_the_index_times_out(settings: Settings) -> None:
    index = QdrantChunkIndex(
        AsyncMock(query_points=AsyncMock(side_effect=deadline_exceeded())), "c"
    )
    retriever = HybridRetriever(
        index, FakeEmbedder(), FakeSparseEncoder(), passages=FakePassageStore()
    )
    llm = FakeLLM("never used")
    service = QueryService(AnswerGenerator(retriever, llm), CitationVerifier(llm))
    app = create_app(settings, probes=[], retriever=retriever, query_service=service)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as http:
        search = await http.post("/v1/search", json={"query": "upload limit"})
        query = await http.post("/v1/query", json={"query": "upload limit"})
        metrics = (await http.get("/metrics")).text

    assert (search.status_code, query.status_code) == (503, 503)
    assert search.json() == {"detail": "retrieval unavailable"}
    assert 'route="/v1/search",status="503"' in metrics
