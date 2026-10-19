import pytest

from fakes import FakeEmbedder, FakeSearcher, FakeSparseEncoder, make_hit
from rag_engine.api.wiring import LazyRetriever
from rag_engine.core.config import Settings
from rag_engine.core.errors import RetrievalUnavailableError
from rag_engine.retrieval.retriever import HybridRetriever


def working_retriever() -> HybridRetriever:
    return HybridRetriever(FakeSearcher(dense=[make_hit("a")]), FakeEmbedder(), FakeSparseEncoder())


async def test_nothing_is_built_until_the_first_query(settings: Settings) -> None:
    builds: list[int] = []

    class Counting(LazyRetriever):
        async def _build(self) -> HybridRetriever:
            builds.append(1)
            return working_retriever()

    lazy = Counting(settings)
    assert builds == []

    await lazy.retrieve("first")
    await lazy.retrieve("second")

    assert builds == [1]


async def test_a_failed_start_is_reported_as_unavailable_and_retried(settings: Settings) -> None:
    attempts: list[int] = []

    class Flaky(LazyRetriever):
        async def _build(self) -> HybridRetriever:
            attempts.append(1)
            if len(attempts) == 1:
                raise ImportError("sentence_transformers is not installed")
            return working_retriever()

    lazy = Flaky(settings)

    with pytest.raises(RetrievalUnavailableError, match="ImportError"):
        await lazy.retrieve("query")
    result = await lazy.retrieve("query", mode="dense")

    assert [hit.point_id for hit in result.hits] == ["a"]
    assert len(attempts) == 2


async def test_closing_before_any_query_is_harmless(settings: Settings) -> None:
    await LazyRetriever(settings).aclose()
