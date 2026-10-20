import asyncio
import threading

import numpy as np
import pytest
from numpy.typing import NDArray

from fakes import (
    FakeEmbedder,
    FakePassageStore,
    FakeReranker,
    FakeSearcher,
    FakeSparseEncoder,
    make_hit,
)
from rag_engine.core.errors import RetrievalUnavailableError
from rag_engine.core.models import SearchFilters, SparseVector
from rag_engine.retrieval.retriever import HybridRetriever, RetrievalConfig


def ids(result) -> list[str]:  # type: ignore[no-untyped-def]
    return [hit.point_id for hit in result.hits]


def make_retriever(
    searcher: FakeSearcher, *, passages: FakePassageStore | None = None, **config: float
) -> HybridRetriever:
    return HybridRetriever(
        searcher,
        FakeEmbedder(),
        FakeSparseEncoder(),
        passages=passages,
        config=RetrievalConfig(**config),  # type: ignore[arg-type]
    )


async def test_hybrid_fuses_both_lists_and_reports_each_stage() -> None:
    searcher = FakeSearcher(
        dense=[make_hit("a"), make_hit("b")], sparse=[make_hit("c"), make_hit("b")]
    )

    result = await make_retriever(searcher).retrieve("how do I rotate the key?")

    assert result.mode == "hybrid"
    assert ids(result)[0] == "b"
    assert set(ids(result)) == {"a", "b", "c"}
    assert set(result.timings_ms) == {
        "embed",
        "dense_search",
        "encode",
        "bm25_search",
        "fuse",
        "total",
    }


async def test_dense_mode_only_runs_the_dense_search() -> None:
    searcher = FakeSearcher(
        dense=[make_hit("a", score=0.8), make_hit("b", score=0.7)], sparse=[make_hit("z")]
    )

    result = await make_retriever(searcher).retrieve("query", mode="dense")

    assert [call[0] for call in searcher.calls] == ["dense"]
    assert ids(result) == ["a", "b"]
    assert [hit.dense_rank for hit in result.hits] == [1, 2]
    assert all(hit.bm25_rank is None for hit in result.hits)
    assert result.hits[0].score == 0.8  # the native similarity, not an RRF score
    assert "bm25_search" not in result.timings_ms


async def test_bm25_mode_only_runs_the_lexical_search() -> None:
    searcher = FakeSearcher(dense=[make_hit("z")], sparse=[make_hit("a"), make_hit("b")])

    result = await make_retriever(searcher).retrieve("query", mode="bm25")

    assert [call[0] for call in searcher.calls] == ["sparse"]
    assert ids(result) == ["a", "b"]
    assert [hit.bm25_rank for hit in result.hits] == [1, 2]
    assert "embed" not in result.timings_ms


async def test_the_query_is_embedded_and_encoded_for_each_retriever() -> None:
    searcher = FakeSearcher()
    embedder = FakeEmbedder()
    retriever = HybridRetriever(searcher, embedder, FakeSparseEncoder())

    await retriever.retrieve("find the runbook")

    assert searcher.dense_vectors == [embedder.embed_query("find the runbook").tolist()]
    assert searcher.sparse_vectors == [FakeSparseEncoder().encode_query("find the runbook")]


async def test_filters_reach_both_searches() -> None:
    searcher = FakeSearcher()
    filters = SearchFilters(source_types=("slack",), doc_ids=("d1",))

    await make_retriever(searcher).retrieve("query", filters=filters)

    assert [call[2] for call in searcher.calls] == [filters, filters]


async def test_every_mode_returns_the_same_number_of_chunks() -> None:
    hits = [make_hit(str(i)) for i in range(20)]
    searcher = FakeSearcher(dense=hits, sparse=list(reversed(hits)))
    retriever = make_retriever(searcher)

    for mode in ("dense", "bm25", "hybrid"):
        result = await retriever.retrieve("query", mode=mode, limit=7)  # type: ignore[arg-type]
        assert len(result.hits) == 7


async def test_default_limit_comes_from_the_config() -> None:
    hits = [make_hit(str(i)) for i in range(10)]
    searcher = FakeSearcher(dense=hits, sparse=hits)

    result = await make_retriever(searcher, default_limit=4).retrieve("query")

    assert len(result.hits) == 4


async def test_each_retriever_searches_at_least_as_deep_as_the_limit() -> None:
    searcher = FakeSearcher()
    retriever = make_retriever(searcher, dense_top_k=50, sparse_top_k=20)

    await retriever.retrieve("query", limit=30)
    await retriever.retrieve("query", limit=80)

    # The two searches start in whichever order their query vectors are ready, so compare per query.
    depths = [(kind, limit) for kind, limit, _ in searcher.calls]
    assert sorted(depths[:2]) == [("dense", 50), ("sparse", 30)]
    assert sorted(depths[2:]) == [("dense", 80), ("sparse", 80)]


async def test_context_is_the_parent_passage_of_each_hit() -> None:
    searcher = FakeSearcher(dense=[make_hit("a", doc_id="d1", parent_start=5, parent_end=50)])
    passages = FakePassageStore()

    result = await make_retriever(searcher, passages=passages).retrieve(
        "query", mode="dense", include_context=True
    )

    assert result.hits[0].context == "d1[5:50]"
    assert "context" in result.timings_ms


async def test_context_is_not_fetched_unless_asked_for() -> None:
    searcher = FakeSearcher(dense=[make_hit("a")])
    passages = FakePassageStore()

    result = await make_retriever(searcher, passages=passages).retrieve("query", mode="dense")

    assert passages.requests == []
    assert result.hits[0].context is None


async def test_a_missing_parent_passage_leaves_the_context_empty() -> None:
    searcher = FakeSearcher(dense=[make_hit("a", doc_id="gone"), make_hit("b", doc_id="here")])
    passages = FakePassageStore(missing_docs=frozenset({"gone"}))

    result = await make_retriever(searcher, passages=passages).retrieve(
        "query", mode="dense", include_context=True
    )

    assert [hit.context for hit in result.hits] == [None, "here[0:40]"]


async def test_context_needs_a_passage_store() -> None:
    retriever = make_retriever(FakeSearcher())

    with pytest.raises(ValueError, match="passage store"):
        await retriever.retrieve("query", include_context=True)


@pytest.mark.parametrize("query", ["", "   \n"])
async def test_an_empty_query_is_rejected(query: str) -> None:
    with pytest.raises(ValueError, match="empty"):
        await make_retriever(FakeSearcher()).retrieve(query)


async def test_a_limit_below_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="limit"):
        await make_retriever(FakeSearcher()).retrieve("query", limit=0)


async def test_document_ids_list_each_document_once_in_rank_order() -> None:
    searcher = FakeSearcher(
        dense=[make_hit("a", doc_id="d2"), make_hit("b", doc_id="d1"), make_hit("c", doc_id="d2")]
    )

    result = await make_retriever(searcher).retrieve("query", mode="dense")

    assert result.document_ids == ["d2", "d1"]


class BlockingEmbedder(FakeEmbedder):
    """Records the thread that embedded, to show the event loop is not blocked."""

    def __init__(self) -> None:
        super().__init__()
        self.thread: int | None = None

    def embed_query(self, text: str) -> NDArray[np.float32]:
        self.thread = threading.get_ident()
        return super().embed_query(text)


async def test_embedding_runs_off_the_event_loop() -> None:
    embedder = BlockingEmbedder()
    retriever = HybridRetriever(FakeSearcher(), embedder, FakeSparseEncoder())

    await retriever.retrieve("query")

    assert embedder.thread is not None
    assert embedder.thread != threading.get_ident()


class SlowSearcher(FakeSearcher):
    def __init__(self) -> None:
        super().__init__()
        self.in_flight = 0
        self.max_in_flight = 0

    async def _track(self) -> None:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(0.05)
        self.in_flight -= 1

    async def search_dense(self, vector, *, limit, filters=None):  # type: ignore[no-untyped-def]
        await self._track()
        return await super().search_dense(vector, limit=limit, filters=filters)

    async def search_sparse(self, vector: SparseVector, *, limit, filters=None):  # type: ignore[no-untyped-def]
        await self._track()
        return await super().search_sparse(vector, limit=limit, filters=filters)


async def test_the_two_searches_run_at_the_same_time() -> None:
    searcher = SlowSearcher()

    await make_retriever(searcher).retrieve("query")

    assert searcher.max_in_flight == 2


def rerank_retriever(
    searcher: FakeSearcher, reranker: FakeReranker | None, **config: int
) -> HybridRetriever:
    return HybridRetriever(
        searcher,
        FakeEmbedder(),
        FakeSparseEncoder(),
        reranker=reranker,
        config=RetrievalConfig(**config),
    )


async def test_rerank_reorders_the_fused_candidates_by_the_cross_encoder() -> None:
    searcher = FakeSearcher(
        dense=[make_hit("a"), make_hit("b"), make_hit("c")], sparse=[make_hit("a"), make_hit("b")]
    )
    reranker = FakeReranker({"text of c": 9.0, "text of b": 5.0, "text of a": 1.0})

    result = await rerank_retriever(searcher, reranker).retrieve(
        "query", mode="hybrid+rerank", limit=2
    )

    assert ids(result) == ["c", "b"]
    assert [hit.score for hit in result.hits] == [9.0, 5.0]
    assert "rerank" in result.timings_ms and "fuse" in result.timings_ms


async def test_reranked_hits_keep_where_each_stage_ranked_them() -> None:
    searcher = FakeSearcher(dense=[make_hit("a"), make_hit("b")], sparse=[make_hit("b")])
    reranker = FakeReranker({"text of a": 2.0, "text of b": 1.0})

    result = await rerank_retriever(searcher, reranker).retrieve("query", mode="hybrid+rerank")

    by_id = {hit.point_id: hit for hit in result.hits}
    assert (by_id["b"].dense_rank, by_id["b"].bm25_rank, by_id["b"].fusion_rank) == (2, 1, 1)
    assert (by_id["a"].dense_rank, by_id["a"].bm25_rank, by_id["a"].fusion_rank) == (1, None, 2)


async def test_only_the_first_rerank_top_k_fused_chunks_are_scored() -> None:
    hits = [make_hit(str(i)) for i in range(10)]
    reranker = FakeReranker({})

    await rerank_retriever(FakeSearcher(dense=hits), reranker, rerank_top_k=4).retrieve(
        "query", mode="hybrid+rerank", limit=2
    )

    assert len(reranker.calls[0][1]) == 4


async def test_a_larger_limit_widens_the_reranked_candidates() -> None:
    hits = [make_hit(str(i)) for i in range(10)]
    reranker = FakeReranker({})

    result = await rerank_retriever(FakeSearcher(dense=hits), reranker, rerank_top_k=4).retrieve(
        "query", mode="hybrid+rerank", limit=7
    )

    assert len(reranker.calls[0][1]) == 7
    assert len(result.hits) == 7


async def test_the_reranker_reads_the_title_and_section_with_the_chunk() -> None:
    hit = make_hit("a", title="Runbook", section="Rotation")
    reranker = FakeReranker({})

    await rerank_retriever(FakeSearcher(dense=[hit]), reranker).retrieve(
        "how to rotate", mode="hybrid+rerank"
    )

    query, passages = reranker.calls[0]
    assert query == "how to rotate"
    assert passages == ["Runbook\nRotation\n\ntext of a"]


async def test_equal_rerank_scores_keep_the_fused_order() -> None:
    searcher = FakeSearcher(dense=[make_hit("a"), make_hit("b"), make_hit("c")])

    result = await rerank_retriever(searcher, FakeReranker({})).retrieve(
        "query", mode="hybrid+rerank"
    )

    assert ids(result) == ["a", "b", "c"]


async def test_rerank_mode_needs_a_reranker() -> None:
    with pytest.raises(RetrievalUnavailableError, match="reranker"):
        await rerank_retriever(FakeSearcher(), None).retrieve("query", mode="hybrid+rerank")


async def test_the_reranker_is_fed_a_pool_fused_with_its_own_bm25_weight() -> None:
    dense, sparse = [make_hit("a"), make_hit("b")], [make_hit("c"), make_hit("d")]
    weights = {"sparse_weight": 5.0, "rerank_sparse_weight": 1.0}

    hybrid = await make_retriever(FakeSearcher(dense=dense, sparse=sparse), **weights).retrieve(
        "query"
    )
    pool = await rerank_retriever(
        FakeSearcher(dense=dense, sparse=sparse), FakeReranker({}), **weights
    ).retrieve("query", mode="hybrid+rerank")

    assert ids(hybrid) == ["c", "d", "a", "b"]  # BM25 counts five times as much
    assert ids(pool) == ["a", "c", "b", "d"]  # equal weights, and the reranker ties keep that order
