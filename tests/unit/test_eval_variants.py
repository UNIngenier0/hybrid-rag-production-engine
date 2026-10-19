import pytest

from fakes import (
    FakeEmbedder,
    FakeReranker,
    FakeSearcher,
    FakeSparseEncoder,
    make_hit,
)
from rag_engine.evaluation.candidates import QuestionCandidates
from rag_engine.evaluation.variants import (
    FusionSetting,
    bm25_ranking,
    dense_ranking,
    fused_ranking,
    reranked_ranking,
)
from rag_engine.retrieval.retriever import HybridRetriever, RetrievalConfig


def candidates() -> QuestionCandidates:
    dense = [make_hit(c, doc_id=f"doc-{c}") for c in "abcdef"]
    bm25 = [make_hit(c, doc_id=f"doc-{c}") for c in "dcgab"]
    return QuestionCandidates("q1", dense, bm25)


def test_single_retriever_rankings_carry_their_rank() -> None:
    assert [h.dense_rank for h in dense_ranking(candidates())] == [1, 2, 3, 4, 5, 6]
    assert [h.bm25_rank for h in bm25_ranking(candidates())] == [1, 2, 3, 4, 5]


def test_depth_cuts_each_candidate_list_before_fusing() -> None:
    shallow = fused_ranking(candidates(), FusionSetting(depth=2))
    deep = fused_ranking(candidates(), FusionSetting(depth=50))

    assert {h.point_id for h in shallow} == {"a", "b", "d", "c"}
    assert {h.point_id for h in deep} == set("abcdefg")


def test_weights_change_the_order() -> None:
    equal = [h.point_id for h in fused_ranking(candidates())]
    bm25_heavy = [h.point_id for h in fused_ranking(candidates(), FusionSetting(sparse_weight=5.0))]

    assert equal != bm25_heavy
    assert bm25_heavy[0] == "d"


async def test_the_offline_ranking_matches_what_the_service_returns() -> None:
    cands = candidates()
    searcher = FakeSearcher(dense=cands.dense, sparse=cands.bm25)
    retriever = HybridRetriever(
        searcher, FakeEmbedder(), FakeSparseEncoder(), config=RetrievalConfig()
    )

    served = await retriever.retrieve("query", mode="hybrid", limit=7)

    assert [h.point_id for h in served.hits] == [h.point_id for h in fused_ranking(cands)]


async def test_the_offline_reranking_matches_what_the_service_returns() -> None:
    cands = candidates()
    scores = {"text of a": 1.0, "text of b": 4.0, "text of c": 2.0, "text of d": 3.0}
    searcher = FakeSearcher(dense=cands.dense, sparse=cands.bm25)
    retriever = HybridRetriever(
        searcher,
        FakeEmbedder(),
        FakeSparseEncoder(),
        reranker=FakeReranker(scores),
        config=RetrievalConfig(rerank_top_k=4),
    )

    served = await retriever.retrieve("query", mode="hybrid+rerank", limit=3)
    by_point = {f"{k[-1]}": v for k, v in scores.items()}
    offline = reranked_ranking(fused_ranking(cands), by_point, pool=4)[:3]

    assert [h.point_id for h in served.hits] == [h.point_id for h in offline]


def test_reranking_only_reorders_the_pool_and_keeps_ties_in_fused_order() -> None:
    fused = fused_ranking(candidates())
    scores = {h.point_id: 0.0 for h in fused}
    scores[fused[2].point_id] = 5.0

    ranked = reranked_ranking(fused, scores, pool=4)

    assert len(ranked) == 4
    assert ranked[0].point_id == fused[2].point_id
    assert [h.point_id for h in ranked[1:]] == [
        fused[0].point_id,
        fused[1].point_id,
        fused[3].point_id,
    ]


def test_missing_scores_are_an_error_not_a_silent_zero() -> None:
    fused = fused_ranking(candidates())

    with pytest.raises(KeyError, match="no reranker score"):
        reranked_ranking(fused, {fused[0].point_id: 1.0}, pool=3)
