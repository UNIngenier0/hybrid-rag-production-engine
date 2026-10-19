import pytest

from fakes import make_hit
from rag_engine.retrieval.fusion import reciprocal_rank_fusion


def ids(hits: list) -> list[str]:  # type: ignore[type-arg]
    return [hit.point_id for hit in hits]


def test_a_chunk_found_by_both_lists_beats_single_list_chunks_at_the_same_rank() -> None:
    dense = [make_hit("a"), make_hit("b")]
    sparse = [make_hit("c"), make_hit("b")]

    fused = reciprocal_rank_fusion(dense, sparse)

    assert ids(fused)[0] == "b"
    assert set(ids(fused)) == {"a", "b", "c"}


def test_scores_follow_the_rrf_formula_with_the_ranks_of_each_list() -> None:
    fused = reciprocal_rank_fusion([make_hit("a"), make_hit("b")], [make_hit("b")], k=10)

    by_id = {hit.point_id: hit for hit in fused}
    assert by_id["a"].score == pytest.approx(1 / 11)
    assert by_id["b"].score == pytest.approx(1 / 12 + 1 / 11)


def test_each_hit_records_where_every_retriever_ranked_it() -> None:
    fused = reciprocal_rank_fusion([make_hit("a"), make_hit("b")], [make_hit("c"), make_hit("b")])

    by_id = {hit.point_id: hit for hit in fused}
    assert (by_id["a"].dense_rank, by_id["a"].bm25_rank) == (1, None)
    assert (by_id["b"].dense_rank, by_id["b"].bm25_rank) == (2, 2)
    assert (by_id["c"].dense_rank, by_id["c"].bm25_rank) == (None, 1)


def test_weights_let_one_retriever_count_for_more() -> None:
    dense = [make_hit("d")]
    sparse = [make_hit("s")]

    assert ids(reciprocal_rank_fusion(dense, sparse, sparse_weight=2.0)) == ["s", "d"]
    assert ids(reciprocal_rank_fusion(dense, sparse, dense_weight=2.0)) == ["d", "s"]


def test_ties_keep_the_dense_list_first() -> None:
    fused = reciprocal_rank_fusion([make_hit("d")], [make_hit("s")])

    assert ids(fused) == ["d", "s"]


def test_two_chunks_of_one_document_stay_separate() -> None:
    dense = [make_hit("a", doc_id="doc"), make_hit("b", doc_id="doc")]

    assert ids(reciprocal_rank_fusion(dense, [])) == ["a", "b"]


def test_an_empty_list_leaves_the_other_one_in_order() -> None:
    hits = [make_hit("a"), make_hit("b"), make_hit("c")]

    assert ids(reciprocal_rank_fusion([], hits)) == ["a", "b", "c"]
    assert reciprocal_rank_fusion([], []) == []


def test_the_inputs_are_not_modified() -> None:
    dense = [make_hit("a", score=0.9)]

    reciprocal_rank_fusion(dense, [make_hit("a", score=5.0)])

    assert dense[0].score == 0.9
    assert dense[0].dense_rank is None
