import math

import pytest

from rag_engine.evaluation.metrics import (
    QuestionScore,
    distinct_documents,
    paired_difference,
    score_question,
    summarize,
    summarize_by_type,
)


def test_distinct_documents_keep_the_order_of_the_first_chunk() -> None:
    assert distinct_documents(["b", "a", "b", "c", "a"]) == ["b", "a", "c"]


def test_a_perfect_ranking_scores_one_everywhere() -> None:
    score = score_question({"a", "b"}, ["a", "b", "x"])

    assert score == QuestionScore(
        recall=1.0, hit=True, all_gold=True, reciprocal_rank=1.0, ndcg=pytest.approx(1.0)
    )


def test_a_ranking_without_gold_documents_scores_zero() -> None:
    score = score_question({"a"}, ["x", "y"])

    assert (score.recall, score.hit, score.all_gold, score.reciprocal_rank, score.ndcg) == (
        0.0,
        False,
        False,
        0.0,
        0.0,
    )


def test_recall_is_the_share_of_gold_documents_found() -> None:
    score = score_question({"a", "b", "c", "d"}, ["x", "b", "d"])

    assert score.recall == 0.5
    assert score.hit and not score.all_gold


def test_the_reciprocal_rank_follows_the_first_gold_document() -> None:
    assert score_question({"a"}, ["x", "y", "a"]).reciprocal_rank == pytest.approx(1 / 3)


def test_ndcg_rewards_gold_documents_near_the_top() -> None:
    early = score_question({"a"}, ["a", "x", "y"]).ndcg
    late = score_question({"a"}, ["x", "y", "a"]).ndcg

    assert early == pytest.approx(1.0)
    assert late == pytest.approx(1 / math.log2(4))
    assert early > late


def test_ndcg_only_looks_at_the_first_k_documents() -> None:
    ranking = [f"x{i}" for i in range(10)] + ["a"]

    assert score_question({"a"}, ranking, ndcg_k=10).ndcg == 0.0
    assert score_question({"a"}, ranking, ndcg_k=11).ndcg > 0.0


def test_a_question_without_gold_documents_cannot_be_scored() -> None:
    with pytest.raises(ValueError, match="gold"):
        score_question(set(), ["a"])


def test_summaries_average_every_metric() -> None:
    scores = [score_question({"a"}, ["a"]), score_question({"a", "b"}, ["x"])]

    summary = summarize(scores)

    assert summary.n == 2
    assert summary.recall == 0.5
    assert summary.hit == 0.5
    assert summary.all_gold == 0.5
    assert summarize([]).n == 0


def test_summaries_can_be_split_by_question_type() -> None:
    hit, miss = score_question({"a"}, ["a"]), score_question({"a"}, ["x"])

    by_type = summarize_by_type([("basic", hit), ("basic", miss), ("semantic", miss)])

    assert list(by_type) == ["basic", "semantic"]
    assert by_type["basic"].recall == 0.5
    assert by_type["semantic"].n == 1


def test_paired_difference_has_an_interval_around_the_mean() -> None:
    a = [1.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    b = [0.0, 1.0, 0.0, 0.0, 1.0, 0.0]

    difference = paired_difference(a, b)

    assert difference.mean == pytest.approx(0.5)
    assert difference.low < difference.mean < difference.high
    assert difference.n == 6


def test_a_clear_gain_is_significant_and_noise_is_not() -> None:
    assert paired_difference([1.0] * 30, [0.0] * 30).significant  # better on every question
    noisy = paired_difference([1.0, 0.0] * 20, [0.0, 1.0] * 20)
    assert noisy.significant is False
    gain = paired_difference([1.0] * 29 + [0.0], [0.0] * 28 + [1.0, 0.0])
    assert gain.significant


def test_both_systems_must_cover_the_same_questions() -> None:
    with pytest.raises(ValueError, match="same questions"):
        paired_difference([1.0], [1.0, 0.0])
