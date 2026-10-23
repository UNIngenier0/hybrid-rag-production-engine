import pytest

from fakes import make_hit
from rag_engine.evaluation.abstention import (
    Evidence,
    auc,
    build_report,
    operating_point,
    operating_points,
    render,
    threshold_for,
    top_score,
)
from rag_engine.evaluation.candidates import QuestionCandidates


def test_top_score_is_the_best_score_of_the_reranked_pool() -> None:
    found = QuestionCandidates(
        "q",
        dense=[make_hit("a", doc_id="x"), make_hit("b", doc_id="y")],
        bm25=[make_hit("c", doc_id="z")],
    )

    assert top_score(found, {"a": -3.0, "b": 2.5, "c": 0.5}) == 2.5


def test_top_score_only_looks_at_the_pool() -> None:
    found = QuestionCandidates(
        "q",
        dense=[make_hit("a", doc_id="x"), make_hit("b", doc_id="y")],
        bm25=[],
    )

    # "b" is fused second, outside a pool of one chunk, so its high score does not count.
    assert top_score(found, {"a": -1.0, "b": 9.0}, pool=1) == -1.0


def test_top_score_needs_a_score_for_every_pooled_chunk() -> None:
    found = QuestionCandidates("q", dense=[make_hit("a", doc_id="x")], bm25=[])

    with pytest.raises(KeyError):
        top_score(found, {})


def test_auc_is_one_when_the_groups_do_not_overlap() -> None:
    assert auc([5.0, 6.0], [1.0, 2.0]) == 1.0


def test_auc_is_half_when_the_scores_carry_no_information() -> None:
    assert auc([1.0, 2.0], [1.0, 2.0]) == 0.5  # identical groups: ties count half
    assert auc([1.0], [1.0]) == 0.5


def test_auc_is_zero_when_unanswerable_questions_score_higher() -> None:
    assert auc([1.0, 2.0], [5.0, 6.0]) == 0.0


def test_auc_needs_both_groups() -> None:
    with pytest.raises(ValueError):
        auc([], [1.0])


def test_threshold_refuses_at_most_the_allowed_share_of_answerable_questions() -> None:
    answerable = [float(score) for score in range(100)]

    threshold = threshold_for(answerable, 0.05)

    assert sum(score < threshold for score in answerable) == 5


def test_threshold_for_no_false_abstention_refuses_no_answerable_question() -> None:
    answerable = [3.0, 1.0, 2.0]

    assert not any(score < threshold_for(answerable, 0.0) for score in answerable)


def test_threshold_rejects_a_share_that_is_not_a_share() -> None:
    with pytest.raises(ValueError):
        threshold_for([1.0], 1.0)


def test_operating_point_counts_both_kinds_of_mistake() -> None:
    point = operating_point(2.0, answerable=[1.0, 3.0, 4.0, 5.0], unanswerable=[0.0, 1.5, 6.0])

    assert (point.refused_answerable, point.caught_unanswerable) == (1, 2)
    assert point.false_abstain == 0.25
    assert point.caught == pytest.approx(2 / 3)


def test_operating_points_cover_every_requested_rate() -> None:
    answerable = [float(score) for score in range(100)]

    points = operating_points(answerable, [-1.0, 50.0], rates=(0.0, 0.1))

    assert [p.refused_answerable for p in points] == [0, 10]
    assert [p.caught_unanswerable for p in points] == [1, 1]


def evidence() -> list[Evidence]:
    answerable = [
        Evidence(f"a{i}", "basic", float(i), correct=i >= 3) for i in range(10)
    ]  # scores 0..9, the first three answers are wrong
    unanswerable = [Evidence(f"u{i}", "info_not_found", float(i)) for i in range(4)]
    high_level = [Evidence("h0", "high_level", 1.5), Evidence("h1", "high_level", 8.0)]
    return answerable + unanswerable + high_level


def test_the_report_summarises_the_scores_of_every_type() -> None:
    report = build_report(evidence())

    rows = {row.question_type: row for row in report.by_type}
    assert (rows["basic"].n, rows["basic"].low, rows["basic"].high) == (10, 0.0, 9.0)
    assert rows["basic"].median == 4.5
    assert rows["info_not_found"].n == 4


def test_the_report_measures_how_well_the_score_finds_unanswerable_questions() -> None:
    report = build_report(evidence(), rates=(0.0, 0.3))

    assert report.auc_unanswerable == pytest.approx(0.8)
    first, second = report.points
    assert (first.refused_answerable, first.caught_unanswerable) == (0, 0)
    assert (second.refused_answerable, second.caught_unanswerable) == (3, 3)
    assert report.high_level_refused == {first.threshold: 0, second.threshold: 1}


def test_the_report_counts_the_wrong_answers_a_threshold_would_stop() -> None:
    report = build_report(evidence(), rates=(0.3,))

    (protection,) = report.protection
    assert (protection.wrong_avoided, protection.wrong_total) == (3, 3)
    assert (protection.right_lost, protection.right_total) == (0, 7)
    assert report.auc_correct == 1.0


def test_without_judged_answers_there_is_nothing_to_measure_the_protection_on() -> None:
    report = build_report([Evidence("u", "info_not_found", 1.0), Evidence("b", "basic", 2.0)])

    assert report.points == [] and report.protection == [] and report.auc_unanswerable is None


def test_the_markdown_names_every_table() -> None:
    text = render(build_report(evidence(), rates=(0.0, 0.3)))

    assert "# Abstention by evidence strength" in text
    assert "| info_not_found | 4 |" in text
    assert "Telling `info_not_found` questions from answerable ones" in text
    assert "Stopping wrong answers" in text
    assert "| 30% (3) | 3.0000 | 3 of 4 | 1 of 2 |" in text


def test_the_report_tabulates_what_answers_say_about_themselves_against_their_correctness() -> None:
    items = [
        Evidence("a1", "basic", 0.9, True, "answered"),
        Evidence("a2", "basic", 0.9, True, "answered"),
        Evidence("a3", "basic", 0.5, False, "partial"),
        Evidence("a4", "basic", 0.4, True, "partial"),
        Evidence("u1", "info_not_found", 0.8, True, "partial"),
        Evidence("h1", "high_level", 0.7, False, "answered"),
    ]

    table = {(r.group, r.status): (r.n, r.correct) for r in build_report(items).statuses}

    assert table == {
        ("answerable", "answered"): (2, 2),
        ("answerable", "partial"): (2, 1),
        ("info_not_found", "partial"): (1, 1),
        ("high_level", "answered"): (1, 0),
    }


def test_unanswerable_types_are_never_counted_among_the_answerable_ones() -> None:
    items = [Evidence("a", "basic", 0.9, True), Evidence("u", "info_not_found", 0.1, True)]

    report = build_report(items, rates=(0.0,))

    assert report.protection == []  # one right answer and no wrong one: nothing to compare
    assert report.auc_unanswerable == 1.0  # only "a" is answerable
