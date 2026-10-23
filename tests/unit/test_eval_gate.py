import json
from pathlib import Path

import pytest

from fakes import make_hit
from rag_engine.core.config import Settings
from rag_engine.evaluation.candidates import QuestionCandidates
from rag_engine.evaluation.gate import (
    GateError,
    Pipeline,
    abstention_report,
    build_fixture,
    check_abstention_report,
    check_answer_report,
    load_fixture,
    make_baseline,
    measure,
    run_gate,
    shipped_pipeline,
    violations,
)
from rag_engine.evaluation.questions import Question

PIPELINE = Pipeline(
    depth=50, rrf_k=60, dense_weight=1.0, sparse_weight=1.0, rerank_sparse_weight=1.0, rerank_pool=3
)


def question(question_id: str, gold: tuple[str, ...], qtype: str = "basic") -> Question:
    return Question(question_id, qtype, "text", gold, (), "", ())


def candidates(question_id: str) -> QuestionCandidates:
    # Dense finds the gold document in third place, BM25 finds a decoy first.
    return QuestionCandidates(
        question_id,
        dense=[
            make_hit("d1", doc_id="x"),
            make_hit("d2", doc_id="y"),
            make_hit("g", doc_id="gold"),
        ],
        bm25=[make_hit("b1", doc_id="z"), make_hit("g", doc_id="gold")],
    )


def fixture(tmp_path: Path, scores: dict[str, float] | None = None) -> Path:
    scores = scores or {"d1": 0.1, "d2": 0.2, "g": 0.9, "b1": 0.3}
    out = tmp_path / "fixture.jsonl.gz"
    build_fixture(
        [question("q1", ("gold",))],
        {"q1": candidates("q1")},
        {"q1": scores},
        out,
        reranker="some/reranker",
    )
    return out


def test_the_fixture_keeps_each_chunks_document_rank_and_reranker_score(tmp_path: Path) -> None:
    meta, cases = load_fixture(fixture(tmp_path))

    assert meta["reranker"] == "some/reranker"
    (case,) = cases
    assert [hit.doc_id for hit in case.candidates.dense] == ["x", "y", "gold"]
    assert [hit.doc_id for hit in case.candidates.bm25] == ["z", "gold"]
    # A chunk found by both retrievers is one chunk, so it keeps one score.
    assert case.candidates.dense[2].point_id == case.candidates.bm25[1].point_id
    assert case.scores[case.candidates.dense[2].point_id] == 0.9
    assert case.question.gold_doc_ids == ("gold",)


def test_questions_without_candidates_or_scores_are_left_out(tmp_path: Path) -> None:
    out = tmp_path / "fixture.jsonl.gz"

    count = build_fixture(
        [question("q1", ("gold",)), question("q2", ("gold",)), question("q3", ("gold",))],
        {"q1": candidates("q1"), "q2": candidates("q2")},
        {"q1": {"d1": 1.0}, "q3": {"d1": 1.0}},
        out,
        reranker="r",
    )

    assert count == 1


def test_measure_reports_every_system_at_every_budget(tmp_path: Path) -> None:
    _, cases = load_fixture(fixture(tmp_path))

    result = measure(cases, PIPELINE)

    assert result["questions"] == 1
    recall = {
        system: result["metrics"][system]["5"]["overall"]["recall"]
        for system in ("dense", "bm25", "hybrid", "hybrid+rerank")
    }
    assert recall == {"dense": 1.0, "bm25": 1.0, "hybrid": 1.0, "hybrid+rerank": 1.0}
    top_one = measure(cases, PIPELINE)["metrics"]["hybrid+rerank"]["5"]["overall"]["mrr"]
    assert top_one == 1.0  # the reranker puts the gold chunk first
    assert result["metrics"]["dense"]["5"]["by_type"] == {"basic": 1.0}


def test_the_reranker_moves_the_gold_document_up(tmp_path: Path) -> None:
    _, cases = load_fixture(fixture(tmp_path))

    metrics = measure(cases, PIPELINE)["metrics"]

    assert metrics["dense"]["5"]["overall"]["mrr"] == pytest.approx(1 / 3)
    assert metrics["hybrid+rerank"]["5"]["overall"]["mrr"] == 1.0


def test_a_pool_without_reranker_scores_stops_the_gate_instead_of_guessing(
    tmp_path: Path,
) -> None:
    _, cases = load_fixture(fixture(tmp_path, scores={"g": 0.9}))

    with pytest.raises(GateError, match="no reranker score"):
        measure(cases, PIPELINE)


def baseline_for(tmp_path: Path) -> tuple[Path, Path]:
    path = fixture(tmp_path)
    meta, cases = load_fixture(path)
    baseline = make_baseline(path, meta, PIPELINE, measure(cases, PIPELINE))
    target = tmp_path / "baseline.json"
    target.write_text(json.dumps(baseline), encoding="utf-8")
    return path, target


def test_a_drop_beyond_the_tolerance_is_a_violation_and_an_improvement_is_not() -> None:
    def numbers(recall: float) -> dict[str, object]:
        overall = {"recall": recall, "hit": 1.0, "mrr": 1.0, "ndcg": 1.0}
        return {"metrics": {"hybrid": {"5": {"overall": overall, "by_type": {"basic": recall}}}}}

    assert violations(numbers(0.70), numbers(0.70)) == []
    assert violations(numbers(0.70), numbers(0.80)) == []
    assert violations(numbers(0.70), numbers(0.698)) == []  # inside the tolerance
    found = violations(numbers(0.70), numbers(0.60))
    assert {v.metric for v in found} == {"hybrid @5 recall", "hybrid @5 recall on basic"}
    assert all(v.delta == pytest.approx(-0.10) for v in found)


def test_a_type_that_disappears_counts_as_a_drop_to_zero() -> None:
    before = {
        "metrics": {"hybrid": {"5": {"overall": {}, "by_type": {"basic": 0.5, "semantic": 0.5}}}}
    }
    after = {"metrics": {"hybrid": {"5": {"overall": {}, "by_type": {"basic": 0.5}}}}}

    assert [v.metric for v in violations(before, after)] == ["hybrid @5 recall on semantic"]


def test_the_gate_passes_against_its_own_baseline(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    path, baseline = baseline_for(tmp_path)
    monkeypatch.setattr("rag_engine.evaluation.gate.shipped_pipeline", lambda: PIPELINE)

    result = run_gate(path, baseline)

    assert result.passed
    assert "No regression." in result.text


def test_the_gate_fails_when_the_pipeline_it_replays_got_worse(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    path, baseline = baseline_for(tmp_path)
    # Fusion keeps only the first dense chunk, so the gold document is no longer found.
    worse = Pipeline(1, 60, 1.0, 1.0, 1.0, 1)
    monkeypatch.setattr("rag_engine.evaluation.gate.shipped_pipeline", lambda: worse)
    data = json.loads(baseline.read_text(encoding="utf-8"))
    data["pipeline"] = worse.describe()  # compare the numbers, not the settings
    baseline.write_text(json.dumps(data), encoding="utf-8")

    result = run_gate(path, baseline)

    assert not result.passed
    assert any("hybrid" in violation.metric for violation in result.violations)


def test_a_changed_fixture_or_pipeline_is_reported_even_if_the_numbers_hold(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    path, baseline = baseline_for(tmp_path)
    monkeypatch.setattr(
        "rag_engine.evaluation.gate.shipped_pipeline",
        lambda: Pipeline(50, 60, 1.0, 1.5, 1.0, 3),
    )

    result = run_gate(path, baseline)

    assert not result.passed
    assert "baseline was measured with" in result.text


def test_the_shipped_pipeline_is_read_from_the_settings_defaults() -> None:
    pipeline = shipped_pipeline()
    defaults = Settings.model_fields

    assert pipeline.sparse_weight == defaults["rrf_sparse_weight"].default
    assert pipeline.rerank_sparse_weight == defaults["rerank_sparse_weight"].default
    assert pipeline.rerank_pool == defaults["rerank_top_k"].default


def test_the_committed_baseline_still_holds() -> None:
    """The regression gate itself: the committed fixture replayed with the shipped settings."""
    root = Path(__file__).resolve().parents[2]

    result = run_gate(
        root / "eval/baselines/gate-fixture.jsonl.gz", root / "eval/baselines/retrieval.json"
    )

    assert result.passed, result.text


def test_the_committed_answer_comparison_is_current_and_like_for_like(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(Path(__file__).resolve().parents[2])

    assert check_answer_report() == []


def fixture_with_unanswerable(tmp_path: Path) -> Path:
    out = tmp_path / "fixture.jsonl.gz"
    build_fixture(
        [question("q1", ("gold",)), question("u1", (), "info_not_found")],
        {"q1": candidates("q1"), "u1": candidates("u1")},
        {
            "q1": {"d1": 0.1, "d2": 0.2, "g": 0.9, "b1": 0.3},
            "u1": {"d1": 0.1, "d2": 0.2, "g": 0.3, "b1": 0.1},
        },
        out,
        reranker="some/reranker",
    )
    return out


def test_questions_without_gold_documents_take_no_part_in_the_retrieval_numbers(
    tmp_path: Path,
) -> None:
    _, cases = load_fixture(fixture_with_unanswerable(tmp_path))
    _, only_gold = load_fixture(fixture(tmp_path))

    assert len(cases) == 2
    assert measure(cases, PIPELINE) == measure(only_gold, PIPELINE)


def write_run(folder: Path, answers: dict[str, tuple[str, bool]]) -> Path:
    folder.mkdir(parents=True)
    (folder / "trace.jsonl").write_text(
        "\n".join(json.dumps({"question_id": q, "raw_answer": a}) for q, (a, _) in answers.items())
        + "\n",
        encoding="utf-8",
    )
    rows = [
        {
            "question_id": q,
            "question_type": "basic" if q.startswith("q") else "info_not_found",
            "answer_correct": ok,
            "completeness_pct": 100.0 if ok else 0.0,
        }
        for q, (_, ok) in answers.items()
    ]
    (folder / "results-claude-haiku-5-5.json").write_text(
        json.dumps({"questions": rows}), encoding="utf-8"
    )
    return folder


def test_the_abstention_report_pairs_scores_with_correctness_and_what_answers_say(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr("rag_engine.evaluation.gate.shipped_pipeline", lambda: PIPELINE)
    run = write_run(
        tmp_path / "run",
        {"q1": ("The gold value is 5 [1].", True), "u1": ("The documents do not give it.", True)},
    )

    report = abstention_report(fixture_with_unanswerable(tmp_path), [run])

    rows = {(r.group, r.status): (r.n, r.correct) for r in report.statuses}
    assert rows == {("answerable", "answered"): (1, 1), ("info_not_found", "not_found"): (1, 1)}
    types = {row.question_type: row.median for row in report.by_type}
    assert types["basic"] == pytest.approx(0.9)
    assert types["info_not_found"] == pytest.approx(0.3)


def test_the_committed_abstention_numbers_must_match_the_fixture_and_the_runs(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    from dataclasses import asdict

    monkeypatch.setattr("rag_engine.evaluation.gate.shipped_pipeline", lambda: PIPELINE)
    path = fixture_with_unanswerable(tmp_path)
    run = write_run(
        tmp_path / "run", {"q1": ("Five [1].", True), "u1": ("The documents do not say.", True)}
    )
    published = tmp_path / "abstention.json"
    published.write_text(json.dumps(asdict(abstention_report(path, [run]))), encoding="utf-8")

    assert check_abstention_report(path, published, [run]) == []

    write_run(tmp_path / "other", {"q1": ("Five [1].", False), "u1": ("It is X.", False)})
    (run / "trace.jsonl").write_text(
        (tmp_path / "other" / "trace.jsonl").read_text(encoding="utf-8"), encoding="utf-8"
    )
    assert "does not match" in check_abstention_report(path, published, [run])[0]


def test_missing_runs_are_reported_instead_of_crashing(tmp_path: Path) -> None:
    published = tmp_path / "x.json"
    published.write_text("{}", encoding="utf-8")

    problems = check_abstention_report(
        fixture_with_unanswerable(tmp_path), published, [tmp_path / "nowhere"]
    )

    assert "missing" in problems[0]


def test_nothing_is_checked_before_the_abstention_numbers_are_published(tmp_path: Path) -> None:
    assert (
        check_abstention_report(fixture_with_unanswerable(tmp_path), tmp_path / "none.json") == []
    )


def test_a_comparison_that_is_not_published_yet_is_not_checked(tmp_path: Path) -> None:
    assert check_answer_report([("a", tmp_path / "nowhere")], report=tmp_path / "none.json") == []
