import json
from pathlib import Path
from typing import Any

from rag_engine.evaluation.audit import audit_runs, write_manifest
from rag_engine.evaluation.compare import Arm, Outcome, compare

JUDGE = "judge-model"
QUESTIONS = ["q1", "q2", "q3"]


def make_run(
    root: Path,
    name: str,
    *,
    system: str = "rag",
    model: str = "gen-model",
    passages: int | None = 5,
    documents: str = "cited",
    questions: list[str] | None = None,
    answered: list[str] | None = None,
    scored: list[str] | None = None,
    judge_model: str = JUDGE,
    used_judge: str = JUDGE,
    corrected: int = 0,
    usage_model: str | None = None,
    strip: bool = False,
) -> Path:
    folder = root / name
    folder.mkdir(parents=True)
    questions = QUESTIONS if questions is None else questions
    (folder / "questions.txt").write_text("\n".join(questions) + "\n", encoding="utf-8")
    ids = questions if answered is None else answered
    (folder / "answers.jsonl").write_text(
        "".join(json.dumps({"question_id": q, "answer": "a"}) + "\n" for q in ids),
        encoding="utf-8",
    )
    write_manifest(
        folder,
        {"system": system, "model": model, "passages": passages, "documents": documents},
    )
    results: dict[str, Any] = {
        "aggregate_stats": {"num_corrected_questions": corrected, "skipped_rows": 0},
        "questions": [{"question_id": q} for q in (questions if scored is None else scored)],
    }
    (folder / f"results-{JUDGE}.json").write_text(json.dumps(results), encoding="utf-8")
    (folder / f"judge-{JUDGE}.json").write_text(
        json.dumps({"judge": JUDGE, "model": judge_model, "strip_citations": strip}),
        encoding="utf-8",
    )
    (folder / f"cost-judge-{JUDGE}.json").write_text(
        json.dumps({"by_model": {used_judge: {}}}), encoding="utf-8"
    )
    (folder / "trace.jsonl").write_text(
        json.dumps({"question_id": "q1", "model": usage_model or model, "input_tokens": 10}) + "\n",
        encoding="utf-8",
    )
    return folder


def failed(tmp_path: Path, **second: Any) -> set[str]:
    arms = [
        ("ours", make_run(tmp_path, "ours")),
        ("rival", make_run(tmp_path, "rival", system="grep-oneshot", **second)),
    ]
    return {check.name for check in audit_runs(arms, JUDGE).failures}


def test_comparable_runs_pass_every_check(tmp_path: Path) -> None:
    arms = [
        ("ours", make_run(tmp_path, "ours")),
        ("rival", make_run(tmp_path, "rival", system="grep-oneshot")),
    ]

    audit = audit_runs(arms, JUDGE)

    assert audit.passed, audit.failures
    assert len(audit.checks) == 9


def test_different_questions_are_caught(tmp_path: Path) -> None:
    assert "same questions" in failed(tmp_path, questions=["q1", "q2", "q4"])


def test_a_missing_answer_is_caught(tmp_path: Path) -> None:
    assert "every question answered" in failed(tmp_path, answered=["q1", "q2"])


def test_an_unscored_answer_is_caught(tmp_path: Path) -> None:
    assert "every answer scored" in failed(tmp_path, scored=["q1"])


def test_a_corrected_gold_set_is_caught(tmp_path: Path) -> None:
    assert "same gold answers for everyone" in failed(tmp_path, corrected=3)


def test_another_answering_model_is_caught(tmp_path: Path) -> None:
    assert "same answering model" in failed(tmp_path, model="bigger-model")


def test_usage_from_another_model_than_declared_is_caught(tmp_path: Path) -> None:
    assert "same answering model" in failed(tmp_path, usage_model="bigger-model")


def test_another_judge_is_caught(tmp_path: Path) -> None:
    assert "same judge" in failed(
        tmp_path, judge_model="stronger-judge", used_judge="stronger-judge"
    )


def test_a_judge_that_is_not_the_one_declared_is_caught(tmp_path: Path) -> None:
    assert "same judge" in failed(tmp_path, used_judge="other-judge")


def test_a_different_evidence_budget_is_caught_for_single_pass_systems(tmp_path: Path) -> None:
    assert "same evidence budget for single-pass systems" in failed(tmp_path, passages=10)


def test_a_different_document_rule_is_caught(tmp_path: Path) -> None:
    assert "same document-reporting rule" in failed(tmp_path, documents="retrieved")


def test_a_run_without_a_manifest_cannot_vouch_for_itself(tmp_path: Path) -> None:
    arms = [("ours", make_run(tmp_path, "ours")), ("rival", make_run(tmp_path, "rival"))]
    (tmp_path / "rival" / "run.json").unlink()

    names = {check.name for check in audit_runs(arms, JUDGE).failures}

    assert "same answering model" in names


def test_the_iterative_agent_is_exempt_from_the_evidence_budget_but_noted(tmp_path: Path) -> None:
    arms = [
        ("ours", make_run(tmp_path, "ours")),
        (
            "agent",
            make_run(
                tmp_path,
                "agent",
                system="grep-agent",
                passages=None,
                documents="reported",
                strip=True,
            ),
        ),
    ]

    audit = audit_runs(arms, JUDGE)

    assert audit.passed, audit.failures
    assert any("decides for itself" in note for note in audit.notes)
    assert any("removes citations" in note for note in audit.notes)


def test_the_comparison_carries_the_checks(tmp_path: Path) -> None:
    arms = [("ours", make_run(tmp_path, "ours")), ("rival", make_run(tmp_path, "rival"))]
    rows = {q: Outcome("basic", True, 1.0, 1.0, 0) for q in QUESTIONS}

    result = compare([Arm("ours", rows), Arm("rival", rows)], "ours", audit_runs(arms, JUDGE))

    assert "## Fairness checks" in result.markdown
    assert result.data["audit"]["passed"] is True
    assert len(result.data["audit"]["checks"]) == 9


def test_a_metering_report_without_calls_is_not_a_cost_of_zero(tmp_path: Path) -> None:
    arms = [
        ("ours", make_run(tmp_path, "ours")),
        ("rival", make_run(tmp_path, "rival", system="grep-oneshot")),
    ]
    (tmp_path / "rival" / "cost-grep.json").write_text(
        json.dumps({"spent_usd": 0, "calls": 0, "by_model": {}}), encoding="utf-8"
    )

    audit = audit_runs(arms, JUDGE)

    assert "cost measured for every system" in {check.name for check in audit.failures}
    assert "shows no calls" in next(c.detail for c in audit.failures)


def test_a_run_with_neither_a_usage_report_nor_token_counts_has_no_cost(tmp_path: Path) -> None:
    arms = [("ours", make_run(tmp_path, "ours"))]
    (tmp_path / "ours" / "trace.jsonl").write_text(
        json.dumps({"question_id": "q1", "model": "gen-model"}) + "\n", encoding="utf-8"
    )

    assert "cost measured for every system" in {c.name for c in audit_runs(arms, JUDGE).failures}
