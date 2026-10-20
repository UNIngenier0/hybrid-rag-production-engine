import json
from pathlib import Path

import pytest

from rag_engine.evaluation.compare import Arm, Outcome, compare, load_outcomes
from rag_engine.evaluation.runs import answering_cost


def outcome(qtype: str, correct: bool, completeness: float, recall: float | None = 1.0) -> Outcome:
    return Outcome(qtype, correct, completeness, recall, 0 if recall is not None else None)


def arm(name: str, rows: dict[str, Outcome], cost: float | None = None) -> Arm:
    return Arm(name, rows, cost)


def test_the_combined_score_counts_completeness_only_for_correct_answers() -> None:
    assert outcome("basic", True, 0.8).combined == 0.8
    assert outcome("basic", False, 0.8).combined == 0.0


def test_results_are_read_from_the_evaluators_file(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    path.write_text(
        json.dumps(
            {
                "questions": [
                    {
                        "question_id": "q1",
                        "question_type": "basic",
                        "answer_correct": True,
                        "completeness_pct": 75.0,
                        "document_recall_pct": 50.0,
                        "invalid_extra_docs": 2,
                    },
                    {
                        "question_id": "q2",
                        "question_type": "info_not_found",
                        "answer_correct": False,
                        "completeness_pct": 0.0,
                        "document_recall_pct": None,
                        "invalid_extra_docs": None,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    outcomes = load_outcomes(path)

    assert outcomes["q1"] == Outcome("basic", True, 0.75, 0.5, 2)
    assert outcomes["q2"].recall is None


def test_systems_are_compared_on_the_questions_they_all_answered() -> None:
    rag = arm("rag", {f"q{i}": outcome("basic", True, 1.0) for i in range(6)})
    grep = arm("grep", {f"q{i}": outcome("basic", i < 3, 1.0) for i in range(4)})  # answered q0-q3

    result = compare([rag, grep], baseline="rag")

    assert result.data["questions"] == 4
    assert result.data["arms"]["rag"]["combined"] == 1.0
    assert result.data["arms"]["grep"]["combined"] == 0.75


def test_a_clear_difference_is_marked_significant() -> None:
    questions = [f"q{i}" for i in range(40)]
    rag = arm("rag", {q: outcome("basic", True, 1.0) for q in questions})
    grep = arm("grep", {q: outcome("basic", False, 1.0) for q in questions})

    result = compare([rag, grep], baseline="rag")

    assert "-100.0 (-100.0, -100.0)*" in result.markdown


def test_the_tables_name_every_system_and_question_type_and_show_the_cost() -> None:
    rows = {"q1": outcome("basic", True, 1.0), "q2": outcome("semantic", False, 0.5)}
    result = compare(
        [arm("rag", rows, cost=0.08), arm("agent", rows, cost=2.0), arm("grep", rows)], "rag"
    )

    for text in ("| rag |", "| agent |", "| grep |", "basic (1)", "semantic (1)"):
        assert text in result.markdown
    assert "$0.08 ($0.0400 per question)" in result.markdown
    assert "n/a" in result.markdown  # the system without a known cost


def test_questions_without_gold_documents_do_not_count_toward_recall() -> None:
    rows = {
        "q1": outcome("basic", True, 1.0, recall=0.5),
        "q2": outcome("info_not_found", True, 1.0, recall=None),
    }

    result = compare([arm("rag", rows)], "rag")

    assert result.data["arms"]["rag"]["recall"] == 0.5


def test_the_cost_of_a_metered_system_comes_from_its_usage_report(tmp_path: Path) -> None:
    (tmp_path / "cost-agent.json").write_text(json.dumps({"spent_usd": 1.25}), encoding="utf-8")

    assert answering_cost(tmp_path) == 1.25


def test_the_cost_of_the_rag_comes_from_the_tokens_in_its_trace(tmp_path: Path) -> None:
    rows = [
        {
            "question_id": "q1",
            "model": "claude-haiku-5-5",
            "input_tokens": 1000000,
            "output_tokens": 0,
        },
        {"question_id": "q2", "error": "a failed question has no tokens"},
    ]
    (tmp_path / "trace.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    assert answering_cost(tmp_path) == pytest.approx(0.10)


def test_a_folder_with_no_cost_information_has_no_cost(tmp_path: Path) -> None:
    assert answering_cost(tmp_path) is None
