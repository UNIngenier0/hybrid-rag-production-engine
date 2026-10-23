import hashlib
import json
from pathlib import Path

import pytest

from rag_engine.evaluation.fullbench import (
    SubmissionError,
    build_report,
    load_rows,
    render,
    submission_rows,
    write_submission,
)

JUDGE = "claude-haiku-5-5"


def result(question_id: str, question_type: str, correct: bool, completeness: float, **extra):  # type: ignore[no-untyped-def]
    return {
        "question_id": question_id,
        "question_type": question_type,
        "answer_correct": correct,
        "completeness_pct": completeness,
        **extra,
    }


def make_run(folder: Path, rows: list[dict], answers: dict[str, str]) -> Path:  # type: ignore[type-arg]
    folder.mkdir(parents=True)
    (folder / f"results-{JUDGE}.json").write_text(json.dumps({"questions": rows}), encoding="utf-8")
    (folder / "answers.jsonl").write_text(
        "".join(
            json.dumps({"question_id": q, "answer": a, "document_ids": ["d1"]}) + "\n"
            for q, a in answers.items()
        ),
        encoding="utf-8",
    )
    (folder / "run.json").write_text(
        json.dumps({"model": "claude-haiku-5-5", "mode": "hybrid+rerank", "passages": 5}),
        encoding="utf-8",
    )
    (folder / "trace.jsonl").write_text(
        json.dumps(
            {
                "question_id": "q1",
                "input_tokens": 1000,
                "output_tokens": 100,
                "model": "claude-haiku-5-5",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return folder


@pytest.fixture
def runs(tmp_path: Path) -> list[Path]:
    gold = make_run(
        tmp_path / "gold",
        [
            result("q1", "basic", True, 100.0, document_recall_pct=100.0, invalid_extra_docs=1),
            result("q2", "basic", False, 50.0, document_recall_pct=0.0, invalid_extra_docs=3),
        ],
        {"q1": "Ten [1].", "q2": "Five [1]."},
    )
    none = make_run(
        tmp_path / "none",
        [
            result(
                "q3", "info_not_found", True, 100.0, document_recall_pct=0.0, invalid_extra_docs=0
            )
        ],
        {"q3": "The documents do not say."},
    )
    return [gold, none]


def test_the_score_counts_a_wrong_answer_as_zero_whatever_its_completeness(runs) -> None:  # type: ignore[no-untyped-def]
    report = build_report(runs, JUDGE, ["q1", "q2", "q3"])

    assert report["overall"]["questions"] == 3
    assert report["overall"]["combined"] == pytest.approx(66.67)  # (100 + 0 + 100) / 3
    assert report["overall"]["correct_pct"] == pytest.approx(66.67)
    assert report["by_type"]["basic"]["combined"] == 50.0


def test_the_mean_of_the_types_differs_from_the_mean_of_the_questions(runs) -> None:  # type: ignore[no-untyped-def]
    report = build_report(runs, JUDGE, ["q1", "q2", "q3"])

    assert report["mean_of_type_combined"] == 75.0  # (50 + 100) / 2


def test_document_metrics_only_count_questions_that_have_gold_documents(runs) -> None:  # type: ignore[no-untyped-def]
    report = build_report(runs, JUDGE, ["q1", "q2", "q3"])

    assert report["overall"]["document_recall_pct"] == 50.0  # q1 and q2, not the unanswerable
    assert report["by_type"]["info_not_found"]["document_recall_pct"] is None


def test_a_question_nobody_answered_stops_the_report(runs) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(SubmissionError, match="1 benchmark questions are not answered"):
        build_report(runs, JUDGE, ["q1", "q2", "q3", "q4"])


def test_an_answer_to_a_question_outside_the_benchmark_stops_the_report(runs) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(SubmissionError, match="not in the benchmark"):
        build_report(runs, JUDGE, ["q1", "q2"])


def test_a_question_judged_in_two_runs_would_count_twice_so_it_is_refused(tmp_path: Path) -> None:
    one = make_run(tmp_path / "one", [result("q1", "basic", True, 100.0)], {"q1": "a"})
    two = make_run(tmp_path / "two", [result("q1", "basic", True, 100.0)], {"q1": "a"})

    with pytest.raises(SubmissionError, match="more than one run"):
        load_rows([one, two], JUDGE)
    with pytest.raises(SubmissionError, match="more than one run"):
        submission_rows([one, two])


def test_the_submission_has_every_answer_once_in_question_order(runs) -> None:  # type: ignore[no-untyped-def]
    rows = submission_rows(list(reversed(runs)))

    assert [row["question_id"] for row in rows] == ["q1", "q2", "q3"]
    assert rows[0] == {"question_id": "q1", "answer": "Ten [1].", "document_ids": ["d1"]}


def test_an_empty_answer_cannot_be_submitted(tmp_path: Path) -> None:
    run = make_run(tmp_path / "r", [result("q1", "basic", True, 100.0)], {"q1": "  "})

    with pytest.raises(SubmissionError, match="empty answer"):
        submission_rows([run])


def test_the_checksum_is_that_of_the_file_that_was_written(runs, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    checksum = write_submission(submission_rows(runs), tmp_path / "out")

    written = (tmp_path / "out" / "answers.jsonl").read_bytes()
    assert checksum == hashlib.sha256(written).hexdigest()
    assert written.count(b"\n") == 3


def test_the_markdown_gives_the_headline_and_a_row_per_type(runs) -> None:  # type: ignore[no-untyped-def]
    text = render(build_report(runs, JUDGE, ["q1", "q2", "q3"]))

    assert "All 3 questions" in text
    assert "| basic | 2 |" in text and "| info_not_found | 1 |" in text
    assert "**66.67**" in text and "**75.0**" in text
