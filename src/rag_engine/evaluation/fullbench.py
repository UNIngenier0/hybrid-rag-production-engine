"""The whole benchmark in one place: every question answered, scored and ready to submit.

The 500 questions were answered in two runs, the 470 that have gold documents and the 30 that have
none (information that is not in the corpus, and broad questions), because the second kind needs
no document metrics. This puts them together: one score and one table per question type, the
answers in the file format the benchmark's evaluator reads, and a check that no question is
missing or answered twice.
"""

import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from rag_engine.evaluation.abstention import NOT_ANSWERABLE
from rag_engine.evaluation.runs import answering_cost

DEFAULT_RUNS = (
    Path("eval/runs/rag-haiku-all"),
    Path("eval/runs/rag-haiku-unanswerable"),
)


class SubmissionError(Exception):
    """The runs do not add up to exactly the benchmark's questions."""


def _results(run: Path, judge: str) -> list[dict[str, Any]]:
    path = run / f"results-{judge}.json"
    rows: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))["questions"]
    return rows


def load_rows(runs: Sequence[Path], judge: str) -> list[dict[str, Any]]:
    """The judged result of every question, in question order. A question judged in two runs is an
    error: it would count twice in the score."""
    rows: dict[str, dict[str, Any]] = {}
    for run in runs:
        for row in _results(run, judge):
            if row["question_id"] in rows:
                raise SubmissionError(f"{row['question_id']} is judged in more than one run")
            rows[row["question_id"]] = row
    return [rows[question_id] for question_id in sorted(rows)]


def _mean(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 2) if values else 0.0


def _summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    combined = [row["completeness_pct"] if row["answer_correct"] else 0.0 for row in rows]
    with_gold = [row for row in rows if row["question_type"] not in NOT_ANSWERABLE]
    recall = [
        row["document_recall_pct"]
        for row in with_gold
        if row.get("document_recall_pct") is not None
    ]
    extra = [
        row["invalid_extra_docs"] for row in with_gold if row.get("invalid_extra_docs") is not None
    ]
    return {
        "questions": len(rows),
        "correct_pct": _mean([100.0 * bool(row["answer_correct"]) for row in rows]),
        "completeness_pct": _mean([row["completeness_pct"] for row in rows]),
        "combined": _mean(combined),
        # Document metrics only mean something where a question has gold documents.
        "document_recall_pct": _mean(recall) if recall else None,
        "invalid_extra_docs": round(sum(extra) / len(extra), 2) if extra else None,
    }


def build_report(runs: Sequence[Path], judge: str, expected_ids: Iterable[str]) -> dict[str, Any]:
    rows = load_rows(runs, judge)
    expected = set(expected_ids)
    found = {row["question_id"] for row in rows}
    if found != expected:
        raise SubmissionError(
            f"{len(expected - found)} benchmark questions are not answered and "
            f"{len(found - expected)} answered ones are not in the benchmark"
        )
    by_type = {
        question_type: _summary([row for row in rows if row["question_type"] == question_type])
        for question_type in sorted({row["question_type"] for row in rows})
    }
    manifests = [json.loads((run / "run.json").read_text(encoding="utf-8")) for run in runs]
    answers = [answering_cost(run) for run in runs]
    judging = [
        json.loads((run / f"cost-judge-{judge}.json").read_text(encoding="utf-8"))["spent_usd"]
        for run in runs
        if (run / f"cost-judge-{judge}.json").exists()
    ]
    return {
        "judge": judge,
        "system": {
            key: manifests[0][key] for key in ("model", "mode", "passages") if key in manifests[0]
        },
        "runs": [str(run) for run in runs],
        "overall": _summary(rows),
        # The benchmark's leaderboard averages its ten categories, not its questions.
        "mean_of_type_combined": _mean([summary["combined"] for summary in by_type.values()]),
        "by_type": by_type,
        "cost_usd": {
            "answers": round(sum(c for c in answers if c is not None), 4),
            "judging": round(sum(judging), 4),
        },
    }


def render(report: dict[str, Any]) -> str:
    overall, system = report["overall"], report["system"]
    lines = [
        "# The whole benchmark",
        "",
        f"All {overall['questions']} questions, answered by hybrid retrieval with reranking and "
        f"{system.get('model')} ({system.get('passages')} passages each, mode "
        f"`{system.get('mode')}`) and scored by the benchmark's own evaluator with "
        f"{report['judge']} as the judge, against the original gold answers. The judge is not "
        "the GPT model of the public leaderboard, so these numbers are not comparable with it.",
        "",
        "| | Questions | Correct | Completeness | Combined | Document recall "
        "| Invalid extra docs |",
        "|---|---|---|---|---|---|---|",
    ]

    def row(name: str, summary: dict[str, Any]) -> str:
        recall = summary["document_recall_pct"]
        extra = summary["invalid_extra_docs"]
        return (
            f"| {name} | {summary['questions']} | {summary['correct_pct']}% "
            f"| {summary['completeness_pct']}% | **{summary['combined']}** "
            f"| {'-' if recall is None else f'{recall}%'} | {'-' if extra is None else extra} |"
        )

    lines.append(row("**All questions**", overall))
    for question_type, summary in report["by_type"].items():
        lines.append(row(question_type, summary))
    cost = report["cost_usd"]
    lines += [
        "",
        f"The benchmark's headline is the combined score over the questions, "
        f"**{overall['combined']}**. "
        f"Averaging the {len(report['by_type'])} question types instead, as the leaderboard "
        f"describes, gives **{report['mean_of_type_combined']}**. Answering all of it cost "
        f"${cost['answers']:.2f} and judging it ${cost['judging']:.2f}.",
    ]
    return "\n".join(lines) + "\n"


def submission_rows(runs: Sequence[Path]) -> list[dict[str, Any]]:
    """Every answer in the benchmark's file format, one per question, in question order."""
    rows: dict[str, dict[str, Any]] = {}
    for run in runs:
        for line in (run / "answers.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["question_id"] in rows:
                raise SubmissionError(f"{row['question_id']} is answered in more than one run")
            if not str(row.get("answer", "")).strip():
                raise SubmissionError(f"{row['question_id']} has an empty answer")
            rows[row["question_id"]] = {
                "question_id": row["question_id"],
                "answer": row["answer"],
                "document_ids": list(row.get("document_ids", [])),
            }
    return [rows[question_id] for question_id in sorted(rows)]


def write_submission(rows: Sequence[dict[str, Any]], out_dir: Path) -> str:
    """Write answers.jsonl and return its SHA-256, which the manifest records."""
    out_dir.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    (out_dir / "answers.jsonl").write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
