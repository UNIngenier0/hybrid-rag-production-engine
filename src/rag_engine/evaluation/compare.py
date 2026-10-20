"""Compare systems on the answers they gave, from the benchmark evaluator's results files.

Each system is scored by the benchmark's own evaluator against the same gold answers, so the
numbers can be compared question by question. The headline is the benchmark's combined score:
the average over questions of completeness, counted as zero when the answer is not correct.
"""

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_engine.evaluation.metrics import paired_difference


@dataclass(frozen=True, slots=True)
class Outcome:
    question_type: str
    correct: bool
    completeness: float  # 0 to 1
    recall: float | None  # 0 to 1, None when the question has no gold documents
    invalid_extra: int | None

    @property
    def combined(self) -> float:
        return self.completeness if self.correct else 0.0


@dataclass(frozen=True, slots=True)
class Arm:
    name: str
    outcomes: Mapping[str, Outcome]
    cost_usd: float | None = None  # what producing the answers cost, judge not included


@dataclass(frozen=True, slots=True)
class Comparison:
    markdown: str
    data: dict[str, Any]


def load_outcomes(results_file: Path) -> dict[str, Outcome]:
    results = json.loads(results_file.read_text(encoding="utf-8"))
    outcomes = {}
    for row in results["questions"]:
        recall = row.get("document_recall_pct")
        outcomes[row["question_id"]] = Outcome(
            question_type=row["question_type"],
            correct=bool(row["answer_correct"]),
            completeness=float(row["completeness_pct"]) / 100,
            recall=None if recall is None else float(recall) / 100,
            invalid_extra=row.get("invalid_extra_docs"),
        )
    return outcomes


def compare(arms: Sequence[Arm], baseline: str) -> Comparison:
    """Tables over the questions every system answered, so each comparison is paired."""
    common = sorted(set.intersection(*(set(arm.outcomes) for arm in arms)))
    base = next(arm for arm in arms if arm.name == baseline)
    lines = [
        "# Answer comparison",
        "",
        f"{len(common)} questions answered by every system, scored by the benchmark's evaluator "
        "against the original gold answers and documents. *Combined* is the benchmark's score: "
        "the mean over questions of completeness, counted as zero when the answer is not "
        "correct. The judge is a Claude model, not the GPT model of the public leaderboard, so "
        "these numbers are not comparable with it. `*` marks a paired difference whose 95% "
        "interval excludes zero.",
        "",
        "| System | Correct | Completeness | Combined | Doc recall | Invalid extra docs "
        f"| Combined vs {baseline} | Cost of the answers |",
        "|---|---|---|---|---|---|---|---|",
    ]
    data: dict[str, Any] = {"questions": len(common), "arms": {}}
    for arm in arms:
        rows = [arm.outcomes[q] for q in common]
        with_gold = [row for row in rows if row.recall is not None]
        stats = {
            "correct": _mean([float(r.correct) for r in rows]),
            "completeness": _mean([r.completeness for r in rows]),
            "combined": _mean([r.combined for r in rows]),
            "recall": _mean([r.recall for r in with_gold if r.recall is not None]),
            "invalid_extra": _mean([float(r.invalid_extra or 0) for r in with_gold]),
        }
        if arm.name == baseline:
            delta = ""
        else:
            diff = paired_difference(
                [r.combined for r in rows], [base.outcomes[q].combined for q in common]
            )
            star = "*" if diff.significant else ""
            delta = f"{100 * diff.mean:+.1f} ({100 * diff.low:+.1f}, {100 * diff.high:+.1f}){star}"
            stats["combined_vs_baseline"] = diff.mean
        cost = (
            "n/a"
            if arm.cost_usd is None
            else f"${arm.cost_usd:.2f} (${arm.cost_usd / len(common):.4f} per question)"
        )
        lines.append(
            f"| {arm.name} | {100 * stats['correct']:.1f}% | {100 * stats['completeness']:.1f}% "
            f"| **{100 * stats['combined']:.1f}** | {100 * stats['recall']:.1f}% "
            f"| {stats['invalid_extra']:.2f} | {delta} | {cost} |"
        )
        data["arms"][arm.name] = stats | {"cost_usd": arm.cost_usd}

    types = sorted({base.outcomes[q].question_type for q in common})
    counts = {t: sum(1 for q in common if base.outcomes[q].question_type == t) for t in types}
    lines += [
        "",
        "## Combined score by question type",
        "",
        "| System | " + " | ".join(f"{t} ({counts[t]})" for t in types) + " |",
        "|---|" + "---|" * len(types),
    ]
    for arm in arms:
        by_type: dict[str, list[float]] = defaultdict(list)
        for q in common:
            by_type[arm.outcomes[q].question_type].append(arm.outcomes[q].combined)
        lines.append(
            f"| {arm.name} | " + " | ".join(f"{100 * _mean(by_type[t]):.1f}" for t in types) + " |"
        )
    return Comparison("\n".join(lines) + "\n", data)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
