"""Does the strength of the evidence tell a question the corpus answers from one it does not?

The reranker gives every retrieved passage a score. The simplest abstention rule declines to answer
when the best score is below a threshold. Whether that works depends on one thing: how different
the best scores of answerable and unanswerable questions are. This module measures it from stored
candidates and stored scores, with no model call.

The benchmark's unanswerable questions are written to be hard for exactly this rule: each is
related to a cluster of real documents, and differs from them in a detail the documents lack. A
threshold that separates them is not guaranteed to exist, and the numbers here say how well one
does.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rag_engine.evaluation.candidates import QuestionCandidates
from rag_engine.evaluation.variants import FusionSetting, fused_ranking, reranked_ranking

POOL = 30  # the chunks the service reranks, as in the answer comparison


def top_score(
    candidates: QuestionCandidates,
    scores: Mapping[str, float],
    fusion: FusionSetting | None = None,
    pool: int = POOL,
) -> float:
    """The best cross-encoder score among the reranked pool, which is what the service sees."""
    ranked = reranked_ranking(fused_ranking(candidates, fusion), scores, pool)
    return ranked[0].score


def auc(answerable: Sequence[float], unanswerable: Sequence[float]) -> float:
    """Probability that a random answerable question has a higher best score than a random
    unanswerable one (ties count half). 0.5 is no information, 1.0 is a perfect separation."""
    if not answerable or not unanswerable:
        raise ValueError("both groups need at least one question")
    wins = sum(1.0 if a > u else 0.5 if a == u else 0.0 for a in answerable for u in unanswerable)
    return wins / (len(answerable) * len(unanswerable))


def threshold_for(answerable: Sequence[float], false_abstain: float) -> float:
    """The threshold below which at most `false_abstain` of the answerable questions fall."""
    if not 0 <= false_abstain < 1:
        raise ValueError("false_abstain is a share between 0 and 1")
    ordered = sorted(answerable)
    allowed = int(false_abstain * len(ordered))
    # Abstain when the score is below the threshold: the smallest `allowed` scores may fall below.
    return ordered[allowed]


@dataclass(frozen=True, slots=True)
class OperatingPoint:
    """What a threshold does: the answerable questions it refuses, the unanswerable it catches."""

    threshold: float
    false_abstain: float
    caught: float
    refused_answerable: int
    caught_unanswerable: int


def operating_point(
    threshold: float, answerable: Sequence[float], unanswerable: Sequence[float]
) -> OperatingPoint:
    refused = sum(score < threshold for score in answerable)
    caught = sum(score < threshold for score in unanswerable)
    return OperatingPoint(
        threshold=threshold,
        false_abstain=refused / len(answerable),
        caught=caught / len(unanswerable),
        refused_answerable=refused,
        caught_unanswerable=caught,
    )


def operating_points(
    answerable: Sequence[float],
    unanswerable: Sequence[float],
    rates: Sequence[float] = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20),
) -> list[OperatingPoint]:
    """The thresholds that refuse the given shares of the answerable questions."""
    return [
        operating_point(threshold_for(answerable, rate), answerable, unanswerable) for rate in rates
    ]


@dataclass(frozen=True, slots=True)
class Evidence:
    """What the retriever found for one question, and how the answer to it was judged."""

    question_id: str
    question_type: str
    top_score: float
    correct: bool | None = None  # the judge's verdict on the answer, when there was one
    status: str | None = None  # what the answer says about itself (generation.coverage)


@dataclass(frozen=True, slots=True)
class TypeRow:
    question_type: str
    n: int
    low: float
    q1: float
    median: float
    q3: float
    high: float


@dataclass(frozen=True, slots=True)
class Protection:
    """What a threshold does to the answers of the questions the corpus does answer: the wrong
    answers it stops and the right ones it gives up."""

    threshold: float
    wrong_avoided: int
    wrong_total: int
    right_lost: int
    right_total: int


@dataclass(frozen=True, slots=True)
class StatusRow:
    """How many answers of a group said what they said about themselves, and how many were right."""

    group: str
    status: str
    n: int
    correct: int


NOT_ANSWERABLE = ("info_not_found", "high_level")  # types that have no gold documents


@dataclass(slots=True)
class AbstentionReport:
    by_type: list[TypeRow]
    unanswerable_type: str
    unanswerable_total: int
    auc_unanswerable: float | None
    points: list[OperatingPoint]
    high_level_refused: dict[float, int]  # threshold -> how many high-level questions it refuses
    high_level_total: int
    auc_correct: float | None
    protection: list[Protection]
    statuses: list[StatusRow]


def quantile(ordered: Sequence[float], share: float) -> float:
    """Linear interpolation between neighbours, as numpy's default."""
    position = share * (len(ordered) - 1)
    below = int(position)
    above = min(below + 1, len(ordered) - 1)
    return ordered[below] + (ordered[above] - ordered[below]) * (position - below)


def build_report(
    evidence: Sequence[Evidence],
    *,
    unanswerable_type: str = "info_not_found",
    rates: Sequence[float] = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20),
) -> AbstentionReport:
    by_type: dict[str, list[float]] = {}
    for item in evidence:
        by_type.setdefault(item.question_type, []).append(item.top_score)
    rows = []
    for name, scores in sorted(by_type.items()):
        ordered = sorted(scores)
        rows.append(
            TypeRow(
                name,
                len(ordered),
                ordered[0],
                quantile(ordered, 0.25),
                quantile(ordered, 0.5),
                quantile(ordered, 0.75),
                ordered[-1],
            )
        )
    # The questions the corpus answers: the judged ones of the types that have gold documents.
    answerable = [
        e for e in evidence if e.correct is not None and e.question_type not in NOT_ANSWERABLE
    ]
    answerable_scores = [e.top_score for e in answerable]
    unanswerable = by_type.get(unanswerable_type, [])
    thresholds = (
        [threshold_for(answerable_scores, rate) for rate in rates] if answerable_scores else []
    )
    points = (
        [operating_point(t, answerable_scores, unanswerable) for t in thresholds]
        if unanswerable
        else []
    )
    high = by_type.get("high_level", [])
    right = [e.top_score for e in answerable if e.correct]
    wrong = [e.top_score for e in answerable if not e.correct]
    return AbstentionReport(
        by_type=rows,
        unanswerable_type=unanswerable_type,
        unanswerable_total=len(unanswerable),
        auc_unanswerable=auc(answerable_scores, unanswerable) if points else None,
        points=points,
        high_level_refused={t: sum(score < t for score in high) for t in thresholds},
        high_level_total=len(high),
        auc_correct=auc(right, wrong) if right and wrong else None,
        protection=[
            Protection(
                t,
                wrong_avoided=sum(score < t for score in wrong),
                wrong_total=len(wrong),
                right_lost=sum(score < t for score in right),
                right_total=len(right),
            )
            for t in thresholds
        ]
        if right and wrong
        else [],
        statuses=_statuses(evidence, answerable),
    )


def _statuses(evidence: Sequence[Evidence], answerable: Sequence[Evidence]) -> list[StatusRow]:
    groups = {
        "answerable": list(answerable),
        **{
            kind: [e for e in evidence if e.question_type == kind and e.status is not None]
            for kind in NOT_ANSWERABLE
        },
    }
    rows = []
    for name, members in groups.items():
        for status in ("answered", "partial", "not_found"):
            chosen = [e for e in members if e.status == status]
            if chosen:
                rows.append(
                    StatusRow(name, status, len(chosen), sum(bool(e.correct) for e in chosen))
                )
    return rows


def render(report: AbstentionReport) -> str:
    lines = [
        "# Abstention by evidence strength",
        "",
        "The best cross-encoder score among the 30 reranked passages of each question, by question "
        "type. A threshold on this score is the simplest rule for declining to answer.",
        "",
        "| Question type | Questions | Lowest | 25% | Median | 75% | Highest |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in report.by_type:
        lines.append(
            f"| {row.question_type} | {row.n} | {row.low:.3f} | {row.q1:.3f} | {row.median:.3f} "
            f"| {row.q3:.3f} | {row.high:.3f} |"
        )
    if report.points:
        lines += [
            "",
            f"## Telling `{report.unanswerable_type}` questions from answerable ones",
            "",
            f"Probability that a random answerable question has a higher best score than a random "
            f"`{report.unanswerable_type}` one: **{report.auc_unanswerable:.2f}** (0.50 means the "
            "score carries no information, 1.00 a perfect separation).",
            "",
            "Each row is a threshold, chosen so that it refuses the stated share of the answerable "
            "questions:",
            "",
            f"| Answerable refused | Threshold | `{report.unanswerable_type}` caught | "
            "High-level refused |",
            "|---|---|---|---|",
        ]
        for point in report.points:
            caught = f"{point.caught_unanswerable} of {report.unanswerable_total}"
            lines.append(
                f"| {100 * point.false_abstain:.0f}% ({point.refused_answerable}) "
                f"| {point.threshold:.4f} | {caught} "
                f"| {report.high_level_refused[point.threshold]} of {report.high_level_total} |"
            )
    if report.protection:
        lines += [
            "",
            "## Stopping wrong answers",
            "",
            "Among the answerable questions, whether the answer was judged correct, against the "
            f"best score: probability that a correct answer has the higher score "
            f"**{report.auc_correct:.2f}**.",
            "",
            "| Threshold | Wrong answers avoided | Right answers given up |",
            "|---|---|---|",
        ]
        for p in report.protection:
            lines.append(
                f"| {p.threshold:.4f} | {p.wrong_avoided} of {p.wrong_total} "
                f"| {p.right_lost} of {p.right_total} |"
            )
    if report.statuses:
        lines += [
            "",
            "## What the answer says about itself",
            "",
            "`answered`: nothing is said to be missing. `partial`: part is stated and the rest is "
            "said to be missing. `not_found`: only that nothing was found. Read from the answer "
            "text, with no model call.",
            "",
            "| Questions | Status | Answers | Judged correct |",
            "|---|---|---|---|",
        ]
        for entry in report.statuses:
            lines.append(
                f"| {entry.group} | {entry.status} | {entry.n} | {entry.correct} "
                f"({100 * entry.correct / entry.n:.0f}%) |"
            )
    return "\n".join(lines) + "\n"
