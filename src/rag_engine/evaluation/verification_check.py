"""Does the citation check catch wrong citations? Plant some and see.

Real answers are verified first. Then two kinds of damage are done to statements that were
verified, and the verifier is asked again:

- **A wrong citation.** The statement is moved to cite another passage that was shown for the same
  question, so it now cites something that does not say it (unless that passage happens to).
- **A wrong value.** A number in the statement is replaced by one that appears in none of the
  passages it cites.

Each kind is reported with how many the judge alone caught and how many the whole check caught, so
the value of the plain-text checks next to the judge is visible. A statement that survives is a
miss. Nothing here says the check is right about the statements it did not touch: that needs
labels, and the report lists a sample for reading by hand.
"""

import asyncio
import random
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from rag_engine.evaluation.questions import Question
from rag_engine.generation.prompt import ContextBlock
from rag_engine.generation.query import QueryService
from rag_engine.generation.verify import (
    NUMBER,
    CitationVerifier,
    Claim,
    missing_values,
    values_of,
    visible_text,
)


@dataclass(frozen=True, slots=True)
class Probe:
    """A statement that was verified, with the passages of its question."""

    question_id: str
    claim: Claim
    blocks: tuple[ContextBlock, ...]


@dataclass(slots=True)
class Outcome:
    kind: str
    planted: int = 0
    flagged: int = 0  # not `verified` after the damage: the whole check caught it
    judge_flagged: int = 0  # the judge alone said partial or no
    ambiguous: int = 0  # the damage left a statement the passages may still support

    @property
    def rate(self) -> float:
        return self.flagged / self.planted if self.planted else 0.0

    @property
    def judge_rate(self) -> float:
        return self.judge_flagged / self.planted if self.planted else 0.0


@dataclass(slots=True)
class CheckReport:
    answers: int = 0
    answers_all_verified: int = 0
    statuses: Counter[str] = field(default_factory=Counter)
    outcomes: list[Outcome] = field(default_factory=list)
    sample: list[dict[str, object]] = field(default_factory=list)
    flagged: list[dict[str, object]] = field(default_factory=list)  # statements to read by hand
    cost_usd: float = 0.0
    failed: int = 0
    judge_errors: Counter[str] = field(default_factory=Counter)


async def check_verification(
    questions: Sequence[Question],
    service: QueryService,
    verifier: CitationVerifier,
    *,
    seed: int = 0,
    concurrency: int = 4,
    sample_size: int = 30,
) -> CheckReport:
    report = CheckReport()
    semaphore = asyncio.Semaphore(concurrency)
    probes: list[Probe] = []

    async def run(question: Question) -> None:
        async with semaphore:
            try:
                result = await service.query(question.question)
            except Exception:
                report.failed += 1
                return
        report.answers += 1
        if result.verification.error:
            report.judge_errors[result.verification.error] += 1
        report.cost_usd += result.cost_usd or 0.0
        report.statuses.update(claim.status for claim in result.verification.claims)
        report.answers_all_verified += result.verification.all_verified
        for claim in result.verification.claims:
            if claim.status == "verified":
                probes.append(Probe(question.question_id, claim, result.answer.blocks))
            elif claim.status in ("partially_supported", "unsupported"):
                report.flagged.append(
                    _review_row(question.question_id, claim, result.answer.blocks)
                )

    await asyncio.gather(*(run(question) for question in questions))
    probes.sort(key=lambda probe: (probe.question_id, probe.claim.text))
    rng = random.Random(seed)
    report.outcomes = [
        await _plant(probes, verifier, move_citation, "wrong citation", rng, concurrency),
        await _plant(probes, verifier, change_number, "wrong number", rng, concurrency),
    ]
    report.sample = [
        _review_row(probe.question_id, probe.claim, probe.blocks)
        for probe in rng.sample(probes, min(sample_size, len(probes)))
    ]
    report.flagged = _balanced(report.flagged, sample_size)
    return report


def _balanced(rows: list[dict[str, object]], per_status: int) -> list[dict[str, object]]:
    """The same number of rows of each status, so a common status does not crowd out a rare one."""
    kept: dict[str, list[dict[str, object]]] = {}
    for row in sorted(rows, key=lambda row: (str(row["status"]), str(row["question_id"]))):
        kept.setdefault(str(row["status"]), []).append(row)
    return [row for group in kept.values() for row in group[:per_status]]


def _review_row(
    question_id: str, claim: Claim, blocks: Sequence[ContextBlock]
) -> dict[str, object]:
    return {
        "question_id": question_id,
        "status": claim.status,
        "statement": claim.text,
        "cites": list(claim.markers),
        "reason": claim.reason,
        "evidence": claim.quote,
        "passages": {b.number: b.text for b in blocks if b.number in claim.markers},
    }


Damaged = tuple[str, tuple[int, ...]]  # the statement and the markers it now carries
Damager = Callable[[Probe, random.Random], tuple[Damaged | None, bool]]


def move_citation(probe: Probe, rng: random.Random) -> tuple[Damaged | None, bool]:
    """Cite another passage of the same question. Also returns whether that passage might still
    support the statement, because it holds every value the statement states."""
    if len(probe.claim.markers) != 1:
        return None, False
    others = [block for block in probe.blocks if block.number != probe.claim.markers[0]]
    if not others:
        return None, False
    block = rng.choice(others)
    numbers, identifiers = values_of(probe.claim.text)
    plausible = bool(numbers or identifiers) and not missing_values(
        probe.claim.text, [visible_text(block)]
    )
    return (probe.claim.text, (block.number,)), plausible


def change_number(probe: Probe, rng: random.Random) -> tuple[Damaged | None, bool]:
    """Replace a number by one that none of the cited passages contains."""
    cited = [visible_text(b) for b in probe.blocks if b.number in probe.claim.markers]
    found = list(NUMBER.finditer(probe.claim.text))
    if not found:
        return None, False
    target = rng.choice(found)
    for _ in range(50):
        replacement = str(rng.randint(2, 997))
        if replacement != target.group(0) and missing_values(replacement, cited):
            text = (
                probe.claim.text[: target.start()] + replacement + probe.claim.text[target.end() :]
            )
            return (text, probe.claim.markers), False
    return None, False


async def _plant(
    probes: Sequence[Probe],
    verifier: CitationVerifier,
    damager: Damager,
    kind: str,
    rng: random.Random,
    concurrency: int,
) -> Outcome:
    outcome = Outcome(kind)
    grouped: dict[str, list[tuple[str, tuple[int, ...], bool, Probe]]] = {}
    for probe in probes:
        damaged, ambiguous = damager(probe, rng)
        if damaged is not None:
            grouped.setdefault(probe.question_id, []).append((*damaged, ambiguous, probe))
    semaphore = asyncio.Semaphore(concurrency)

    async def verify(items: list[tuple[str, tuple[int, ...], bool, Probe]]) -> None:
        text = " ".join(
            f"{statement} " + "".join(f"[{n}]" for n in markers)
            for statement, markers, _, _ in items
        )
        async with semaphore:
            result = await verifier.verify(text, items[0][3].blocks)
        if len(result.claims) != len(items):  # the answer was split differently: do not guess
            return
        for (_, _, ambiguous, _), claim in zip(items, result.claims, strict=True):
            outcome.planted += 1
            outcome.flagged += claim.status != "verified"
            outcome.judge_flagged += claim.judged in ("partial", "no")
            outcome.ambiguous += ambiguous

    await asyncio.gather(*(verify(items) for items in grouped.values()))
    return outcome


def render(report: CheckReport) -> str:
    total = sum(report.statuses.values())
    lines = [
        "# Citation verification check",
        "",
        f"{report.answers} answers were written and verified ({report.failed} failed); "
        f"{report.answers_all_verified} of them had every statement verified. "
        f"Cost of the check: ${report.cost_usd:.2f}.",
        *_judge_failures(report),
        "",
        "## What the verifier said about real answers",
        "",
        "| Status | Statements | Share |",
        "|---|---|---|",
    ]
    for status, count in report.statuses.most_common():
        lines.append(f"| {status} | {count} | {100 * count / total:.1f}% |")
    lines += [
        "",
        "## Damage planted in verified statements",
        "",
        "| Damage | Planted | Caught by the judge alone | Caught by the whole check | "
        "Ambiguous (the new passage may still support it) |",
        "|---|---|---|---|---|",
    ]
    for outcome in report.outcomes:
        lines.append(
            f"| {outcome.kind} | {outcome.planted} | {100 * outcome.judge_rate:.0f}% "
            f"| {100 * outcome.rate:.0f}% | {outcome.ambiguous} |"
        )
    return "\n".join(lines) + "\n"


def _judge_failures(report: CheckReport) -> list[str]:
    if not report.judge_errors:
        return []
    detail = "; ".join(f"{count} x {error}" for error, count in report.judge_errors.most_common())
    return ["", f"The judge could not give verdicts for some answers: {detail}."]
