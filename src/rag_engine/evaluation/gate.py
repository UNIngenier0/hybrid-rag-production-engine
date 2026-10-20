"""The retrieval regression gate: fail when a change makes the shipped pipeline find less.

The real index cannot run in CI, so the gate replays a small fixture instead: for every benchmark
question with gold documents, the ranked chunks each retriever returned from the real index and
the reranker's score for the fused pool. Fusion, reranking and the metrics run through the same
code the service uses, with the fusion settings the service ships with, so a change to any of them
moves the numbers. The numbers are compared with a committed baseline, and a drop beyond the
tolerance fails the build.

The fixture keeps no text and no chunk identifiers, only each chunk's document and its rank, so it
is small. What the gate cannot see is a change to the index itself (another embedding model, other
chunking): that needs the real services and a new fixture.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_engine.core.config import Settings
from rag_engine.core.models import ChunkHit
from rag_engine.evaluation.audit import audit_runs
from rag_engine.evaluation.candidates import QuestionCandidates
from rag_engine.evaluation.compare import compare
from rag_engine.evaluation.metrics import (
    QuestionScore,
    distinct_documents,
    score_question,
    summarize,
    summarize_by_type,
)
from rag_engine.evaluation.questions import Question
from rag_engine.evaluation.runs import load_arms
from rag_engine.evaluation.store import append_jsonl, read_jsonl
from rag_engine.evaluation.variants import (
    FusionSetting,
    bm25_ranking,
    dense_ranking,
    fused_ranking,
    reranked_ranking,
)

FIXTURE_FORMAT = 1
BUDGETS = (5, 10)  # chunks that reach the model, and a more generous budget
SYSTEMS = ("dense", "bm25", "hybrid", "hybrid+rerank")
OVERALL_METRICS = ("recall", "hit", "mrr", "ndcg")

# A drop beyond these fails the gate. The replay is deterministic, so any difference comes from a
# code or configuration change; the tolerance only lets a deliberate small trade-off through.
OVERALL_TOLERANCE = 0.005
BY_TYPE_TOLERANCE = 0.02


class GateError(Exception):
    """The gate could not run, as opposed to running and finding a regression."""


@dataclass(frozen=True, slots=True)
class GateCase:
    question: Question
    candidates: QuestionCandidates
    scores: Mapping[str, float]  # reranker score of each chunk in the fused pool


@dataclass(frozen=True, slots=True)
class Violation:
    metric: str
    baseline: float
    current: float
    tolerance: float

    @property
    def delta(self) -> float:
        return self.current - self.baseline


@dataclass(frozen=True, slots=True)
class Pipeline:
    """The retrieval settings the service ships with, read from the defaults of `Settings`."""

    depth: int
    rrf_k: int
    dense_weight: float
    sparse_weight: float
    rerank_sparse_weight: float
    rerank_pool: int

    def hybrid(self) -> FusionSetting:
        return FusionSetting(self.depth, self.rrf_k, self.dense_weight, self.sparse_weight)

    def rerank_pool_fusion(self) -> FusionSetting:
        return FusionSetting(self.depth, self.rrf_k, self.dense_weight, self.rerank_sparse_weight)

    def describe(self) -> dict[str, Any]:
        return {
            "candidates_per_retriever": self.depth,
            "rrf_k": self.rrf_k,
            "dense_weight": self.dense_weight,
            "sparse_weight": self.sparse_weight,
            "rerank_sparse_weight": self.rerank_sparse_weight,
            "rerank_pool": self.rerank_pool,
        }


def shipped_pipeline() -> Pipeline:
    defaults = {name: field.default for name, field in Settings.model_fields.items()}
    if defaults["dense_top_k"] != defaults["sparse_top_k"]:
        raise GateError("the gate replays equal candidate depths for both retrievers")
    return Pipeline(
        depth=int(defaults["dense_top_k"]),
        rrf_k=int(defaults["rrf_k"]),
        dense_weight=float(defaults["rrf_dense_weight"]),
        sparse_weight=float(defaults["rrf_sparse_weight"]),
        rerank_sparse_weight=float(defaults["rerank_sparse_weight"]),
        rerank_pool=int(defaults["rerank_top_k"]),
    )


def build_fixture(
    questions: Sequence[Question],
    candidates: Mapping[str, QuestionCandidates],
    scores: Mapping[str, Mapping[str, float]],
    out: Path,
    *,
    reranker: str,
    depth: int = 50,
) -> int:
    """Write the fixture for the questions that have candidates and reranker scores."""
    rows: list[dict[str, Any]] = [{"meta": {"format": FIXTURE_FORMAT, "depth": depth}}]
    rows[0]["meta"]["reranker"] = reranker
    for question in questions:
        found = candidates.get(question.question_id)
        pool = scores.get(question.question_id)
        if found is None or pool is None:
            continue
        number: dict[str, int] = {}
        documents: list[str] = []
        for hit in [*found.dense[:depth], *found.bm25[:depth]]:
            if hit.point_id not in number:
                number[hit.point_id] = len(documents)
                documents.append(hit.doc_id)
        rows.append(
            {
                "id": question.question_id,
                "type": question.question_type,
                "gold": list(question.gold_doc_ids),
                "chunks": documents,  # the document of each distinct chunk, by chunk number
                "dense": [number[hit.point_id] for hit in found.dense[:depth]],
                "bm25": [number[hit.point_id] for hit in found.bm25[:depth]],
                "scores": {
                    str(number[point_id]): round(float(score), 6)
                    for point_id, score in pool.items()
                    if point_id in number
                },
            }
        )
    out.unlink(missing_ok=True)
    append_jsonl(out, rows)
    return len(rows) - 1


def load_fixture(path: Path) -> tuple[dict[str, Any], list[GateCase]]:
    rows = list(read_jsonl(path))
    if not rows or "meta" not in rows[0]:
        raise GateError(f"{path} is not a gate fixture")
    meta = rows[0]["meta"]
    if meta.get("format") != FIXTURE_FORMAT:
        raise GateError(f"fixture format {meta.get('format')} is not {FIXTURE_FORMAT}")
    return meta, [_case(row) for row in rows[1:]]


def _case(row: dict[str, Any]) -> GateCase:
    question = Question(
        question_id=row["id"],
        question_type=row["type"],
        question="",
        gold_doc_ids=tuple(row["gold"]),
        source_types=(),
        gold_answer="",
        answer_facts=(),
    )
    chunks: list[str] = row["chunks"]

    def hit(number: int) -> ChunkHit:
        return ChunkHit(
            point_id=f"{row['id']}/{number}",
            doc_id=chunks[number],
            content_sha256="",
            source_type="",
            title="",
            section=None,
            chunk_index=0,
            text="",
            char_start=0,
            char_end=0,
            parent_start=0,
            parent_end=0,
            score=0.0,
        )

    return GateCase(
        question,
        QuestionCandidates(
            row["id"], [hit(n) for n in row["dense"]], [hit(n) for n in row["bm25"]]
        ),
        {f"{row['id']}/{number}": float(score) for number, score in row["scores"].items()},
    )


def measure(cases: Sequence[GateCase], pipeline: Pipeline) -> dict[str, Any]:
    """Recall and the rank metrics of every system at every budget, overall and by question type."""
    results: dict[str, dict[str, list[QuestionScore]]] = {
        system: {str(budget): [] for budget in BUDGETS} for system in SYSTEMS
    }
    for case in cases:
        for system, ranked in _rankings(case, pipeline).items():
            for budget in BUDGETS:
                documents = distinct_documents(hit.doc_id for hit in ranked[:budget])
                results[system][str(budget)].append(
                    score_question(case.question.gold_doc_ids, documents)
                )
    types = [case.question.question_type for case in cases]
    metrics: dict[str, Any] = {}
    for system, by_budget in results.items():
        metrics[system] = {}
        for label, scores in by_budget.items():
            overall = summarize(scores)
            by_type = summarize_by_type(list(zip(types, scores, strict=True)))
            metrics[system][label] = {
                "overall": {name: round(getattr(overall, name), 6) for name in OVERALL_METRICS},
                "by_type": {name: round(s.recall, 6) for name, s in by_type.items()},
            }
    return {"questions": len(cases), "metrics": metrics}


def _rankings(case: GateCase, pipeline: Pipeline) -> dict[str, list[ChunkHit]]:
    candidates = case.candidates
    pool = fused_ranking(candidates, pipeline.rerank_pool_fusion())
    try:
        rerank = reranked_ranking(pool, case.scores, pipeline.rerank_pool)
    except KeyError as error:
        raise GateError(
            f"{case.question.question_id}: the fixture has no reranker score for the pool this "
            f"configuration builds ({error.args[0]}). Score the new pool and rebuild the fixture."
        ) from error
    return {
        "dense": dense_ranking(candidates, pipeline.depth),
        "bm25": bm25_ranking(candidates, pipeline.depth),
        "hybrid": fused_ranking(candidates, pipeline.hybrid()),
        "hybrid+rerank": rerank,
    }


def violations(
    baseline: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    overall_tolerance: float = OVERALL_TOLERANCE,
    by_type_tolerance: float = BY_TYPE_TOLERANCE,
) -> list[Violation]:
    """Every tracked number that dropped by more than the tolerance (improvements never fail)."""
    found: list[Violation] = []
    for system, by_budget in baseline["metrics"].items():
        for budget, tracked in by_budget.items():
            now = current["metrics"][system][budget]
            for name, before in tracked["overall"].items():
                after = now["overall"][name]
                if before - after > overall_tolerance:
                    found.append(
                        Violation(f"{system} @{budget} {name}", before, after, overall_tolerance)
                    )
            for question_type, before in tracked["by_type"].items():
                after = now["by_type"].get(question_type, 0.0)
                if before - after > by_type_tolerance:
                    label = f"{system} @{budget} recall on {question_type}"
                    found.append(Violation(label, before, after, by_type_tolerance))
    return found


def fixture_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_baseline(
    fixture: Path, meta: Mapping[str, Any], pipeline: Pipeline, measured: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "fixture": {
            "sha256": fixture_sha256(fixture),
            "questions": measured["questions"],
            "reranker": meta.get("reranker"),
        },
        "pipeline": pipeline.describe(),
        "tolerance": {"overall": OVERALL_TOLERANCE, "by_type": BY_TYPE_TOLERANCE},
        "metrics": measured["metrics"],
    }


@dataclass(frozen=True, slots=True)
class GateResult:
    passed: bool
    text: str
    violations: list[Violation]


def run_gate(fixture: Path, baseline_path: Path) -> GateResult:
    """Replay the fixture and compare it with the committed baseline."""
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    problems: list[str] = []
    if baseline["fixture"]["sha256"] != fixture_sha256(fixture):
        problems.append(
            "the fixture changed since the baseline was written: review the new numbers and "
            "run `gate --update-baseline`"
        )
    meta, cases = load_fixture(fixture)
    pipeline = shipped_pipeline()
    if pipeline.describe() != baseline["pipeline"]:
        problems.append(
            f"the shipped pipeline is {pipeline.describe()}, the baseline was measured with "
            f"{baseline['pipeline']}"
        )
    current = measure(cases, pipeline)
    found = violations(baseline, current)
    return GateResult(
        passed=not found and not problems,
        text=_render(baseline, current, found, problems, meta),
        violations=found,
    )


def _render(
    baseline: Mapping[str, Any],
    current: Mapping[str, Any],
    found: Sequence[Violation],
    problems: Sequence[str],
    meta: Mapping[str, Any],
) -> str:
    lines = [
        f"Retrieval gate over {current['questions']} questions "
        f"(reranker scores from {meta.get('reranker')}).",
        "",
        "| System | Chunks | Recall | Baseline | Change |",
        "|---|---|---|---|---|",
    ]
    for system, by_budget in current["metrics"].items():
        for budget, now in by_budget.items():
            before = baseline["metrics"][system][budget]["overall"]["recall"]
            after = now["overall"]["recall"]
            lines.append(
                f"| {system} | {budget} | {after:.3f} | {before:.3f} | {after - before:+.3f} |"
            )
    lines.append("")
    if found:
        lines.append("Regressions beyond the tolerance:")
        lines += [
            f"- {v.metric}: {v.baseline:.3f} -> {v.current:.3f} ({v.delta:+.3f}, "
            f"tolerance {v.tolerance:g})"
            for v in found
        ]
    lines += [f"- {problem}" for problem in problems]
    if not found and not problems:
        lines.append("No regression.")
    return "\n".join(lines) + "\n"


# The answer comparison that is committed with the repository: its runs, its judge and its report.
COMMITTED_ARMS = (
    ("Hybrid + rerank (ours)", Path("eval/runs/rag-haiku")),
    ("Iterative grep agent", Path("eval/runs/grep-agent-haiku")),
    ("Single-attempt grep", Path("eval/runs/grep-oneshot-haiku")),
)
COMMITTED_JUDGE = "claude-haiku-5-5"
COMMITTED_REPORT = Path("eval/reports/answer-comparison.json")


def check_answer_report(
    arms: Sequence[tuple[str, Path]] = COMMITTED_ARMS,
    judge: str = COMMITTED_JUDGE,
    report: Path = COMMITTED_REPORT,
) -> list[str]:
    """Problems with the committed answer comparison: unfair setup or numbers that no longer
    match the run folders. Answers cannot be regenerated in CI (they cost money), so this checks
    that what is published is what the stored results say, and that the systems were comparable."""
    audit = audit_runs(arms, judge)
    problems = [f"{check.name}: {check.detail}" for check in audit.failures]
    comparison = compare(load_arms(arms, judge), arms[0][0], audit)
    if json.loads(report.read_text(encoding="utf-8")) != comparison.data:
        problems.append(
            f"{report} does not match the run folders: regenerate it with the `compare` command"
        )
    return problems
