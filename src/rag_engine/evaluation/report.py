"""The retrieval ablation: every retriever, fusion setting and reranker on the same candidates."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any

from rag_engine.core.models import ChunkHit
from rag_engine.evaluation.candidates import QuestionCandidates
from rag_engine.evaluation.metrics import (
    QuestionScore,
    distinct_documents,
    paired_difference,
    score_question,
    summarize,
    summarize_by_type,
)
from rag_engine.evaluation.questions import Question
from rag_engine.evaluation.variants import (
    FusionSetting,
    bm25_ranking,
    dense_ranking,
    fused_ranking,
    reranked_ranking,
)

# Chunks that reach the LLM, and a more generous budget.
BUDGETS = (5, 10)
BASELINE = "hybrid, equal weights"
RERANK_POOLS = (20, 30, 50)
BM25_WEIGHTS = (1.5, 2.0, 3.0)

Ranker = Callable[[QuestionCandidates], list[ChunkHit]]


@dataclass(frozen=True, slots=True)
class RerankerRun:
    """Cross-encoder scores for the fused candidates of each question."""

    label: str
    model: str
    scores: Mapping[str, Mapping[str, float]]
    ms_per_pair: float
    device: str = "unknown"


@dataclass(frozen=True, slots=True)
class System:
    name: str
    rank: Ranker


@dataclass(frozen=True, slots=True)
class Report:
    markdown: str
    data: dict[str, Any]


def standard_systems(rerankers: Sequence[RerankerRun]) -> list[System]:
    systems = [
        System("dense", dense_ranking),
        System("bm25", bm25_ranking),
        System(BASELINE, fused_ranking),
    ]
    for weight in BM25_WEIGHTS:
        setting = FusionSetting(sparse_weight=weight)
        systems.append(System(f"hybrid, BM25 weight {weight:g}", partial(_fused, setting=setting)))
    systems.append(System("hybrid, depth 100", partial(_fused, setting=FusionSetting(depth=100))))
    for run in rerankers:
        for pool in RERANK_POOLS:
            systems.append(System(f"rerank {run.label}, pool {pool}", _reranker(run, pool)))
    return systems


def build_report(
    questions: Sequence[Question],
    candidates: Mapping[str, QuestionCandidates],
    rerankers: Sequence[RerankerRun] = (),
) -> Report:
    usable = [
        q
        for q in questions
        if q.gold_doc_ids
        and q.question_id in candidates
        and all(q.question_id in run.scores for run in rerankers)
    ]
    systems = standard_systems(rerankers)
    results: dict[tuple[str, int], list[QuestionScore]] = {}
    for system in systems:
        for budget in BUDGETS:
            results[(system.name, budget)] = [
                _score(q, system.rank(candidates[q.question_id])[:budget]) for q in usable
            ]
    ceilings = _ceilings(usable, candidates)

    lines = [
        "# Retrieval ablation",
        "",
        f"{len(usable)} questions with gold documents, scored on the distinct documents among the "
        "first N chunks a system returns. Dense and BM25 search 100 chunks deep over the full "
        "index; "
        "fusion is chunk-level RRF (k = 60) and uses the first 50 of each list unless stated. "
        "Recall is the share of a question's gold documents that were found. `*` marks a paired "
        "difference whose 95% interval excludes zero.",
    ]
    data: dict[str, Any] = {"questions": len(usable), "systems": {}, "ceilings": ceilings}
    for budget in BUDGETS:
        lines += ["", f"## {budget} chunks returned", "", *_table(systems, results, budget)]
    lines += ["", "## Candidate pools (ceiling for a reranker)", "", *_ceiling_table(ceilings)]
    lines += [
        "",
        "## Recall by question type, 5 chunks",
        "",
        *_type_table(usable, systems, results),
    ]
    if rerankers:
        lines += ["", "## Reranker cost", "", *_cost_table(rerankers)]
    for system in systems:
        data["systems"][system.name] = {
            str(budget): {
                "overall": asdict(summarize(results[(system.name, budget)])),
                "by_type": {
                    qtype: asdict(summary)
                    for qtype, summary in summarize_by_type(
                        list(
                            zip(
                                (q.question_type for q in usable),
                                results[(system.name, budget)],
                                strict=True,
                            )
                        )
                    ).items()
                },
            }
            for budget in BUDGETS
        }
    return Report("\n".join(lines) + "\n", data)


def _fused(candidates: QuestionCandidates, *, setting: FusionSetting) -> list[ChunkHit]:
    return fused_ranking(candidates, setting)


def _reranker(run: RerankerRun, pool: int) -> Ranker:
    def rank(candidates: QuestionCandidates) -> list[ChunkHit]:
        scores = run.scores[candidates.question_id]
        return reranked_ranking(fused_ranking(candidates), scores, pool)

    return rank


def _score(question: Question, chunks: Sequence[ChunkHit]) -> QuestionScore:
    documents = distinct_documents(hit.doc_id for hit in chunks)
    return score_question(question.gold_doc_ids, documents)


def _ceilings(
    questions: Sequence[Question], candidates: Mapping[str, QuestionCandidates]
) -> dict[str, float]:
    pools: dict[str, Ranker] = {
        "dense, 50 chunks": partial(dense_ranking, depth=50),
        "BM25, 50 chunks": partial(bm25_ranking, depth=50),
        "hybrid depth 50, 30 chunks": fused_ranking,
        "hybrid depth 50, 50 chunks": fused_ranking,
        "hybrid depth 100, 50 chunks": partial(_fused, setting=FusionSetting(depth=100)),
        "hybrid depth 100, 100 chunks": partial(_fused, setting=FusionSetting(depth=100)),
    }
    sizes = {name: int(name.rsplit(",", 1)[1].split()[0]) for name in pools}
    return {
        name: summarize(
            [_score(q, rank(candidates[q.question_id])[: sizes[name]]) for q in questions]
        ).recall
        for name, rank in pools.items()
    }


def _table(
    systems: Sequence[System], results: Mapping[tuple[str, int], list[QuestionScore]], budget: int
) -> list[str]:
    baseline = [s.recall for s in results[(BASELINE, budget)]]
    rows = [
        "| System | Recall | Any gold | All gold | MRR | nDCG@10 | Recall vs hybrid |",
        "|---|---|---|---|---|---|---|",
    ]
    for system in systems:
        scores = results[(system.name, budget)]
        summary = summarize(scores)
        if system.name == BASELINE:
            delta = ""
        else:
            diff = paired_difference([s.recall for s in scores], baseline)
            star = "*" if diff.significant else ""
            delta = f"{diff.mean:+.3f} ({diff.low:+.3f}, {diff.high:+.3f}){star}"
        rows.append(
            f"| {system.name} | {summary.recall:.3f} | {summary.hit:.3f} | {summary.all_gold:.3f} "
            f"| {summary.mrr:.3f} | {summary.ndcg:.3f} | {delta} |"
        )
    return rows


def _ceiling_table(ceilings: Mapping[str, float]) -> list[str]:
    rows = ["| Candidate pool | Recall |", "|---|---|"]
    rows += [f"| {name} | {recall:.3f} |" for name, recall in ceilings.items()]
    return rows


def _type_table(
    questions: Sequence[Question],
    systems: Sequence[System],
    results: Mapping[tuple[str, int], list[QuestionScore]],
) -> list[str]:
    types = sorted({q.question_type for q in questions})
    counts = {t: sum(1 for q in questions if q.question_type == t) for t in types}
    best = _best_reranker(systems, results)
    shown = [BASELINE, "dense", "bm25", *([best] if best else [])]
    rows = [
        "| System | " + " | ".join(f"{t} ({counts[t]})" for t in types) + " |",
        "|---|" + "---|" * len(types),
    ]
    for name in shown:
        by_type = summarize_by_type(
            list(zip((q.question_type for q in questions), results[(name, 5)], strict=True))
        )
        rows.append(f"| {name} | " + " | ".join(f"{by_type[t].recall:.3f}" for t in types) + " |")
    return rows


def _best_reranker(
    systems: Sequence[System], results: Mapping[tuple[str, int], list[QuestionScore]]
) -> str | None:
    names = [s.name for s in systems if s.name.startswith("rerank ")]
    if not names:
        return None
    return max(names, key=lambda name: summarize(results[(name, 5)]).recall)


def _cost_table(rerankers: Sequence[RerankerRun]) -> list[str]:
    rows = [
        "| Reranker | Model | Scored on | ms per pair | Time to score 30 passages |",
        "|---|---|---|---|---|",
    ]
    for run in rerankers:
        rows.append(
            f"| {run.label} | `{run.model}` | {run.device} | {run.ms_per_pair:.1f} "
            f"| {30 * run.ms_per_pair:.0f} ms |"
        )
    rows.append("")
    rows.append("Speed is whatever the named device delivered while scoring; it says nothing about")
    rows.append("another machine.")
    return rows
