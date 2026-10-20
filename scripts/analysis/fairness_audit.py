"""Checks on how representative and how robust the answer comparison is.

Run from the repository root with the project environment:

    uv run python scripts/analysis/fairness_audit.py

It needs the stored candidates and reranker scores in eval/cache and the benchmark questions in
data/raw/questions, and uses the run folders in eval/runs. Nothing here calls a model or a service.
The tables are the ones quoted in docs/evaluation.md.
"""

import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path

from rag_engine.core.models import ChunkHit
from rag_engine.evaluation.candidates import load_candidates
from rag_engine.evaluation.metrics import (
    distinct_documents,
    paired_difference,
    score_question,
)
from rag_engine.evaluation.questions import (
    Question,
    load_questions,
    stratified_subset,
    with_gold_documents,
)
from rag_engine.evaluation.rerank_scores import load_scores
from rag_engine.evaluation.variants import FusionSetting, fused_ranking, reranked_ranking

ROOT = Path.cwd()
JUDGE = "claude-haiku-5-5"
RERANKERS = {
    "bge-reranker-v2-m3": "BAAI_bge-reranker-v2-m3",
    "bge-reranker-base": "BAAI_bge-reranker-base",
    "mxbai-rerank-base": "mixedbread-ai_mxbai-rerank-base-v1",
    "MiniLM-L12": "cross-encoder_ms-marco-MiniLM-L-12-v2",
    "MiniLM-L6": "cross-encoder_ms-marco-MiniLM-L-6-v2",
}
SHIPPED_HYBRID = FusionSetting(sparse_weight=1.5)

Ranker = Callable[[Question], list[ChunkHit]]


def main() -> None:
    questions_file = ROOT / "data/raw/questions/test.parquet"
    candidates_file = ROOT / "eval/cache/candidates.jsonl.gz"
    if not questions_file.exists() or not candidates_file.exists():
        sys.exit("needs data/raw/questions and eval/cache (see docs/evaluation.md#how-it-runs)")
    questions = with_gold_documents(load_questions(questions_file))
    candidates = load_candidates(candidates_file)
    scores = {
        label: load_scores(ROOT / f"eval/cache/rerank-{name}.jsonl.gz")
        for label, name in RERANKERS.items()
    }
    evaluated = stratified_subset(questions, 12, seed=0)
    evaluated_ids = {q.question_id for q in evaluated}
    others = [q for q in questions if q.question_id not in evaluated_ids]

    def hybrid(q: Question, setting: FusionSetting | None = None) -> list[ChunkHit]:
        return fused_ranking(candidates[q.question_id], setting)

    def rerank(q: Question, label: str) -> list[ChunkHit]:
        return reranked_ranking(hybrid(q), scores[label][q.question_id], 30)

    print("# Fairness audit\n")
    print(f"{len(questions)} questions with gold documents; the answer comparison used")
    print(f"{len(evaluated)} of them (12 per type, seed 0).\n")
    reranker_choice(questions, evaluated, others, hybrid, rerank)
    shipped_vs_measured(questions, evaluated, candidates, scores, hybrid, rerank)
    answers_view(questions, evaluated_ids, others)


def recall(q: Question, ranked: Sequence[ChunkHit], budget: int = 5) -> float:
    documents = distinct_documents(hit.doc_id for hit in ranked[:budget])
    return score_question(q.gold_doc_ids, documents).recall


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def reranker_choice(
    questions: Sequence[Question],
    evaluated: Sequence[Question],
    others: Sequence[Question],
    hybrid: Callable[..., list[ChunkHit]],
    rerank: Callable[[Question, str], list[ChunkHit]],
) -> None:
    print("## Is the sample of 96 typical for retrieval? (document recall at 5 chunks)\n")
    print("| Questions | n | Hybrid | " + " | ".join(RERANKERS) + " |")
    print("|---|---|---|" + "---|" * len(RERANKERS))
    for name, group in (
        ("the 96 evaluated", evaluated),
        ("the other 374", others),
        ("all 470", questions),
    ):
        base = [recall(q, hybrid(q)) for q in group]
        cells = []
        for label in RERANKERS:
            values = [recall(q, rerank(q, label)) for q in group]
            diff = paired_difference(values, base)
            cells.append(f"{mean(values):.3f} ({diff.mean:+.3f})")
        print(f"| {name} | {len(group)} | {mean(base):.3f} | " + " | ".join(cells) + " |")
    print()


def shipped_vs_measured(
    questions: Sequence[Question],
    evaluated: Sequence[Question],
    candidates: Mapping[str, object],
    scores: Mapping[str, Mapping[str, Mapping[str, float]]],
    hybrid: Callable[..., list[ChunkHit]],
    rerank: Callable[[Question, str], list[ChunkHit]],
) -> None:
    best = scores["bge-reranker-v2-m3"]
    missing = [
        sum(1 for hit in hybrid(q, SHIPPED_HYBRID)[:30] if hit.point_id not in best[q.question_id])
        for q in questions
    ]
    print("## Reranking with the heavier BM25 weight (1.5) instead of equal weights\n")
    print(
        f"With weight 1.5 the pool of 30 holds chunks the reranker never scored for "
        f"{sum(1 for m in missing if m)} of {len(questions)} questions (mean {mean(missing):.2f} "
        "chunks). Scoring only the chunks that have a score is a lower bound for that pipeline:\n"
    )
    print("| Questions | Equal weights + rerank | Weight 1.5 + rerank (lower bound) | Difference |")
    print("|---|---|---|---|")

    def heavier(q: Question) -> list[ChunkHit]:
        known = best[q.question_id]
        pool = [hit for hit in hybrid(q, SHIPPED_HYBRID)[:30] if hit.point_id in known]
        ordered = sorted(pool, key=lambda hit: known[hit.point_id], reverse=True)
        return [replace(hit, score=known[hit.point_id]) for hit in ordered]

    for name, group in (("the 96 evaluated", evaluated), ("all 470", questions)):
        equal = [recall(q, rerank(q, "bge-reranker-v2-m3")) for q in group]
        weighted = [recall(q, heavier(q)) for q in group]
        diff = paired_difference(weighted, equal)
        print(
            f"| {name} | {mean(equal):.3f} | {mean(weighted):.3f} "
            f"| {diff.mean:+.3f} ({diff.low:+.3f}, {diff.high:+.3f}) |"
        )
    print(
        "\nNo gain is visible once the cross-encoder reorders the pool, so the service reranks a "
        "pool fused with equal weights, which is the pipeline every measurement used.\n"
    )


def read_results(run: str, label: str = JUDGE) -> dict[str, tuple[float, bool]]:
    """Per question: completeness (0 to 1) and whether the answer was judged correct."""
    path = ROOT / f"eval/runs/{run}/results-{label}.json"
    if not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))["questions"]
    return {
        r["question_id"]: (r["completeness_pct"] / 100, bool(r["answer_correct"])) for r in rows
    }


def judge_repeatability(runs: Mapping[str, str], ids: Sequence[str]) -> None:
    """The same answers judged twice by the same judge (evaluate.sh with LABEL=repeat)."""
    rows = []
    for name, run in runs.items():
        first, again = read_results(run), read_results(run, "repeat")
        if len(again) < len(ids):
            return
        differ = sum(first[i][1] != again[i][1] for i in ids)
        before, after = 100 * mean(combined(first, ids)), 100 * mean(combined(again, ids))
        rows.append(f"| {name} | {before:.1f} | {after:.1f} | {differ} of {len(ids)} |")
    print("## The same answers judged twice\n")
    print(
        "| System | Combined, first judgment | Combined, second judgment | Verdicts that changed |"
    )
    print("|---|---|---|---|")
    print("\n".join(rows) + "\n")


def combined(results: Mapping[str, tuple[float, bool]], ids: Sequence[str]) -> list[float]:
    return [results[i][0] if results[i][1] else 0.0 for i in ids]


def answers_view(
    questions: Sequence[Question], evaluated_ids: set[str], others: Sequence[Question]
) -> None:
    runs = {
        "Hybrid + rerank (ours)": "rag-haiku",
        "Iterative grep agent": "grep-agent-haiku",
        "Single-attempt grep": "grep-oneshot-haiku",
    }
    results = {name: read_results(run) for name, run in runs.items()}
    ids = sorted(evaluated_ids)
    pilot = {"qst_0011", "qst_0067", "qst_0056"}  # the three questions used to try the prompts
    print("## Robustness of the answer comparison (combined score)\n")
    print("| System | All 96 | Without the 3 pilot questions |")
    print("|---|---|---|")
    for name, rows in results.items():
        rest = [i for i in ids if i not in pilot]
        everything, without_pilot = mean(combined(rows, ids)), mean(combined(rows, rest))
        print(f"| {name} | {100 * everything:.1f} | {100 * without_pilot:.1f} |")
    rest = [i for i in ids if i not in pilot]
    gap = paired_difference(
        combined(results["Hybrid + rerank (ours)"], rest),
        combined(results["Iterative grep agent"], rest),
    )
    print(
        f"\nOurs minus the agent without the pilot questions: {100 * gap.mean:+.1f} "
        f"({100 * gap.low:+.1f}, {100 * gap.high:+.1f}).\n"
    )

    judge_repeatability(runs, ids)

    searches = ROOT / "eval/runs/grep-oneshot-haiku/trace.jsonl"
    empty = [
        row["question_id"]
        for row in map(json.loads, searches.read_text(encoding="utf-8").splitlines())
        if "retrieved_doc_ids" in row and not row["retrieved_doc_ids"]
    ]
    grep = results["Single-attempt grep"]
    best_case = (sum(combined(grep, [i for i in ids if i not in empty])) + len(empty)) / len(ids)
    print(
        f"The single-attempt grep returned no file for {len(empty)} of {len(ids)} questions (its "
        f"search never finished or matched nothing). Even if all {len(empty)} had been answered "
        f"perfectly, it would score {100 * best_case:.1f}.\n"
    )

    full = read_results("rag-haiku-all")
    if len(full) < len(questions):
        return  # that run has not been made, or has not been judged yet
    print("## Our RAG on every question with gold documents\n")
    print("| Questions | n | Combined | Correct |")
    print("|---|---|---|---|")
    groups = (
        ("the 96, first run", results["Hybrid + rerank (ours)"], ids),
        ("the same 96, second run", full, ids),
        ("the other 374", full, [q.question_id for q in others]),
        ("all 470", full, [q.question_id for q in questions]),
    )
    for name, rows, group in groups:
        values = combined(rows, group)
        correct = mean([float(rows[i][1]) for i in group])
        print(f"| {name} | {len(group)} | {100 * mean(values):.1f} | {100 * correct:.1f}% |")
    agent = combined(results["Iterative grep agent"], ids)
    print("\nOurs minus the iterative agent on the same 96 questions, for each run of ours:\n")
    for name, rows in (("first run", results["Hybrid + rerank (ours)"]), ("second run", full)):
        gap = paired_difference(combined(rows, ids), agent)
        print(f"- {name}: {100 * gap.mean:+.1f} ({100 * gap.low:+.1f}, {100 * gap.high:+.1f})")
    print()


if __name__ == "__main__":
    main()
