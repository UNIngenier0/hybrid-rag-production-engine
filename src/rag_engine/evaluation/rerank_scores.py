"""Cross-encoder scores for the fused candidates, kept so reranking can be compared offline."""

import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from rag_engine.core.ports import Reranker
from rag_engine.evaluation.candidates import QuestionCandidates
from rag_engine.evaluation.questions import Question
from rag_engine.evaluation.store import append_jsonl, hit_to_dict, read_jsonl
from rag_engine.evaluation.variants import fused_ranking
from rag_engine.indexing.contextual import index_text

# The pool that gets scored is the default fusion's first chunks. Smaller pools are prefixes of it.
POOL_DEPTH = 50


def score_pools(
    questions: Sequence[Question],
    candidates: Mapping[str, QuestionCandidates],
    reranker: Reranker,
    out: Path,
    *,
    pool: int = POOL_DEPTH,
    on_progress: Callable[[int, int], None] | None = None,
) -> int:
    """Score the pool of every question that is not in `out` yet. Returns how many were scored."""
    done = {row["question_id"] for row in read_jsonl(out)}
    pending = [q for q in questions if q.question_id not in done and q.question_id in candidates]
    for number, question in enumerate(pending, start=1):
        hits = fused_ranking(candidates[question.question_id])[:pool]
        passages = [index_text(hit.title, hit.section, hit.text) for hit in hits]
        started = time.perf_counter()
        scores = reranker.score(question.question, passages)
        seconds = time.perf_counter() - started
        append_jsonl(
            out,
            [
                {
                    "question_id": question.question_id,
                    "model": reranker.model_name,
                    "device": getattr(reranker, "device", "unknown"),
                    "pairs": len(hits),
                    "seconds": seconds,
                    "scores": {hit.point_id: float(s) for hit, s in zip(hits, scores, strict=True)},
                }
            ],
        )
        if on_progress is not None:
            on_progress(number, len(pending))
    return len(pending)


def export_pools(
    questions: Sequence[Question],
    candidates: Mapping[str, QuestionCandidates],
    out: Path,
    *,
    pool: int = POOL_DEPTH,
) -> int:
    """Write the fused pool of each question as a compact candidates file, for scoring elsewhere.

    The pool goes in the `dense` list, already in fused order, so scoring the exported file picks
    the same chunks as scoring the full one. Only the pool is kept, which makes the file about a
    quarter of the size: the part worth shipping to a machine that has the GPU.
    """
    rows = [
        {
            "question_id": question.question_id,
            "dense": [
                hit_to_dict(hit) for hit in fused_ranking(candidates[question.question_id])[:pool]
            ],
            "bm25": [],
        }
        for question in questions
        if question.question_id in candidates
    ]
    out.unlink(missing_ok=True)
    append_jsonl(out, rows)
    return len(rows)


def load_scores(path: Path) -> dict[str, dict[str, float]]:
    return {row["question_id"]: row["scores"] for row in read_jsonl(path)}


def scoring_speed(path: Path) -> tuple[str, float, str]:
    """The model, the milliseconds one (query, passage) pair took on average, and the device."""
    pairs = seconds = 0.0
    model = device = ""
    for row in read_jsonl(path):
        model = row["model"]
        device = row.get("device", "unknown")
        pairs += row["pairs"]
        seconds += row["seconds"]
    return model, (1000 * seconds / pairs) if pairs else 0.0, device
