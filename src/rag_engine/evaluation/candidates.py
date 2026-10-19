"""Candidate lists: what each retriever returns for each question, collected once and reused.

Running the real searches is the slow part of an evaluation (the index is large and the machine
may be small). Collecting the first 100 chunks of both retrievers per question lets every fusion
setting, chunk budget and reranker be compared offline on exactly the same candidates.
"""

import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from rag_engine.core.models import ChunkHit
from rag_engine.core.ports import ChunkSearcher, Embedder, SparseEncoder
from rag_engine.evaluation.questions import Question
from rag_engine.evaluation.store import append_jsonl, hit_from_dict, hit_to_dict, read_jsonl


@dataclass(frozen=True, slots=True)
class QuestionCandidates:
    question_id: str
    dense: list[ChunkHit]
    bm25: list[ChunkHit]


def load_candidates(path: Path) -> dict[str, QuestionCandidates]:
    return {
        row["question_id"]: QuestionCandidates(
            question_id=row["question_id"],
            dense=[hit_from_dict(hit) for hit in row["dense"]],
            bm25=[hit_from_dict(hit) for hit in row["bm25"]],
        )
        for row in read_jsonl(path)
    }


async def collect_candidates(
    questions: Sequence[Question],
    searcher: ChunkSearcher,
    embedder: Embedder,
    sparse_encoder: SparseEncoder,
    out: Path,
    *,
    depth: int = 100,
    concurrency: int = 2,
    on_progress: Callable[[int, int], None] | None = None,
) -> int:
    """Search every question that is not in `out` yet and append its candidates. Returns how many
    questions were searched in this call, so an interrupted run continues where it stopped."""
    done = {row["question_id"] for row in read_jsonl(out)}
    pending = [question for question in questions if question.question_id not in done]
    semaphore = asyncio.Semaphore(concurrency)

    async def search(question: Question) -> QuestionCandidates:
        async with semaphore:
            vector, sparse = await asyncio.gather(
                asyncio.to_thread(embedder.embed_query, question.question),
                asyncio.to_thread(sparse_encoder.encode_query, question.question),
            )
            dense, bm25 = await asyncio.gather(
                searcher.search_dense(vector.tolist(), limit=depth),
                searcher.search_sparse(sparse, limit=depth),
            )
        return QuestionCandidates(question.question_id, dense, bm25)

    finished = 0
    for completed in asyncio.as_completed([search(question) for question in pending]):
        item = await completed
        append_jsonl(
            out,
            [
                {
                    "question_id": item.question_id,
                    "dense": [hit_to_dict(hit) for hit in item.dense],
                    "bm25": [hit_to_dict(hit) for hit in item.bm25],
                }
            ],
        )
        finished += 1
        if on_progress is not None:
            on_progress(finished, len(pending))
    return finished


def candidate_ids(candidates: Mapping[str, QuestionCandidates]) -> set[str]:
    return set(candidates)
