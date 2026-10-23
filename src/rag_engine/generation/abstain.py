"""When the evidence is too weak to answer: decline instead of writing something.

Two cases, decided from the retrieval result alone, before any model call:

- No passage was retrieved. There is nothing to answer from.
- The best passage scores below the threshold. This only means something for reranked results,
  where the score is the cross-encoder's: the fused score of plain hybrid search and the cosine
  or BM25 scores of the single retrievers are not on a scale that a threshold can use.

A model can also decline by itself (the answer prompt tells it to say when the passages do not
hold the answer); that is not decided here.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from rag_engine.core.models import ChunkHit, RetrievalMode

MESSAGE = (
    "I could not find enough information in the documents to answer this question. "
    "The passages that came closest do not state it."
)


@dataclass(frozen=True, slots=True)
class Abstention:
    """Why no answer was written."""

    reason: str  # "no_passages" or "weak_evidence"
    top_score: float | None  # the best reranker score, when there was one
    threshold: float | None
    message: str = MESSAGE


class AbstentionPolicy:
    def __init__(self, threshold: float | None = None) -> None:
        self._threshold = threshold

    @property
    def threshold(self) -> float | None:
        return self._threshold

    def check(self, hits: Sequence[ChunkHit], mode: RetrievalMode) -> Abstention | None:
        """None when an answer should be written."""
        if not hits:
            return Abstention("no_passages", None, self._threshold)
        if self._threshold is None or mode != "hybrid+rerank":
            return None
        best = max(hit.score for hit in hits)
        if best < self._threshold:
            return Abstention("weak_evidence", best, self._threshold)
        return None
