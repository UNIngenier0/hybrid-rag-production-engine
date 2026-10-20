"""A retriever that replays stored candidates, so answers can be generated without the index.

The candidates were collected from the real index and the reranker scores come from the GPU job,
so the only services an answer run needs are the model and, for the parent passages, PostgreSQL.
Fusion is the equal-weight one the scores were computed on.
"""

from collections.abc import Mapping, Sequence
from dataclasses import replace

from rag_engine.core.models import ChunkHit, RetrievalMode, RetrievalResult, SearchFilters
from rag_engine.core.ports import PassageStore
from rag_engine.evaluation.candidates import QuestionCandidates
from rag_engine.evaluation.questions import Question
from rag_engine.evaluation.variants import (
    FusionSetting,
    bm25_ranking,
    dense_ranking,
    fused_ranking,
    reranked_ranking,
)


class StoredRetriever:
    def __init__(
        self,
        questions: Sequence[Question],
        candidates: Mapping[str, QuestionCandidates],
        scores: Mapping[str, Mapping[str, float]] | None = None,
        passages: PassageStore | None = None,
        *,
        fusion: FusionSetting | None = None,
        rerank_pool: int = 30,
    ) -> None:
        self._by_text = {question.question: question.question_id for question in questions}
        self._candidates = candidates
        self._scores = scores
        self._passages = passages
        self._fusion = fusion or FusionSetting()
        self._rerank_pool = rerank_pool

    async def retrieve(
        self,
        query: str,
        *,
        mode: RetrievalMode = "hybrid",
        filters: SearchFilters | None = None,
        limit: int | None = None,
        include_context: bool = False,
    ) -> RetrievalResult:
        if filters is not None and (filters.source_types or filters.doc_ids):
            raise ValueError("stored candidates cannot be filtered")
        question_id = self._by_text[query]
        candidates = self._candidates[question_id]
        hits = self._rank(candidates, mode)[: limit or 30]
        if include_context:
            hits = await self._with_context(hits)
        return RetrievalResult(query=query, mode=mode, hits=hits)

    def _rank(self, candidates: QuestionCandidates, mode: RetrievalMode) -> list[ChunkHit]:
        if mode == "dense":
            return dense_ranking(candidates)
        if mode == "bm25":
            return bm25_ranking(candidates)
        fused = fused_ranking(candidates, self._fusion)
        if mode == "hybrid":
            return fused
        if self._scores is None:
            raise ValueError("no reranker scores were given")
        return reranked_ranking(fused, self._scores[candidates.question_id], self._rerank_pool)

    async def _with_context(self, hits: list[ChunkHit]) -> list[ChunkHit]:
        if self._passages is None:
            raise ValueError("no passage store was given")
        passages = await self._passages.passages([hit.parent_span for hit in hits])
        return [replace(hit, context=passages.get(hit.parent_span)) for hit in hits]
