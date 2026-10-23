"""The query pipeline: retrieve, write a cited answer, verify the citations."""

import time
from dataclasses import dataclass

from rag_engine.core.models import LLMUsage, RetrievalMode, SearchFilters
from rag_engine.generation.answer import AnswerGenerator, GeneratedAnswer
from rag_engine.generation.cost import cost_usd
from rag_engine.generation.coverage import ABSTAINED, Coverage, coverage
from rag_engine.generation.verify import CitationVerifier, Verification


@dataclass(frozen=True, slots=True)
class QueryResult:
    question: str
    answer: GeneratedAnswer
    verification: Verification
    coverage: Coverage
    timings_ms: dict[str, float]
    input_tokens: int
    output_tokens: int
    cost_usd: float | None  # None when the model has no known price


class QueryService:
    """Everything between a question and a verified answer. The HTTP layer only calls `query`."""

    def __init__(self, generator: AnswerGenerator, verifier: CitationVerifier) -> None:
        self._generator = generator
        self._verifier = verifier

    async def query(
        self,
        question: str,
        *,
        mode: RetrievalMode | None = None,
        chunks: int | None = None,
        filters: SearchFilters | None = None,
        verify: bool = True,
    ) -> QueryResult:
        started = time.perf_counter()
        answer = await self._generator.answer(question, mode=mode, chunks=chunks, filters=filters)
        verifying = time.perf_counter()
        if answer.abstention is not None:
            # Nothing was written, so there is nothing to check and nothing was spent on a model.
            verification = Verification(claims=[], method="none")
            covered = ABSTAINED
        else:
            covered = coverage(answer.raw_text)
            verification = await self._verifier.verify(
                answer.raw_text, answer.blocks, use_judge=verify
            )
        finished = time.perf_counter()
        timings = {
            **{f"retrieval_{stage}": ms for stage, ms in answer.retrieval_timings_ms.items()},
            "generation": answer.generation_ms,
            "verification": round((finished - verifying) * 1000, 1),
            "total": round((finished - started) * 1000, 1),
        }
        usages = [] if answer.abstention is not None else [(answer.model, answer.usage)]
        if verification.usage is not None and verification.model is not None:
            usages.append((verification.model, verification.usage))
        return QueryResult(
            question=question,
            answer=answer,
            verification=verification,
            coverage=covered,
            timings_ms=timings,
            input_tokens=sum(usage.input_tokens for _, usage in usages),
            output_tokens=sum(usage.output_tokens for _, usage in usages),
            cost_usd=_cost(usages),
        )


def _cost(usages: list[tuple[str, LLMUsage]]) -> float | None:
    try:
        return round(sum(cost_usd(model, usage) for model, usage in usages), 6)
    except KeyError:
        return None
