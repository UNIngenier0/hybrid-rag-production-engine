"""Answer a question from retrieved passages and report which documents the answer rests on."""

import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from rag_engine.core.errors import LLMError
from rag_engine.core.models import ChunkHit, LLMUsage, RetrievalMode, SearchFilters
from rag_engine.core.ports import LLMClient, Retriever
from rag_engine.generation.abstain import Abstention, AbstentionPolicy
from rag_engine.generation.citations import cited_numbers, resolve, strip_markers
from rag_engine.generation.prompt import (
    SYSTEM_PROMPT,
    ContextBlock,
    build_context,
    render_prompt,
)


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    text: str  # the answer without its [n] markers
    raw_text: str  # exactly what the model wrote
    cited_doc_ids: list[str]  # documents behind the markers that point at a passage
    retrieved_doc_ids: list[str]  # documents of every passage the model was shown
    unresolved_markers: list[int]  # markers that point at no passage
    usage: LLMUsage
    model: str
    # The numbered passages the model was shown, so the citations can be checked against them.
    blocks: tuple[ContextBlock, ...] = ()
    mode: RetrievalMode = "hybrid"
    retrieval_timings_ms: dict[str, float] = field(default_factory=dict)
    generation_ms: float = 0.0
    # Set when the evidence was too weak and no answer was written (no model call was made).
    abstention: Abstention | None = None


class AnswerGenerator:
    """Retrieve, number the passages, ask the model and read the citations back."""

    def __init__(
        self,
        retriever: Retriever,
        llm: LLMClient,
        *,
        mode: RetrievalMode = "hybrid+rerank",
        chunks: int = 5,
        max_tokens: int = 4000,
        temperature: float | None = None,  # newer models refuse a temperature (HTTP 400)
        abstention: AbstentionPolicy | None = None,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._abstention = abstention or AbstentionPolicy()
        self._mode: RetrievalMode = mode
        self._chunks = chunks
        self._max_tokens = max_tokens
        self._temperature = temperature

    async def answer(
        self,
        question: str,
        *,
        mode: RetrievalMode | None = None,
        chunks: int | None = None,
        filters: SearchFilters | None = None,
    ) -> GeneratedAnswer:
        """Answer from the best chunks. The options override, for this question only, what the
        generator was built with."""
        mode = mode or self._mode
        result = await self._retriever.retrieve(
            question,
            mode=mode,
            filters=filters,
            limit=chunks or self._chunks,
            include_context=True,
        )
        declined = self._abstention.check(result.hits, mode)
        if declined is not None:
            return _declined(declined, result.hits, mode, result.timings_ms)
        blocks = build_context(result.hits)
        started = time.perf_counter()
        reply = await self._llm.complete(
            SYSTEM_PROMPT,
            render_prompt(question, blocks),
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        generation_ms = round((time.perf_counter() - started) * 1000, 1)
        if not reply.text.strip():
            # The model spent its whole budget reasoning: better a failure to retry than a blank.
            raise LLMError(f"the model returned no text ({reply.usage.output_tokens} tokens used)")
        doc_ids = [block.doc_id for block in blocks]
        cited, unresolved = resolve(cited_numbers(reply.text), doc_ids)
        return GeneratedAnswer(
            text=strip_markers(reply.text),
            raw_text=reply.text,
            cited_doc_ids=cited,
            retrieved_doc_ids=list(dict.fromkeys(doc_ids)),
            unresolved_markers=unresolved,
            usage=reply.usage,
            model=reply.model,
            blocks=tuple(blocks),
            mode=mode,
            retrieval_timings_ms=result.timings_ms,
            generation_ms=generation_ms,
        )


def _declined(
    abstention: Abstention,
    hits: Sequence[ChunkHit],
    mode: RetrievalMode,
    retrieval_timings_ms: dict[str, float],
) -> GeneratedAnswer:
    """The answer of a question whose evidence was too weak: no model was called, so no tokens."""
    return GeneratedAnswer(
        text=abstention.message,
        raw_text=abstention.message,
        cited_doc_ids=[],
        retrieved_doc_ids=list(dict.fromkeys(hit.doc_id for hit in hits)),
        unresolved_markers=[],
        usage=LLMUsage(input_tokens=0, output_tokens=0),
        model="",
        mode=mode,
        retrieval_timings_ms=retrieval_timings_ms,
        abstention=abstention,
    )
