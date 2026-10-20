"""Answer a question from retrieved passages and report which documents the answer rests on."""

from dataclasses import dataclass

from rag_engine.core.errors import LLMError
from rag_engine.core.models import LLMUsage, RetrievalMode
from rag_engine.core.ports import LLMClient, Retriever
from rag_engine.generation.citations import cited_numbers, resolve, strip_markers
from rag_engine.generation.prompt import SYSTEM_PROMPT, build_context, render_prompt


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    text: str  # the answer without its [n] markers
    raw_text: str  # exactly what the model wrote
    cited_doc_ids: list[str]  # documents behind the markers that point at a passage
    retrieved_doc_ids: list[str]  # documents of every passage the model was shown
    unresolved_markers: list[int]  # markers that point at no passage
    usage: LLMUsage
    model: str


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
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._mode: RetrievalMode = mode
        self._chunks = chunks
        self._max_tokens = max_tokens
        self._temperature = temperature

    async def answer(self, question: str) -> GeneratedAnswer:
        result = await self._retriever.retrieve(
            question, mode=self._mode, limit=self._chunks, include_context=True
        )
        blocks = build_context(result.hits)
        reply = await self._llm.complete(
            SYSTEM_PROMPT,
            render_prompt(question, blocks),
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
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
        )
