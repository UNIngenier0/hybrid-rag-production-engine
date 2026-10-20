"""The prompt that turns retrieved passages into a cited answer."""

from collections.abc import Sequence
from dataclasses import dataclass

from rag_engine.core.models import ChunkHit

SYSTEM_PROMPT = """\
You answer questions about a company's internal documents: Slack threads, email, tickets, wiki \
pages, code hosting, CRM records and meeting transcripts. Use only the numbered passages you are \
given; each starts with its number in square brackets.

- Answer the question directly and completely. Give every specific value it asks for (numbers, \
names, identifiers, dates, thresholds) exactly as the passages state them.
- Write plain prose in a few sentences, without headings, bold text or tables. Use a short list \
only when the question asks for several items.
- Cite the passages that support each statement with their numbers in square brackets, placed \
at the end of the sentence, like "The limit is 10 MiB [2]." or "Both agree [1][3]." Never use a \
marker as a word in a sentence, never write "passage 2 says" in the text, and cite only \
passages that really contain what you state.
- When passages disagree, or a later or final one replaces an earlier proposal, give the current \
value and say what it replaced.
- If the passages do not contain the answer, say that you could not find it in the documents. Do \
not guess and do not use outside knowledge.
- Be concise: no preamble and no restating of the question.\
"""


@dataclass(frozen=True, slots=True)
class ContextBlock:
    """One numbered passage the model sees, and the document it comes from."""

    number: int
    doc_id: str
    title: str
    source_type: str
    text: str


def build_context(hits: Sequence[ChunkHit]) -> list[ContextBlock]:
    """Number the passages in rank order. A chunk is shown with its parent passage when it has
    one, and chunks that share a parent passage are shown once."""
    blocks: list[ContextBlock] = []
    seen: set[tuple[str, int, int]] = set()
    for hit in hits:
        key = (hit.doc_id, hit.parent_start, hit.parent_end)
        if hit.context is not None and key in seen:
            continue
        seen.add(key)
        blocks.append(
            ContextBlock(
                number=len(blocks) + 1,
                doc_id=hit.doc_id,
                title=hit.title,
                source_type=hit.source_type,
                text=hit.context if hit.context is not None else hit.text,
            )
        )
    return blocks


def render_prompt(question: str, blocks: Sequence[ContextBlock]) -> str:
    passages = "\n\n".join(
        f"[{block.number}] {block.title} ({block.source_type})\n{block.text}" for block in blocks
    )
    return f"Passages:\n\n{passages}\n\nQuestion: {question}"
