"""The single-attempt grep baseline: one search, one try, then the same answer step.

A language model reads the question and writes one regular expression (and, if it wants, which
sources to search). The expression is run once over the raw documents, the files with the most
matches are kept, and the part of each that holds the matches is handed to the same answer
generator as every other system. It is what a person with grep and one go would do, which is why
it is a fair floor: it has no ranking beyond match counts and no second chance.
"""

import asyncio
import json
import re
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_engine.core.models import ChunkHit, RetrievalMode, RetrievalResult, SearchFilters
from rag_engine.core.ports import LLMClient

SOURCES = (
    "confluence",
    "fireflies",
    "github",
    "gmail",
    "google_drive",
    "hubspot",
    "jira",
    "linear",
    "slack",
)

PLANNER_PROMPT = """\
You search a company's internal documents with a single grep command, and you get exactly one \
try. The documents are files grouped by source: confluence, fireflies, github, gmail, \
google_drive, hubspot, jira, linear and slack.

Write one case-insensitive extended regular expression that the documents which answer the \
question are most likely to contain, and optionally limit the search to some sources.
- Use the most distinctive words of the question: identifiers, product and feature names, \
people, numbers, error codes, settings. Join alternatives with |. Do not use the whole question.
- Make the pattern specific enough to match few documents and broad enough not to miss the right \
one.

Reply with JSON only, for example {"pattern": "mtu|clamp|retransmit", "sources": ["slack", \
"linear"]}. An empty list of sources searches everything.\
"""


@dataclass(frozen=True, slots=True)
class SearchPlan:
    pattern: str
    sources: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FileMatch:
    path: Path
    matches: int


def parse_plan(text: str) -> SearchPlan:
    """Read the model's JSON, tolerating a code fence or words around it."""
    found = re.search(r"\{.*\}", text, re.DOTALL)
    if not found:
        raise ValueError("no JSON object in the search plan")
    data = json.loads(found.group(0))
    pattern = str(data.get("pattern", "")).strip()
    if not pattern:
        raise ValueError("the search plan has no pattern")
    sources = tuple(s for s in data.get("sources") or [] if s in SOURCES)
    return SearchPlan(pattern, sources)


async def plan_search(llm: LLMClient, question: str) -> SearchPlan:
    reply = await llm.complete(PLANNER_PROMPT, f"Question: {question}", max_tokens=1000)
    return parse_plan(reply.text)


Searcher = Callable[[SearchPlan, Path], list[FileMatch]]

# A search that takes longer than this is abandoned: a pattern that matches most of the corpus
# (the company's own name, say) only produces a list nobody could rank.
SEARCH_TIMEOUT_SECONDS = 300


def search_corpus(plan: SearchPlan, corpus: Path) -> list[FileMatch]:
    """Run the plan once over the corpus; files come back with the most matches first."""
    roots = [corpus / source for source in plan.sources if (corpus / source).is_dir()] or [corpus]
    found = _ripgrep(plan.pattern, roots) if shutil.which("rg") else _scan(plan.pattern, roots)
    return sorted(found, key=lambda match: (-match.matches, str(match.path)))


def _ripgrep(pattern: str, roots: Sequence[Path]) -> list[FileMatch]:
    result = subprocess.run(
        ["rg", "--count", "--ignore-case", "--no-messages", "--regexp", pattern, *map(str, roots)],
        capture_output=True,
        text=True,
        timeout=SEARCH_TIMEOUT_SECONDS,
        check=False,
    )
    matches = []
    for line in result.stdout.splitlines():
        path, _, count = line.rpartition(":")
        if path and count.isdigit():
            matches.append(FileMatch(Path(path), int(count)))
    return matches


def _scan(pattern: str, roots: Sequence[Path]) -> list[FileMatch]:
    """The same search in Python, for machines without ripgrep (slow on the full corpus)."""
    expression = re.compile(pattern, re.IGNORECASE)
    matches = []
    for root in roots:
        for path in root.rglob("*.json"):
            count = len(expression.findall(path.read_text(encoding="utf-8", errors="ignore")))
            if count:
                matches.append(FileMatch(path, count))
    return matches


@dataclass(frozen=True, slots=True)
class Document:
    doc_id: str
    title: str
    source_type: str
    text: str


def load_document(path: Path, corpus: Path) -> Document:
    """A raw document: its id, its title and the fields that hold its content."""
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    title = str(data.get(data.get("title_field_name", ""), path.stem))
    fields = data.get("content_field_names") or []
    if isinstance(fields, str):  # some documents store the list as its text form
        fields = re.findall(r"'([^']+)'", fields)
    parts = []
    for field in fields:
        value = data.get(field, "")
        if isinstance(value, list):
            value = "\n".join(str(item) for item in value)
        parts.append(str(value) if len(fields) == 1 else f"{field}:\n{value}")
    source = path.relative_to(corpus).parts[0]
    return Document(str(data["dataset_doc_uuid"]), title, source, "\n\n".join(parts))


def densest_window(text: str, pattern: str, size: int = 3200) -> str:
    """The stretch of the text with the most matches, so the answer step sees where they cluster."""
    if len(text) <= size:
        return text
    try:
        starts = [m.start() for m in re.finditer(pattern, text, re.IGNORECASE)][:2000]
    except re.error:
        starts = []
    if not starts:
        return text[:size]
    best = max(starts, key=lambda s: sum(1 for t in starts if s <= t < s + size))
    begin = max(0, min(best - 200, len(text) - size))
    return text[begin : begin + size]


class GrepRetriever:
    """A `Retriever` that answers with the single-attempt grep search."""

    def __init__(
        self,
        llm: LLMClient,
        corpus: Path,
        *,
        files: int = 5,
        window: int = 3200,
        searcher: Searcher = search_corpus,
    ) -> None:
        self._llm = llm
        self._corpus = corpus
        self._files = files
        self._window = window
        self._searcher = searcher
        self.plans: dict[str, SearchPlan | None] = {}

    async def retrieve(
        self,
        query: str,
        *,
        mode: RetrievalMode = "hybrid",
        filters: SearchFilters | None = None,
        limit: int | None = None,
        include_context: bool = False,
    ) -> RetrievalResult:
        try:
            plan: SearchPlan | None = await plan_search(self._llm, query)
        except ValueError:
            plan = None  # the model gave no usable pattern: the single try is spent
        self.plans[query] = plan
        if plan is None:
            return RetrievalResult(query=query, mode=mode, hits=[])
        try:
            found = await asyncio.to_thread(self._searcher, plan, self._corpus)
        except subprocess.TimeoutExpired:
            # A pattern that matches half the corpus never finishes: the single try found nothing.
            found = []
        hits = [
            await asyncio.to_thread(self._hit, match, plan)
            for match in found[: min(self._files, limit or self._files)]
        ]
        return RetrievalResult(query=query, mode=mode, hits=hits)

    def _hit(self, match: FileMatch, plan: SearchPlan) -> ChunkHit:
        document = load_document(match.path, self._corpus)
        text = densest_window(document.text, plan.pattern, self._window)
        return ChunkHit(
            point_id=f"{document.doc_id}:grep",
            doc_id=document.doc_id,
            content_sha256="",
            source_type=document.source_type,
            title=document.title,
            section=None,
            chunk_index=0,
            text=text,
            char_start=0,
            char_end=len(text),
            parent_start=0,
            parent_end=len(text),
            score=float(match.matches),
        )
