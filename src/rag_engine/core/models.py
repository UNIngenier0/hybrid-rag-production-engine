"""Domain models shared by ingestion, indexing and retrieval. Plain data, no infrastructure."""

from dataclasses import dataclass, field
from typing import Literal

# Which retriever answers a query. `hybrid` fuses the two others.
RetrievalMode = Literal["dense", "bm25", "hybrid"]


@dataclass(frozen=True, slots=True)
class SourceDocument:
    """A cleaned document ready to be chunked.

    `doc_id` is the identifier the benchmark uses for evaluation, so it is kept as is. It is not
    unique in the source data (a few IDs are reused by different documents), which is why the
    identity of a stored document is the pair (`doc_id`, `content_sha256`).
    """

    doc_id: str
    content_sha256: str
    source_type: str
    title: str
    content: str


@dataclass(frozen=True, slots=True)
class SequencedDocument:
    """A document with its position in the source file, which gives runs a stable order."""

    seq: int
    document: SourceDocument


@dataclass(frozen=True, slots=True)
class Chunk:
    """A child chunk: the unit that is embedded, indexed and retrieved.

    Spans are character offsets into `SourceDocument.content`. The parent span is the larger
    passage around the chunk, used as context for the LLM, so its text is never stored twice.
    """

    doc_id: str
    content_sha256: str
    index: int
    text: str
    char_start: int
    char_end: int
    parent_start: int
    parent_end: int
    section: str | None = None


@dataclass(frozen=True, slots=True)
class SparseVector:
    """Sparse vector as parallel lists of term indices and weights."""

    indices: list[int]
    values: list[float]


@dataclass(frozen=True, slots=True)
class ChunkPoint:
    """A chunk ready to be written to the index: ID, searchable payload and BM25 vector."""

    point_id: str
    payload: dict[str, object]
    sparse: SparseVector


@dataclass(frozen=True, slots=True)
class DocumentProgress:
    """Counts of documents per ingestion stage."""

    staged: int
    sparse_indexed: int
    dense_indexed: int
    chunks: int


@dataclass(frozen=True, slots=True)
class SearchFilters:
    """Restrict a search to some sources or documents. An empty field does not restrict."""

    source_types: tuple[str, ...] = ()
    doc_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PassageSpan:
    """A stretch of one stored document, in characters of its cleaned content."""

    doc_id: str
    content_sha256: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class ChunkHit:
    """A retrieved chunk.

    `score` is only comparable within one result: cosine similarity for dense search, the BM25
    score for lexical search and the fused RRF score for hybrid search. The ranks (1 is best) say
    where each retriever put the chunk, and are `None` when it did not return it. `context` is
    the larger parent passage around the chunk, filled in only when it was asked for.
    """

    point_id: str
    doc_id: str
    content_sha256: str
    source_type: str
    title: str
    section: str | None
    chunk_index: int
    text: str
    char_start: int
    char_end: int
    parent_start: int
    parent_end: int
    score: float
    dense_rank: int | None = None
    bm25_rank: int | None = None
    context: str | None = None

    @property
    def parent_span(self) -> PassageSpan:
        return PassageSpan(self.doc_id, self.content_sha256, self.parent_start, self.parent_end)


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Ranked chunks for one query, with how long each stage took."""

    query: str
    mode: RetrievalMode
    hits: list[ChunkHit]
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def document_ids(self) -> list[str]:
        """Distinct documents in the order their best chunk appears."""
        return list(dict.fromkeys(hit.doc_id for hit in self.hits))
