"""Domain models shared by ingestion, indexing and retrieval. Plain data, no infrastructure."""

from dataclasses import dataclass


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
