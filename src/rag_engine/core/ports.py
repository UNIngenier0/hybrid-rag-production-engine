"""Interfaces that the rest of the code depends on. Adapters implement them."""

from collections.abc import Mapping, Sequence
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from rag_engine.core.models import (
    ChunkHit,
    ChunkPoint,
    DocumentProgress,
    LLMReply,
    PassageSpan,
    RateDecision,
    RetrievalMode,
    RetrievalResult,
    SearchFilters,
    SequencedDocument,
    SparseVector,
)

# The two passes that make a document searchable: BM25 first (fast, CPU), dense vectors second.
IndexStage = Literal["sparse", "dense"]
# (index, total): a run handles the documents whose `seq % total == index`.
Shard = tuple[int, int]


class HealthProbe(Protocol):
    """Checks whether one external dependency is ready to serve requests."""

    name: str

    async def check(self) -> None:
        """Return normally when the dependency is ready, raise any exception when it is not."""
        ...


class Embedder(Protocol):
    """Turns text into dense vectors. Blocking: callers run it in a thread."""

    model_name: str
    dimension: int

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Embed passages into a (len(texts), dimension) array of L2-normalized vectors."""
        ...

    def embed_query(self, text: str) -> NDArray[np.float32]:
        """Embed a search query, applying any query-side instruction the model expects."""
        ...


class Reranker(Protocol):
    """Scores how well passages answer a query, more accurately than similarity search can.

    Blocking: callers run it in a thread.
    """

    model_name: str

    def score(self, query: str, passages: Sequence[str]) -> NDArray[np.float32]:
        """One relevance score per passage; higher is more relevant."""
        ...


class SparseEncoder(Protocol):
    """Turns text into BM25 sparse vectors. Term weights only: IDF is applied by the index."""

    model_name: str

    def encode_documents(self, texts: Sequence[str]) -> list[SparseVector]: ...

    def encode_query(self, text: str) -> SparseVector: ...


class DocumentStore(Protocol):
    """Holds the cleaned documents and the per-document ingestion progress."""

    async def ensure_schema(self) -> None: ...

    async def last_staged_seq(self) -> int:
        """Highest staged `seq`, or -1 when nothing is staged."""
        ...

    async def add_documents(self, documents: Sequence[SequencedDocument]) -> int:
        """Insert documents, ignoring ones already stored. Returns how many were new."""
        ...

    async def pending(
        self, stage: IndexStage, *, after_seq: int, limit: int, shard: Shard | None = None
    ) -> list[SequencedDocument]:
        """Documents that still need `stage`, in `seq` order, starting after `after_seq`."""
        ...

    async def mark_indexed(self, stage: IndexStage, chunk_counts: Mapping[int, int]) -> None:
        """Record that `stage` finished for these documents (`seq` -> number of chunks)."""
        ...

    async def chunk_counts(self, seq_start: int, seq_end: int) -> dict[int, int]:
        """Chunks per document for documents with `seq_start <= seq < seq_end` that the sparse
        pass has indexed."""
        ...

    async def progress(self) -> DocumentProgress: ...


class ChunkIndex(Protocol):
    """The searchable index of chunks."""

    async def ensure_ready(self, dimension: int) -> None:
        """Create the collection and its payload indexes if they do not exist."""
        ...

    async def upsert_sparse(self, points: Sequence[ChunkPoint]) -> None:
        """Write chunks with their BM25 vector. Existing points with the same ID are replaced."""
        ...

    async def add_dense(self, vectors: Mapping[str, Sequence[float]]) -> None:
        """Attach dense vectors (point ID -> vector) to points that already exist."""
        ...

    async def count(self) -> int: ...

    async def count_missing_dense(self) -> int:
        """Points that have no dense vector yet."""
        ...

    async def wait_until_idle(self, timeout_seconds: float) -> None:
        """Return once the index has absorbed pending writes, or after the timeout.

        Bulk loading faster than the index can optimize makes its memory grow until the process
        is killed, so loaders pause here between large batches.
        """
        ...


class ChunkSearcher(Protocol):
    """Nearest-neighbour search over the indexed chunks. Hits come best first."""

    async def search_dense(
        self, vector: Sequence[float], *, limit: int, filters: SearchFilters | None = None
    ) -> list[ChunkHit]: ...

    async def search_sparse(
        self, vector: SparseVector, *, limit: int, filters: SearchFilters | None = None
    ) -> list[ChunkHit]: ...


class PassageStore(Protocol):
    """Reads back stretches of stored documents, used as context around a retrieved chunk."""

    async def passages(self, spans: Sequence[PassageSpan]) -> dict[PassageSpan, str]:
        """Text of each span. A span whose document is not stored is left out."""
        ...


class LLMClient(Protocol):
    """A language model that answers one prompt. Vendor SDKs stay behind adapters."""

    model: str

    async def complete(
        self, system: str, user: str, *, max_tokens: int = 1024, temperature: float | None = None
    ) -> LLMReply: ...


class Retriever(Protocol):
    """Finds the chunks that answer a query. The query pipeline depends only on this."""

    async def retrieve(
        self,
        query: str,
        *,
        mode: RetrievalMode = "hybrid",
        filters: SearchFilters | None = None,
        limit: int | None = None,
        include_context: bool = False,
    ) -> RetrievalResult: ...


class RateLimiter(Protocol):
    """Counts requests per key in a time window, shared by every API process."""

    async def hit(self, key: str, limit: int, window_seconds: int) -> RateDecision:
        """Count one request against `key`. Raises when the counter cannot be reached."""
        ...


class ResponseCache(Protocol):
    """Holds serialised responses for a while, shared by every API process."""

    async def get(self, key: str) -> str | None: ...

    async def set(self, key: str, value: str, ttl_seconds: int) -> None: ...
