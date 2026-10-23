"""In-memory stand-ins for the document store, the index and the encoders.

They follow the semantics of the real adapters (what counts as pending, what a replaced point
loses) so pipeline tests exercise real behavior without any service.
"""

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray

from rag_engine.core.models import (
    ChunkHit,
    ChunkPoint,
    DocumentProgress,
    LLMReply,
    LLMUsage,
    PassageSpan,
    RateDecision,
    RetrievalMode,
    RetrievalResult,
    SearchFilters,
    SequencedDocument,
    SourceDocument,
    SparseVector,
)
from rag_engine.core.ports import IndexStage, Shard

DIMENSION = 8


class InMemoryDocumentStore:
    def __init__(self) -> None:
        self.documents: dict[int, SourceDocument] = {}
        self.n_chunks: dict[int, int] = {}
        self.sparse_done: set[int] = set()
        self.dense_done: set[int] = set()

    async def ensure_schema(self) -> None:
        return None

    async def last_staged_seq(self) -> int:
        return max(self.documents, default=-1)

    async def add_documents(self, documents: Sequence[SequencedDocument]) -> int:
        added = 0
        seen = {(d.doc_id, d.content_sha256) for d in self.documents.values()}
        for item in documents:
            key = (item.document.doc_id, item.document.content_sha256)
            if item.seq in self.documents or key in seen:
                continue
            self.documents[item.seq] = item.document
            seen.add(key)
            added += 1
        return added

    async def pending(
        self, stage: IndexStage, *, after_seq: int, limit: int, shard: Shard | None = None
    ) -> list[SequencedDocument]:
        index, total = shard or (0, 1)
        wanted = []
        for seq in sorted(self.documents):
            if seq <= after_seq or seq % total != index:
                continue
            if stage == "sparse" and seq not in self.sparse_done:
                wanted.append(seq)
            if stage == "dense" and seq in self.sparse_done and seq not in self.dense_done:
                wanted.append(seq)
        return [SequencedDocument(seq, self.documents[seq]) for seq in wanted[:limit]]

    async def mark_indexed(self, stage: IndexStage, chunk_counts: Mapping[int, int]) -> None:
        for seq, count in chunk_counts.items():
            if stage == "sparse":
                self.n_chunks[seq] = count
                self.sparse_done.add(seq)
                self.dense_done.discard(seq)
            elif seq in self.sparse_done:
                self.dense_done.add(seq)

    async def chunk_counts(self, seq_start: int, seq_end: int) -> dict[int, int]:
        return {seq: self.n_chunks[seq] for seq in self.sparse_done if seq_start <= seq < seq_end}

    async def progress(self) -> DocumentProgress:
        return DocumentProgress(
            staged=len(self.documents),
            sparse_indexed=len(self.sparse_done),
            dense_indexed=len(self.dense_done),
            chunks=sum(self.n_chunks[seq] for seq in self.sparse_done),
        )


class InMemoryChunkIndex:
    def __init__(self) -> None:
        self.points: dict[str, dict[str, object]] = {}
        self.sparse: dict[str, SparseVector] = {}
        self.dense: dict[str, list[float]] = {}
        self.ready_dimension: int | None = None
        self.idle_waits = 0

    async def ensure_ready(self, dimension: int) -> None:
        self.ready_dimension = dimension

    async def upsert_sparse(self, points: Sequence[ChunkPoint]) -> None:
        for point in points:
            # Replacing a point drops its dense vector, as in Qdrant.
            self.dense.pop(point.point_id, None)
            self.points[point.point_id] = point.payload
            self.sparse[point.point_id] = point.sparse

    async def add_dense(self, vectors: Mapping[str, Sequence[float]]) -> None:
        missing = [point_id for point_id in vectors if point_id not in self.points]
        if missing:
            raise KeyError(f"no such points: {missing[:3]}")
        for point_id, vector in vectors.items():
            self.dense[point_id] = list(vector)

    async def count(self) -> int:
        return len(self.points)

    async def count_missing_dense(self) -> int:
        return len(self.points) - len(self.dense)

    async def wait_until_idle(self, timeout_seconds: float) -> None:
        self.idle_waits += 1


class FakeSparseEncoder:
    model_name = "fake-bm25"

    def encode_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        return [self.encode_query(text) for text in texts]

    def encode_query(self, text: str) -> SparseVector:
        terms = sorted({hash(word) % 1000 for word in text.lower().split()})
        return SparseVector(indices=terms, values=[1.0] * len(terms))


class FakeEmbedder:
    """Deterministic unit vectors derived from the text, so equal texts embed equally."""

    model_name = "fake-embedder"
    dimension = DIMENSION

    def __init__(self) -> None:
        self.calls: list[int] = []

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        self.calls.append(len(texts))
        return (
            np.stack([self._vector(text) for text in texts])
            if texts
            else np.empty((0, DIMENSION), np.float32)
        )

    def embed_query(self, text: str) -> NDArray[np.float32]:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> NDArray[np.float32]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vector = np.frombuffer(digest[: DIMENSION * 4], dtype=np.uint32).astype(np.float64)
        vector = vector / np.linalg.norm(vector)
        return vector.astype(np.float32)


def make_document(
    doc_id: str, content: str, *, source_type: str = "confluence", title: str = "Title"
) -> SourceDocument:
    from rag_engine.ingestion.cleaning import normalize_document

    return normalize_document(doc_id, source_type, title, content)[0]


def long_text(paragraphs: int = 12, words: int = 60, seed: str = "p") -> str:
    """A multi-paragraph document of a few thousand characters."""
    return "\n\n".join(
        " ".join(f"{seed}{p}w{w}" for w in range(words)) + "." for p in range(paragraphs)
    )


def write_corpus(path: Path, rows: Sequence[tuple[str, str, str, str]]) -> Path:
    """Write (doc_id, source_type, title, content) rows to a Parquet file."""
    columns = list(zip(*rows, strict=True)) if rows else [[], [], [], []]
    table = pa.table(
        {
            "doc_id": pa.array(columns[0], pa.string()),
            "source_type": pa.array(columns[1], pa.string()),
            "title": pa.array(columns[2], pa.string()),
            "content": pa.array(columns[3], pa.string()),
        }
    )
    pq.write_table(table, path)
    return path


def make_hit(
    point_id: str, *, doc_id: str | None = None, score: float = 1.0, **fields: object
) -> ChunkHit:
    """A retrieved chunk with plausible defaults; keyword arguments override any field."""
    values: dict[str, object] = {
        "point_id": point_id,
        "doc_id": doc_id or f"doc-{point_id}",
        "content_sha256": "sha",
        "source_type": "confluence",
        "title": "Title",
        "section": None,
        "chunk_index": 0,
        "text": f"text of {point_id}",
        "char_start": 0,
        "char_end": 10,
        "parent_start": 0,
        "parent_end": 40,
        "score": score,
    }
    values.update(fields)
    return ChunkHit(**values)  # type: ignore[arg-type]


class FakeSearcher:
    """Returns the lists it was given and records how it was called."""

    def __init__(
        self, dense: list[ChunkHit] | None = None, sparse: list[ChunkHit] | None = None
    ) -> None:
        self.dense = dense or []
        self.sparse = sparse or []
        self.calls: list[tuple[str, int, SearchFilters | None]] = []
        self.dense_vectors: list[list[float]] = []
        self.sparse_vectors: list[SparseVector] = []

    async def search_dense(
        self, vector: Sequence[float], *, limit: int, filters: SearchFilters | None = None
    ) -> list[ChunkHit]:
        self.calls.append(("dense", limit, filters))
        self.dense_vectors.append(list(vector))
        return self.dense[:limit]

    async def search_sparse(
        self, vector: SparseVector, *, limit: int, filters: SearchFilters | None = None
    ) -> list[ChunkHit]:
        self.calls.append(("sparse", limit, filters))
        self.sparse_vectors.append(vector)
        return self.sparse[:limit]


class FakePassageStore:
    """Serves a span as a string that names it, unless the document is listed as missing."""

    def __init__(self, missing_docs: frozenset[str] = frozenset()) -> None:
        self.missing_docs = missing_docs
        self.requests: list[list[PassageSpan]] = []

    async def passages(self, spans: Sequence[PassageSpan]) -> dict[PassageSpan, str]:
        self.requests.append(list(spans))
        return {
            span: f"{span.doc_id}[{span.start}:{span.end}]"
            for span in spans
            if span.doc_id not in self.missing_docs
        }


class FakeReranker:
    """Scores a passage by the first key of `scores` it contains; unknown passages score 0."""

    model_name = "fake-reranker"

    def __init__(self, scores: Mapping[str, float]) -> None:
        self.scores = scores
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, passages: Sequence[str]) -> NDArray[np.float32]:
        self.calls.append((query, list(passages)))
        return np.array(
            [next((v for k, v in self.scores.items() if k in p), 0.0) for p in passages],
            dtype=np.float32,
        )


class FakeLLM:
    """Replies with scripted texts, one per call (the last one repeats), and records the prompts."""

    def __init__(
        self, *replies: str, model: str = "claude-haiku-5-5", usage: LLMUsage | None = None
    ) -> None:
        self.model = model
        self.replies = list(replies) or ["An answer."]
        self.usage = usage or LLMUsage(input_tokens=1000, output_tokens=100)
        self.prompts: list[tuple[str, str]] = []
        self.temperatures: list[float | None] = []

    async def complete(
        self, system: str, user: str, *, max_tokens: int = 1024, temperature: float | None = None
    ) -> LLMReply:
        self.prompts.append((system, user))
        self.temperatures.append(temperature)
        text = self.replies[min(len(self.prompts), len(self.replies)) - 1]
        return LLMReply(text=text, usage=self.usage, model=self.model)


class FakeRetriever:
    """Returns the same hits for any query, as a retriever would, and records the calls."""

    def __init__(self, hits: list[ChunkHit], error: Exception | None = None) -> None:
        self.hits = hits
        self.error = error
        self.calls: list[dict[str, object]] = []

    async def retrieve(
        self,
        query: str,
        *,
        mode: RetrievalMode = "hybrid",
        filters: SearchFilters | None = None,
        limit: int | None = None,
        include_context: bool = False,
    ) -> RetrievalResult:
        if self.error is not None:
            raise self.error
        self.calls.append(
            {
                "query": query,
                "mode": mode,
                "limit": limit,
                "include_context": include_context,
                "filters": filters,
            }
        )
        return RetrievalResult(
            query=query,
            mode=mode,
            hits=self.hits[: limit or len(self.hits)],
            timings_ms={"total": 2.0},
        )


class FakeRateLimiter:
    """Counts hits per key like the Redis one (a fixed window that never ends) and records them."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.counts: dict[str, int] = {}
        self.calls: list[tuple[str, int, int]] = []

    async def hit(self, key: str, limit: int, window_seconds: int) -> RateDecision:
        if self.error is not None:
            raise self.error
        self.calls.append((key, limit, window_seconds))
        self.counts[key] = self.counts.get(key, 0) + 1
        count = self.counts[key]
        return RateDecision(
            allowed=count <= limit,
            limit=limit,
            remaining=max(0, limit - count),
            retry_after_seconds=window_seconds,
        )


class FakeCache:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.values: dict[str, str] = {}
        self.ttls: dict[str, int] = {}

    async def get(self, key: str) -> str | None:
        if self.error is not None:
            raise self.error
        return self.values.get(key)

    async def set(self, key: str, value: str, ttl_seconds: int) -> None:
        if self.error is not None:
            raise self.error
        self.values[key] = value
        self.ttls[key] = ttl_seconds
