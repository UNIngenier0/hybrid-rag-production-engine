"""Bulk ingestion of a document corpus: stage, then BM25 pass, then dense pass.

Each stage is resumable and idempotent:

- `stage_documents` copies cleaned documents from the Parquet file into PostgreSQL. A restarted
  run continues after the highest row already stored.
- `SparseIndexer` chunks documents and writes them to the index with their BM25 vector, so the
  corpus is searchable by keywords as soon as this pass ends.
- `DenseIndexer` embeds the same chunks and attaches the dense vectors. It is the slow pass
  (hours for the benchmark corpus), which is why it is separate and checkpointed.

A batch is recorded as done only after the index acknowledged it, and point IDs are
deterministic, so a crash costs at most one batch of repeated work and never duplicates data.
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from rag_engine.core.models import ChunkPoint, SequencedDocument
from rag_engine.core.ports import (
    ChunkIndex,
    DocumentStore,
    Embedder,
    IndexStage,
    Shard,
    SparseEncoder,
)
from rag_engine.ingestion.chunking import Chunker
from rag_engine.ingestion.parquet_source import CleaningStats, count_rows, iter_document_batches
from rag_engine.pipelines.preparation import PreparedBatch, prepare_batch

logger = logging.getLogger(__name__)

_LOG_EVERY_SECONDS = 30.0


@dataclass(slots=True)
class RunReport:
    """What one run did."""

    documents: int = 0
    chunks: int = 0
    seconds: float = 0.0

    @property
    def documents_per_second(self) -> float:
        return self.documents / self.seconds if self.seconds else 0.0


class _Progress:
    """Logs throughput and an ETA at a steady pace, however fast batches complete."""

    def __init__(self, task: str, total: int | None) -> None:
        self._task = task
        self._total = total
        self._started = time.perf_counter()
        self._last_log = self._started

    def update(self, report: RunReport) -> None:
        now = time.perf_counter()
        report.seconds = now - self._started
        if now - self._last_log >= _LOG_EVERY_SECONDS:
            self._last_log = now
            self._log(report)

    def finish(self, report: RunReport) -> None:
        report.seconds = time.perf_counter() - self._started
        self._log(report)

    def _log(self, report: RunReport) -> None:
        rate = report.documents_per_second
        eta = (self._total - report.documents) / rate if self._total and rate else None
        logger.info(
            "progress",
            extra={
                "task": self._task,
                "documents": report.documents,
                "chunks": report.chunks,
                "documents_per_second": round(rate, 1),
                "eta_minutes": None if eta is None else round(eta / 60, 1),
            },
        )


async def stage_documents(
    store: DocumentStore,
    path: Path,
    *,
    batch_size: int = 2000,
    limit: int | None = None,
    stats: CleaningStats | None = None,
) -> RunReport:
    """Copy the cleaned documents of `path` into the store, resuming where a previous run ended."""
    await store.ensure_schema()
    last_seq = await store.last_staged_seq()
    total = count_rows(path) - (last_seq + 1)
    if limit is not None:
        total = min(total, limit)
    report = RunReport()
    progress = _Progress("stage", total)
    # A small limit should not clean (and count) a whole batch it will not use.
    read_size = batch_size if limit is None else min(batch_size, limit)
    batches = iter_document_batches(path, batch_size=read_size, start_seq=last_seq + 1, stats=stats)
    read_next = asyncio.create_task(asyncio.to_thread(next, batches, None))
    try:
        while (batch := await read_next) is not None:
            if limit is not None:
                batch = batch[: limit - report.documents]
            # Read the next batch (decode + clean, CPU-bound) while this one is written.
            read_next = asyncio.create_task(asyncio.to_thread(next, batches, None))
            await store.add_documents(batch)
            report.documents += len(batch)
            progress.update(report)
            if limit is not None and report.documents >= limit:
                break
    finally:
        read_next.cancel()
    progress.finish(report)
    return report


Preparer = Callable[[list[SequencedDocument]], PreparedBatch]
Handler = Callable[[PreparedBatch], Awaitable[None]]


class SparseIndexer:
    """First pass: chunk every pending document and index it with its BM25 vector."""

    def __init__(
        self,
        store: DocumentStore,
        index: ChunkIndex,
        chunker: Chunker,
        encoder: SparseEncoder,
        *,
        batch_size: int = 200,
    ) -> None:
        self._store = store
        self._index = index
        self._chunker = chunker
        self._encoder = encoder
        self._batch_size = batch_size

    async def run(self, *, limit: int | None = None, shard: Shard | None = None) -> RunReport:
        return await _run_pass(
            "sparse",
            self._store,
            self._batch_size,
            limit,
            shard,
            handle=self._index_batch,
            prepare=lambda documents: prepare_batch(self._chunker, documents),
        )

    async def _index_batch(self, batch: PreparedBatch) -> None:
        vectors = await asyncio.to_thread(self._encoder.encode_documents, batch.texts)
        points = [
            ChunkPoint(point_id=point_id, payload=payload, sparse=vector)
            for point_id, payload, vector in zip(
                batch.point_ids, batch.payloads, vectors, strict=True
            )
        ]
        await self._index.upsert_sparse(points)


class DenseIndexer:
    """Second pass: embed the chunks of documents the first pass indexed and attach the vectors."""

    def __init__(
        self,
        store: DocumentStore,
        index: ChunkIndex,
        chunker: Chunker,
        embedder: Embedder,
        *,
        batch_size: int = 200,
    ) -> None:
        self._store = store
        self._index = index
        self._chunker = chunker
        self._embedder = embedder
        self._batch_size = batch_size

    async def run(self, *, limit: int | None = None, shard: Shard | None = None) -> RunReport:
        return await _run_pass(
            "dense",
            self._store,
            self._batch_size,
            limit,
            shard,
            handle=self._index_batch,
            prepare=lambda documents: prepare_batch(self._chunker, documents),
        )

    async def _index_batch(self, batch: PreparedBatch) -> None:
        vectors = await asyncio.to_thread(self._embedder.embed_documents, batch.texts)
        await self._index.add_dense(
            {
                point_id: vector.tolist()
                for point_id, vector in zip(batch.point_ids, vectors, strict=True)
            }
        )


async def _run_pass(
    stage: IndexStage,
    store: DocumentStore,
    batch_size: int,
    limit: int | None,
    shard: Shard | None,
    *,
    handle: Handler,
    prepare: Preparer,
) -> RunReport:
    """Drive one pass, fetching and chunking the next batch while the current one is encoded."""
    report = RunReport()
    progress = _Progress(stage, await _pending_total(store, stage, shard, limit))
    remaining = limit

    def next_size() -> int:
        return batch_size if remaining is None else min(batch_size, remaining)

    async def fetch(after_seq: int, size: int) -> tuple[int, PreparedBatch] | None:
        documents = await store.pending(stage, after_seq=after_seq, limit=size, shard=shard)
        if not documents:
            return None
        return len(documents), await asyncio.to_thread(prepare, documents)

    prefetched: asyncio.Task[tuple[int, PreparedBatch] | None] | None = asyncio.create_task(
        fetch(-1, next_size())
    )
    try:
        while prefetched is not None:
            fetched = await prefetched
            if fetched is None:
                break
            count, batch = fetched
            if remaining is not None:
                remaining -= count
            prefetched = (
                asyncio.create_task(fetch(batch.last_seq, next_size())) if remaining != 0 else None
            )
            await handle(batch)
            await store.mark_indexed(stage, batch.chunk_counts)
            report.documents += count
            report.chunks += len(batch.point_ids)
            progress.update(report)
    finally:
        if prefetched is not None:
            prefetched.cancel()
    progress.finish(report)
    return report


async def _pending_total(
    store: DocumentStore, stage: IndexStage, shard: Shard | None, limit: int | None
) -> int:
    """Documents this run is expected to handle, for the ETA."""
    done = await store.progress()
    total = (
        done.staged - done.sparse_indexed
        if stage == "sparse"
        else (done.sparse_indexed - done.dense_indexed)
    )
    if shard is not None:
        total //= shard[1]
    return total if limit is None else min(total, limit)
