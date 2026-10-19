"""Embed the corpus into vector shards. Meant to run where a GPU is, not on a laptop.

The job walks the Parquet file once per worker, chunks every document exactly as local indexing
does (same cleaning, chunker and contextual text, enforced by the pipeline fingerprint) and writes
one shard per block of `docs_per_shard` documents. A shard that already exists with matching
metadata is skipped, so a job that is stopped and started again continues where it left off.
"""

import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from rag_engine.core.models import SequencedDocument
from rag_engine.core.ports import Embedder
from rag_engine.indexing.shards import (
    ShardInfo,
    pipeline_fingerprint,
    read_shard_info,
    shard_filename,
    write_shard,
)
from rag_engine.ingestion.chunking import Chunker, ChunkingConfig
from rag_engine.ingestion.parquet_source import iter_document_batches
from rag_engine.pipelines.preparation import prepare_batch

logger = logging.getLogger(__name__)

# Texts handed to the model at once. Bounds memory and sets the pace of the progress log.
_EMBED_BLOCK = 2048


@dataclass(frozen=True, slots=True)
class EmbeddingJob:
    parquet: Path
    out_dir: Path
    model: str
    revision: str | None
    dimension: int
    docs_per_shard: int = 5000
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    # Stop after this many documents (smoke tests); None processes everything.
    limit_docs: int | None = None

    def expected_info(self, shard_index: int, documents: int, chunks: int) -> ShardInfo:
        seq_start = shard_index * self.docs_per_shard
        return ShardInfo(
            model=self.model,
            revision=self.revision,
            dimension=self.dimension,
            fingerprint=pipeline_fingerprint(self.chunking),
            seq_start=seq_start,
            seq_end=seq_start + documents,
            documents=documents,
            chunks=chunks,
        )


@dataclass(frozen=True, slots=True)
class ShardReport:
    shard: int
    documents: int
    chunks: int
    characters: int
    seconds: float
    skipped: bool = False


def shards_in_range(job: EmbeddingJob, total_documents: int) -> int:
    """Number of shards needed to cover the documents this job handles."""
    documents = total_documents if job.limit_docs is None else min(total_documents, job.limit_docs)
    return -(-documents // job.docs_per_shard)


def run_worker(
    job: EmbeddingJob, embedder: Embedder, shard_indexes: Iterable[int]
) -> list[ShardReport]:
    """Produce the given shards, which must be consecutive, in order."""
    indexes = list(shard_indexes)
    if not indexes:
        return []
    if indexes != list(range(indexes[0], indexes[-1] + 1)):
        raise ValueError("a worker handles consecutive shards")
    job.out_dir.mkdir(parents=True, exist_ok=True)
    chunker = Chunker(job.chunking)
    reports: list[ShardReport] = []
    first_seq = indexes[0] * job.docs_per_shard
    batches = iter_document_batches(job.parquet, batch_size=job.docs_per_shard, start_seq=first_seq)
    for shard_index, batch in zip(indexes, batches, strict=False):
        if job.limit_docs is not None:
            batch = batch[: max(job.limit_docs - batch[0].seq, 0)]
            if not batch:
                break
        path = job.out_dir / shard_filename(shard_index)
        if _is_complete(job, path, shard_index, len(batch)):
            reports.append(ShardReport(shard_index, len(batch), 0, 0, 0.0, skipped=True))
            continue
        reports.append(_embed_shard(job, embedder, chunker, path, shard_index, batch))
    return reports


def _is_complete(job: EmbeddingJob, path: Path, shard_index: int, documents: int) -> bool:
    if not path.exists():
        return False
    try:
        existing = read_shard_info(path)
    except Exception:
        return False
    expected = job.expected_info(shard_index, documents, existing.chunks)
    return existing == expected


def _embed_shard(
    job: EmbeddingJob,
    embedder: Embedder,
    chunker: Chunker,
    path: Path,
    shard_index: int,
    batch: list[SequencedDocument],
) -> ShardReport:
    started = time.perf_counter()
    prepared = prepare_batch(chunker, batch)
    texts = prepared.texts
    vectors = np.empty((len(texts), job.dimension), dtype=np.float16)
    for start in range(0, len(texts), _EMBED_BLOCK):
        block = embedder.embed_documents(texts[start : start + _EMBED_BLOCK])
        vectors[start : start + len(block)] = block.astype(np.float16)
    info = job.expected_info(shard_index, len(batch), len(texts))
    write_shard(path, info, prepared.seqs, prepared.chunk_indexes, prepared.point_ids, vectors)
    seconds = time.perf_counter() - started
    characters = sum(len(text) for text in texts)
    logger.info(
        "shard written",
        extra={
            "shard": shard_index,
            "documents": len(batch),
            "chunks": len(texts),
            "seconds": round(seconds, 1),
            "chunks_per_second": round(len(texts) / seconds, 1) if seconds else None,
        },
    )
    return ShardReport(shard_index, len(batch), len(texts), characters, seconds)
