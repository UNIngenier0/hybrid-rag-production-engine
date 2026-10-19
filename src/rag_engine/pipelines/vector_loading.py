"""Attach the vectors from GPU-computed shards to the chunks already in the local index.

The BM25 pass creates every chunk's point; this step adds the dense vector to it. Before touching
the index, each shard is checked against the local pipeline, because a vector attached to the
wrong chunk is worse than a missing one:

- the model, revision, dimension and pipeline fingerprint must match;
- the BM25 pass must already have covered the shard's documents;
- the number of chunks per document must equal what the local chunker produced.
"""

import asyncio
import logging
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from rag_engine.core.ports import ChunkIndex, DocumentStore
from rag_engine.indexing.shards import ShardInfo, iter_shard_rows, read_shard_info

logger = logging.getLogger(__name__)


class ShardMismatchError(Exception):
    """A shard does not belong to the local index."""


@dataclass(frozen=True, slots=True)
class ExpectedPipeline:
    model: str
    revision: str | None
    dimension: int
    fingerprint: str


@dataclass(slots=True)
class LoadReport:
    shards_loaded: int = 0
    shards_skipped: int = 0
    vectors: int = 0
    seconds: float = 0.0


async def load_vectors(
    store: DocumentStore,
    index: ChunkIndex,
    shard_dir: Path,
    expected: ExpectedPipeline,
    *,
    batch_rows: int = 1000,
    idle_timeout_seconds: float = 900.0,
) -> LoadReport:
    paths = await asyncio.to_thread(lambda: sorted(shard_dir.glob("vectors-*.parquet")))
    if not paths:
        raise FileNotFoundError(f"no vector shards in {shard_dir}")
    report = LoadReport()
    started = time.perf_counter()
    for path in paths:
        info = read_shard_info(path)
        _check_pipeline(path, info, expected)
        local_counts = await store.chunk_counts(info.seq_start, info.seq_end)
        if len(local_counts) != info.documents:
            raise ShardMismatchError(
                f"{path.name}: the BM25 pass has indexed {len(local_counts)} of its "
                f"{info.documents} documents; run `index sparse` first"
            )
        if await _already_loaded(store, info):
            report.shards_skipped += 1
            continue
        _check_chunk_counts(path, local_counts)
        for rows in iter_shard_rows(path, batch_rows=batch_rows):
            await index.add_dense(
                {
                    point_id: vector.astype(np.float32).tolist()
                    for point_id, vector in zip(rows.point_ids, rows.vectors, strict=True)
                }
            )
            report.vectors += len(rows.point_ids)
        await store.mark_indexed("dense", local_counts)
        report.shards_loaded += 1
        # Let the index absorb this shard before the next one, so its memory stays bounded.
        await index.wait_until_idle(idle_timeout_seconds)
        logger.info(
            "shard loaded",
            extra={"shard": path.name, "documents": info.documents, "chunks": info.chunks},
        )
    report.seconds = time.perf_counter() - started
    return report


def _check_pipeline(path: Path, info: ShardInfo, expected: ExpectedPipeline) -> None:
    mismatches = [
        f"{name}: shard has {actual!r}, local index expects {wanted!r}"
        for name, actual, wanted in (
            ("model", info.model, expected.model),
            ("revision", info.revision, expected.revision),
            ("dimension", info.dimension, expected.dimension),
            ("fingerprint", info.fingerprint, expected.fingerprint),
        )
        if actual != wanted
    ]
    if mismatches:
        raise ShardMismatchError(
            f"{path.name} was made by a different pipeline ({'; '.join(mismatches)})"
        )


def _check_chunk_counts(path: Path, local_counts: dict[int, int]) -> None:
    seqs = pq.read_table(path, columns=["seq"]).column("seq").to_pylist()
    shard_counts = Counter(seqs)
    wrong = [seq for seq, local in local_counts.items() if shard_counts.get(seq, 0) != local]
    if wrong:
        raise ShardMismatchError(
            f"{path.name}: chunk counts differ from the local chunker for {len(wrong)} documents "
            f"(first: seq {wrong[0]}); the cleaning or chunking code is not the same"
        )


async def _already_loaded(store: DocumentStore, info: ShardInfo) -> bool:
    pending = await store.pending("dense", after_seq=info.seq_start - 1, limit=1)
    return not pending or pending[0].seq >= info.seq_end
