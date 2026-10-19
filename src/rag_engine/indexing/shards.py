"""Vector shards: the hand-off format between the GPU embedding job and the local index.

Embedding the corpus needs a GPU, so it runs elsewhere (a Kaggle notebook) and produces Parquet
shards. A shard covers a contiguous range of documents and holds one row per chunk:

    seq          row number of the document in the source file
    chunk_index  position of the chunk in its document
    point_id     deterministic ID of the chunk's index point
    vector       the dense embedding, half precision

Each shard describes itself in its Parquet metadata (model, revision, chunking fingerprint,
document range), so a loader can refuse shards that were not produced by the same pipeline as the
local index instead of attaching vectors to the wrong chunks.
"""

import hashlib
import json
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray

from rag_engine.indexing.contextual import CONTEXT_VERSION
from rag_engine.ingestion.chunking import CHUNKER_VERSION, ChunkingConfig
from rag_engine.ingestion.cleaning import CLEANING_VERSION

SHARD_FORMAT_VERSION = 1
_METADATA_KEY = b"rag_engine.shard"


def pipeline_fingerprint(chunking: ChunkingConfig) -> str:
    """Identify everything that decides which chunks exist and what text each one embeds.

    Two runs with the same fingerprint produce the same point IDs for the same documents.
    """
    description = {
        "cleaning": CLEANING_VERSION,
        "chunker": CHUNKER_VERSION,
        "context": CONTEXT_VERSION,
        "chunking": asdict(chunking),
    }
    encoded = json.dumps(description, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class ShardInfo:
    """What a shard says about itself."""

    model: str
    revision: str | None
    dimension: int
    fingerprint: str
    seq_start: int  # first document in the shard
    seq_end: int  # one past the last document in the shard
    documents: int
    chunks: int
    format_version: int = SHARD_FORMAT_VERSION


@dataclass(frozen=True, slots=True)
class ShardRows:
    """A slice of a shard's rows."""

    seqs: NDArray[np.int64]
    point_ids: list[str]
    vectors: NDArray[np.float16]


def shard_filename(index: int) -> str:
    return f"vectors-{index:05d}.parquet"


def write_shard(
    path: Path,
    info: ShardInfo,
    seqs: Sequence[int],
    chunk_indexes: Sequence[int],
    point_ids: Sequence[str],
    vectors: NDArray[np.float16],
) -> None:
    """Write a shard atomically: readers never see a half-written file."""
    rows = len(point_ids)
    if not (len(seqs) == len(chunk_indexes) == rows == vectors.shape[0]):
        raise ValueError("a shard needs the same number of seqs, chunk indexes, IDs and vectors")
    if vectors.shape[1] != info.dimension or info.chunks != rows:
        raise ValueError("shard info does not match the data")
    table = pa.table(
        {
            "seq": pa.array(seqs, type=pa.int64()),
            "chunk_index": pa.array(chunk_indexes, type=pa.int32()),
            "point_id": pa.array(point_ids, type=pa.string()),
            "vector": pa.FixedSizeListArray.from_arrays(
                pa.array(np.ascontiguousarray(vectors).reshape(-1), type=pa.float16()),
                info.dimension,
            ),
        }
    ).replace_schema_metadata({_METADATA_KEY: json.dumps(asdict(info)).encode("utf-8")})
    partial = path.with_suffix(".parquet.part")
    pq.write_table(table, partial, compression="zstd", compression_level=1)
    partial.replace(path)


def read_shard_info(path: Path) -> ShardInfo:
    metadata = pq.read_schema(path).metadata or {}
    if _METADATA_KEY not in metadata:
        raise ValueError(f"{path.name} is not a vector shard (no shard metadata)")
    return ShardInfo(**json.loads(metadata[_METADATA_KEY]))


def iter_shard_rows(path: Path, *, batch_rows: int = 1000) -> Iterator[ShardRows]:
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_rows):
        vector_column = batch.column("vector")
        dimension = vector_column.type.list_size
        flat = vector_column.values.to_numpy(zero_copy_only=False)
        yield ShardRows(
            seqs=batch.column("seq").to_numpy(zero_copy_only=False),
            point_ids=batch.column("point_id").to_pylist(),
            vectors=flat.reshape(-1, dimension),
        )
