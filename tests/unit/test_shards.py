from pathlib import Path

import numpy as np
import pytest

from rag_engine.indexing.ids import chunk_point_id
from rag_engine.indexing.shards import (
    ShardInfo,
    iter_shard_rows,
    pipeline_fingerprint,
    read_shard_info,
    shard_filename,
    write_shard,
)
from rag_engine.ingestion.chunking import ChunkingConfig


def info(rows: int = 3, dimension: int = 4) -> ShardInfo:
    return ShardInfo(
        model="m", revision="r1", dimension=dimension, fingerprint="f", seq_start=10, seq_end=12,
        documents=2, chunks=rows,
    )  # fmt: skip


def test_point_ids_are_stable_and_depend_on_every_part() -> None:
    base = chunk_point_id("dsid_1", "abc", 0)

    assert base == chunk_point_id("dsid_1", "abc", 0)
    assert len({base, chunk_point_id("dsid_2", "abc", 0), chunk_point_id("dsid_1", "abd", 0),
                chunk_point_id("dsid_1", "abc", 1)}) == 4  # fmt: skip


def test_shard_names_sort_in_document_order() -> None:
    assert shard_filename(7) == "vectors-00007.parquet"
    assert sorted([shard_filename(10), shard_filename(2)]) == [
        shard_filename(2),
        shard_filename(10),
    ]


def test_a_shard_round_trips_its_rows_and_describes_itself(tmp_path: Path) -> None:
    path = tmp_path / shard_filename(0)
    vectors = np.arange(12, dtype=np.float16).reshape(3, 4) / 10
    write_shard(path, info(), [10, 10, 11], [0, 1, 0], ["a", "b", "c"], vectors)

    assert read_shard_info(path) == info()
    [rows] = list(iter_shard_rows(path))
    assert rows.seqs.tolist() == [10, 10, 11]
    assert rows.point_ids == ["a", "b", "c"]
    assert rows.vectors.dtype == np.float16
    np.testing.assert_array_equal(rows.vectors, vectors)


def test_rows_are_read_in_batches(tmp_path: Path) -> None:
    path = tmp_path / shard_filename(0)
    write_shard(
        path, info(), [10, 10, 11], [0, 1, 0], ["a", "b", "c"], np.zeros((3, 4), np.float16)
    )

    sizes = [len(rows.point_ids) for rows in iter_shard_rows(path, batch_rows=2)]

    assert sizes == [2, 1]


def test_writing_is_atomic_and_leaves_no_partial_file(tmp_path: Path) -> None:
    path = tmp_path / shard_filename(0)
    write_shard(
        path, info(), [10, 10, 11], [0, 1, 0], ["a", "b", "c"], np.zeros((3, 4), np.float16)
    )

    assert [p.name for p in tmp_path.iterdir()] == [path.name]


def test_inconsistent_data_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        write_shard(
            tmp_path / "x.parquet",
            info(rows=3),
            [1, 2],
            [0, 1],
            ["a", "b"],
            np.zeros((2, 4), np.float16),
        )
    with pytest.raises(ValueError):
        write_shard(
            tmp_path / "x.parquet",
            info(rows=2, dimension=5),
            [1, 2],
            [0, 1],
            ["a", "b"],
            np.zeros((2, 4), np.float16),
        )


def test_a_plain_parquet_file_is_not_a_shard(tmp_path: Path) -> None:
    from fakes import write_corpus

    path = write_corpus(tmp_path / "plain.parquet", [("d", "s", "t", "c")])

    with pytest.raises(ValueError, match="not a vector shard"):
        read_shard_info(path)


def test_the_fingerprint_changes_with_the_chunking_but_not_between_runs() -> None:
    base = pipeline_fingerprint(ChunkingConfig())

    assert base == pipeline_fingerprint(ChunkingConfig())
    assert base != pipeline_fingerprint(ChunkingConfig(target_chars=900))
