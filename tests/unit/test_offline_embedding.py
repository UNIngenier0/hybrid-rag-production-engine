import json
from pathlib import Path

import pytest

from fakes import DIMENSION, FakeEmbedder, long_text, write_corpus
from rag_engine.indexing.shards import iter_shard_rows, read_shard_info, shard_filename
from rag_engine.pipelines.offline_embedding import EmbeddingJob, run_worker, shards_in_range
from rag_engine.workers.embed_corpus import _split


def corpus(tmp_path: Path, documents: int = 7) -> Path:
    rows = [
        (f"dsid_{i}", "confluence", f"Page {i}", long_text(paragraphs=6, seed=f"d{i}x"))
        for i in range(documents)
    ]
    return write_corpus(tmp_path / "docs.parquet", rows)


def job(tmp_path: Path, **overrides: object) -> EmbeddingJob:
    fields = {
        "parquet": corpus(tmp_path), "out_dir": tmp_path / "out", "model": "fake", "revision": "r",
        "dimension": DIMENSION, "docs_per_shard": 3,
    }  # fmt: skip
    return EmbeddingJob(**{**fields, **overrides})  # type: ignore[arg-type]


def test_every_document_range_becomes_one_shard(tmp_path: Path) -> None:
    embedding_job = job(tmp_path)
    assert shards_in_range(embedding_job, 7) == 3

    reports = run_worker(embedding_job, FakeEmbedder(), [0, 1, 2])

    assert [r.documents for r in reports] == [3, 3, 1]
    infos = [read_shard_info(embedding_job.out_dir / shard_filename(i)) for i in range(3)]
    assert [(i.seq_start, i.seq_end) for i in infos] == [(0, 3), (3, 6), (6, 7)]
    assert all(i.dimension == DIMENSION and i.model == "fake" for i in infos)


def test_each_row_carries_the_document_chunk_and_a_unit_vector(tmp_path: Path) -> None:
    embedding_job = job(tmp_path)
    run_worker(embedding_job, FakeEmbedder(), [0])

    [rows] = list(iter_shard_rows(embedding_job.out_dir / shard_filename(0), batch_rows=10_000))

    assert set(rows.seqs.tolist()) == {0, 1, 2}
    assert len(set(rows.point_ids)) == len(rows.point_ids)
    norms = (rows.vectors.astype("float32") ** 2).sum(axis=1) ** 0.5
    assert abs(norms - 1).max() < 5e-3


def test_a_finished_shard_is_skipped_on_the_next_run(tmp_path: Path) -> None:
    embedding_job = job(tmp_path)
    first = FakeEmbedder()
    run_worker(embedding_job, first, [0, 1, 2])
    second = FakeEmbedder()

    reports = run_worker(embedding_job, second, [0, 1, 2])

    assert all(report.skipped for report in reports)
    assert second.calls == []


def test_a_shard_from_another_model_is_redone(tmp_path: Path) -> None:
    run_worker(job(tmp_path), FakeEmbedder(), [0])

    reports = run_worker(job(tmp_path, model="another"), FakeEmbedder(), [0])

    assert not reports[0].skipped
    assert read_shard_info(tmp_path / "out" / shard_filename(0)).model == "another"


def test_resuming_in_the_middle_only_does_the_missing_shards(tmp_path: Path) -> None:
    embedding_job = job(tmp_path)
    run_worker(embedding_job, FakeEmbedder(), [0])
    embedder = FakeEmbedder()

    reports = run_worker(embedding_job, embedder, [0, 1, 2])

    assert [r.skipped for r in reports] == [True, False, False]


def test_a_document_limit_truncates_the_last_shard(tmp_path: Path) -> None:
    embedding_job = job(tmp_path, limit_docs=4)
    assert shards_in_range(embedding_job, 7) == 2

    reports = run_worker(embedding_job, FakeEmbedder(), [0, 1])

    assert [r.documents for r in reports] == [3, 1]


def test_a_worker_needs_consecutive_shards(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="consecutive"):
        run_worker(job(tmp_path), FakeEmbedder(), [0, 2])


def test_shards_are_split_into_consecutive_balanced_ranges() -> None:
    assert _split(list(range(7)), 2) == [[0, 1, 2, 3], [4, 5, 6]]
    assert _split([0], 2) == [[0], []]


def test_run_info_is_json_serializable(tmp_path: Path) -> None:
    from rag_engine.workers.embed_corpus import _write_run_info

    embedding_job = job(tmp_path)
    reports = run_worker(embedding_job, FakeEmbedder(), [0])

    _write_run_info(embedding_job, 0, "cpu", reports, 1.5)

    info = json.loads((embedding_job.out_dir / "run-info-0.json").read_text())
    assert info["shards_written"] == 1 and info["chunks"] > 0


def test_the_job_can_be_sent_to_a_worker_process(tmp_path: Path) -> None:
    import pickle

    embedding_job = job(tmp_path)

    assert pickle.loads(pickle.dumps(embedding_job)) == embedding_job


def test_several_devices_mean_several_workers() -> None:
    from rag_engine.workers.embed_corpus import _devices

    assert _devices("cuda:0,cuda:1") == ["cuda:0", "cuda:1"]
    assert _devices("cpu") == ["cpu"]
