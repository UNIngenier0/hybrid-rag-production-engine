from pathlib import Path

import pytest

from fakes import (
    DIMENSION,
    FakeEmbedder,
    FakeSparseEncoder,
    InMemoryChunkIndex,
    InMemoryDocumentStore,
    long_text,
    write_corpus,
)
from rag_engine.indexing.shards import pipeline_fingerprint
from rag_engine.ingestion.chunking import Chunker, ChunkingConfig
from rag_engine.pipelines.bulk_ingestion import SparseIndexer, stage_documents
from rag_engine.pipelines.offline_embedding import EmbeddingJob, run_worker
from rag_engine.pipelines.vector_loading import ExpectedPipeline, ShardMismatchError, load_vectors

EXPECTED = ExpectedPipeline(
    model="fake",
    revision="r",
    dimension=DIMENSION,
    fingerprint=pipeline_fingerprint(ChunkingConfig()),
)


class Setup:
    def __init__(self, tmp_path: Path) -> None:
        rows = [
            (f"dsid_{i}", "confluence", f"Page {i}", long_text(5, seed=f"d{i}x")) for i in range(7)
        ]
        self.parquet = write_corpus(tmp_path / "docs.parquet", rows)
        self.shards = tmp_path / "shards"
        self.store = InMemoryDocumentStore()
        self.index = InMemoryChunkIndex()

    async def index_locally(self) -> None:
        await stage_documents(self.store, self.parquet, batch_size=3)
        await SparseIndexer(
            self.store, self.index, Chunker(), FakeSparseEncoder(), batch_size=3
        ).run()

    def make_shards(self, **overrides: object) -> None:
        fields = {
            "parquet": self.parquet, "out_dir": self.shards, "model": "fake", "revision": "r",
            "dimension": DIMENSION, "docs_per_shard": 3,
        }  # fmt: skip
        run_worker(EmbeddingJob(**{**fields, **overrides}), FakeEmbedder(), [0, 1, 2])  # type: ignore[arg-type]


@pytest.fixture
async def ready(tmp_path: Path) -> Setup:
    setup = Setup(tmp_path)
    await setup.index_locally()
    setup.make_shards()
    return setup


async def test_vectors_land_on_the_chunks_they_were_computed_for(ready: Setup) -> None:
    report = await load_vectors(ready.store, ready.index, ready.shards, EXPECTED, batch_rows=4)

    assert report.shards_loaded == 3
    assert ready.index.idle_waits == 3  # the index gets to absorb each shard before the next
    assert report.vectors == len(ready.index.points)
    assert await ready.index.count_missing_dense() == 0
    assert (await ready.store.progress()).dense_indexed == 7


async def test_loading_twice_does_not_repeat_work(ready: Setup) -> None:
    await load_vectors(ready.store, ready.index, ready.shards, EXPECTED)

    again = await load_vectors(ready.store, ready.index, ready.shards, EXPECTED)

    assert (again.shards_loaded, again.shards_skipped) == (0, 3)


async def test_shards_from_another_model_are_refused(ready: Setup) -> None:
    other = ExpectedPipeline("another", "r", DIMENSION, EXPECTED.fingerprint)

    with pytest.raises(ShardMismatchError, match="model"):
        await load_vectors(ready.store, ready.index, ready.shards, other)

    assert await ready.index.count_missing_dense() == len(ready.index.points)


async def test_shards_made_with_other_chunking_are_refused(ready: Setup) -> None:
    other = ExpectedPipeline(
        "fake", "r", DIMENSION, pipeline_fingerprint(ChunkingConfig(target_chars=900))
    )

    with pytest.raises(ShardMismatchError, match="fingerprint"):
        await load_vectors(ready.store, ready.index, ready.shards, other)


async def test_the_sparse_pass_must_have_covered_the_documents(tmp_path: Path) -> None:
    setup = Setup(tmp_path)
    await stage_documents(setup.store, setup.parquet, batch_size=3)
    setup.make_shards()

    with pytest.raises(ShardMismatchError, match="index sparse"):
        await load_vectors(setup.store, setup.index, setup.shards, EXPECTED)


async def test_a_chunk_count_that_disagrees_with_the_local_chunker_is_refused(ready: Setup) -> None:
    ready.store.n_chunks[1] += 1

    with pytest.raises(ShardMismatchError, match="chunk counts"):
        await load_vectors(ready.store, ready.index, ready.shards, EXPECTED)

    assert not ready.index.dense


async def test_an_empty_folder_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        await load_vectors(InMemoryDocumentStore(), InMemoryChunkIndex(), tmp_path, EXPECTED)
