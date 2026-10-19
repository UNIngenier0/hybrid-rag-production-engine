from pathlib import Path

from fakes import (
    DIMENSION,
    FakeEmbedder,
    FakeSparseEncoder,
    InMemoryChunkIndex,
    InMemoryDocumentStore,
    long_text,
    write_corpus,
)
from rag_engine.ingestion.chunking import Chunker
from rag_engine.ingestion.parquet_source import CleaningStats
from rag_engine.pipelines.bulk_ingestion import DenseIndexer, SparseIndexer, stage_documents


def corpus(tmp_path: Path, documents: int = 7) -> Path:
    rows = [
        (f"dsid_{i}", "confluence" if i % 2 else "slack", f"Page {i}", long_text(5, seed=f"d{i}x"))
        for i in range(documents)
    ]
    return write_corpus(tmp_path / "docs.parquet", rows)


async def staged(tmp_path: Path, documents: int = 7) -> InMemoryDocumentStore:
    store = InMemoryDocumentStore()
    await stage_documents(store, corpus(tmp_path, documents), batch_size=3)
    return store


def sparse_indexer(store: InMemoryDocumentStore, index: InMemoryChunkIndex) -> SparseIndexer:
    return SparseIndexer(store, index, Chunker(), FakeSparseEncoder(), batch_size=2)


async def test_staging_copies_every_document(tmp_path: Path) -> None:
    store = InMemoryDocumentStore()
    stats = CleaningStats()

    report = await stage_documents(store, corpus(tmp_path), batch_size=3, stats=stats)

    assert report.documents == 7
    assert sorted(store.documents) == list(range(7))
    assert stats.documents == 7


async def test_staging_resumes_after_the_last_stored_row(tmp_path: Path) -> None:
    store = InMemoryDocumentStore()
    path = corpus(tmp_path)
    first = await stage_documents(store, path, batch_size=3, limit=4)

    second = await stage_documents(store, path, batch_size=3)

    assert (first.documents, second.documents) == (4, 3)
    assert sorted(store.documents) == list(range(7))


async def test_the_sparse_pass_indexes_every_chunk_with_its_payload(tmp_path: Path) -> None:
    store, index = await staged(tmp_path), InMemoryChunkIndex()

    report = await sparse_indexer(store, index).run()

    assert report.documents == 7
    assert report.chunks == len(index.points) == sum(store.n_chunks.values())
    payload = next(iter(index.points.values()))
    assert {"doc_id", "content_sha256", "source_type", "title", "chunk_index", "text"} <= set(
        payload
    )
    assert (await store.progress()).sparse_indexed == 7


async def test_the_pass_is_idempotent(tmp_path: Path) -> None:
    store, index = await staged(tmp_path), InMemoryChunkIndex()
    await sparse_indexer(store, index).run()
    points = dict(index.points)

    again = await sparse_indexer(store, index).run()

    assert again.documents == 0
    assert index.points == points


async def test_a_limit_stops_the_pass_early_and_the_next_run_continues(tmp_path: Path) -> None:
    store, index = await staged(tmp_path), InMemoryChunkIndex()

    first = await sparse_indexer(store, index).run(limit=3)
    rest = await sparse_indexer(store, index).run()

    assert (first.documents, rest.documents) == (3, 4)


async def test_shards_partition_the_documents(tmp_path: Path) -> None:
    store, index = await staged(tmp_path), InMemoryChunkIndex()

    even = await sparse_indexer(store, index).run(shard=(0, 2))
    odd = await sparse_indexer(store, index).run(shard=(1, 2))

    assert (even.documents, odd.documents) == (4, 3)
    assert (await store.progress()).sparse_indexed == 7


async def test_the_dense_pass_waits_for_the_sparse_pass(tmp_path: Path) -> None:
    store, index = await staged(tmp_path), InMemoryChunkIndex()
    dense = DenseIndexer(store, index, Chunker(), FakeEmbedder(), batch_size=2)

    assert (await dense.run()).documents == 0

    await sparse_indexer(store, index).run()
    report = await dense.run()

    assert report.documents == 7
    assert len(index.dense) == len(index.points)
    assert await index.count_missing_dense() == 0
    assert all(len(vector) == DIMENSION for vector in index.dense.values())


async def test_redoing_the_sparse_pass_invalidates_the_dense_vectors(tmp_path: Path) -> None:
    store, index = await staged(tmp_path), InMemoryChunkIndex()
    await sparse_indexer(store, index).run()
    await DenseIndexer(store, index, Chunker(), FakeEmbedder(), batch_size=2).run()
    store.sparse_done.discard(0)  # as if document 0 had to be re-chunked

    await sparse_indexer(store, index).run()

    assert 0 not in store.dense_done
    assert await index.count_missing_dense() == store.n_chunks[0]
    again = await DenseIndexer(store, index, Chunker(), FakeEmbedder(), batch_size=2).run()
    assert again.documents == 1
    assert await index.count_missing_dense() == 0
