"""Command line for bulk ingestion: `python -m rag_engine.workers.bulk_ingest <command>`.

    profile     measure chunks, tokens and throughput on the corpus (needs no services)
    stage       copy the cleaned corpus from Parquet into PostgreSQL
    index       run the `sparse` (BM25) pass, or the `dense` pass on a machine with a GPU
    load-vectors attach the dense vectors computed by the offline GPU job (vector shards)
    dense-graph turn the dense HNSW graph off while loading and on to build it
    status      show how far each stage has got

The dense vectors are normally produced elsewhere, see docs/kaggle-embedding.md.

Every command can be interrupted and run again; see `rag_engine.pipelines.bulk_ingestion`.
"""

import argparse
import asyncio
import dataclasses
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from qdrant_client import AsyncQdrantClient

from rag_engine.adapters.postgres_documents import PostgresDocumentStore
from rag_engine.adapters.qdrant_index import QdrantChunkIndex
from rag_engine.core.config import Settings, get_settings
from rag_engine.core.ports import Shard
from rag_engine.indexing.shards import pipeline_fingerprint
from rag_engine.indexing.sparse import Bm25SparseEncoder, mean_document_length
from rag_engine.ingestion.chunking import Chunker, ChunkingConfig
from rag_engine.ingestion.parquet_source import CleaningStats
from rag_engine.ingestion.profiling import CorpusProfile, SourceProfile, profile_corpus
from rag_engine.observability.logging import configure_logging
from rag_engine.pipelines.bulk_ingestion import DenseIndexer, SparseIndexer, stage_documents
from rag_engine.pipelines.vector_loading import ExpectedPipeline, load_vectors

logger = logging.getLogger(__name__)

_MAX_EMBEDDING_TOKENS = 512
# Embedding more documents than this on a CPU or laptop GPU is refused unless asked for.
_SLOW_DEVICE_DOCUMENT_LIMIT = 500


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level)
    asyncio.run(args.handler(args, settings))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bulk_ingest", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    profile = commands.add_parser("profile", help="measure chunking and sizing")
    profile.add_argument("--path", type=Path)
    profile.add_argument("--limit", type=int, help="profile only the first N documents")
    profile.add_argument("--tokenizer", help="HF model whose tokenizer counts tokens")
    profile.add_argument("--output", type=Path, help="write the JSON report here too")
    profile.set_defaults(handler=_profile)

    stage = commands.add_parser("stage", help="load the cleaned corpus into PostgreSQL")
    stage.add_argument("--path", type=Path)
    stage.add_argument("--limit", type=int)
    stage.set_defaults(handler=_stage)

    index = commands.add_parser("index", help="run an indexing pass")
    index.add_argument("pass_name", choices=["sparse", "dense"])
    index.add_argument("--limit", type=int, help="handle at most N documents")
    index.add_argument("--shard", type=_shard, help="i/n: handle documents with seq %% n == i")
    index.add_argument(
        "--allow-slow-device",
        action="store_true",
        help="allow the dense pass on a CPU or laptop GPU for more than "
        f"{_SLOW_DEVICE_DOCUMENT_LIMIT} documents (hours for the whole corpus)",
    )
    index.set_defaults(handler=_index)

    load = commands.add_parser("load-vectors", help="attach dense vectors from vector shards")
    load.add_argument("--dir", type=Path, help="folder with the vectors-*.parquet shards")
    load.set_defaults(handler=_load_vectors)

    graph = commands.add_parser("dense-graph", help="turn the dense HNSW graph on or off")
    graph.add_argument("state", choices=["on", "off"])
    graph.set_defaults(handler=_dense_graph)

    status = commands.add_parser("status", help="show ingestion progress")
    status.set_defaults(handler=_status)
    return parser


def _shard(value: str) -> Shard:
    index, _, total = value.partition("/")
    shard = (int(index), int(total))
    if not 0 <= shard[0] < shard[1]:
        raise argparse.ArgumentTypeError("shard must look like i/n with 0 <= i < n")
    return shard


def _chunker(settings: Settings) -> Chunker:
    return Chunker(
        ChunkingConfig(
            target_chars=settings.chunk_target_chars,
            max_chars=settings.chunk_max_chars,
            min_chars=settings.chunk_min_chars,
            overlap_chars=settings.chunk_overlap_chars,
            parent_target_chars=settings.chunk_parent_chars,
        )
    )


def _sparse_encoder(settings: Settings) -> Bm25SparseEncoder:
    return Bm25SparseEncoder(avg_len=settings.bm25_avg_len)


async def _open(settings: Settings) -> tuple[PostgresDocumentStore, QdrantChunkIndex]:
    store = await PostgresDocumentStore.connect(settings.database_url.get_secret_value())
    client = AsyncQdrantClient(
        url=settings.qdrant_url,
        timeout=120,
        prefer_grpc=settings.qdrant_prefer_grpc,
        grpc_port=settings.qdrant_grpc_port,
    )
    return store, QdrantChunkIndex(client, settings.qdrant_collection)


async def _stage(args: argparse.Namespace, settings: Settings) -> None:
    store = await PostgresDocumentStore.connect(settings.database_url.get_secret_value())
    stats = CleaningStats()
    try:
        report = await stage_documents(
            store, args.path or settings.dataset_path, limit=args.limit, stats=stats
        )
    finally:
        await store.close()
    _print({"stage": dataclasses.asdict(report), "cleaning": _cleaning(stats)})


async def _index(args: argparse.Namespace, settings: Settings) -> None:
    store, index = await _open(settings)
    try:
        await store.ensure_schema()
        await index.ensure_ready(settings.embedding_dimension)
        chunker = _chunker(settings)
        if args.pass_name == "sparse":
            sparse = SparseIndexer(
                store,
                index,
                chunker,
                _sparse_encoder(settings),
                batch_size=settings.ingest_batch_size,
            )
            report = await sparse.run(limit=args.limit, shard=args.shard)
        else:
            try:
                # Imported here so the sparse pass and the API do not need torch installed.
                import torch

                from rag_engine.indexing.embedders import (
                    SentenceTransformerEmbedder,
                    resolve_device,
                )
            except ImportError as error:
                raise SystemExit(
                    "the dense pass needs the `embeddings` dependency group "
                    "(uv sync --group embeddings); to compute vectors on a GPU instead, "
                    "see docs/kaggle-embedding.md"
                ) from error
            device = resolve_device(settings.embedding_device, torch)
            _refuse_slow_device(device, args)  # before loading a model that would not be used
            embedder = SentenceTransformerEmbedder(
                settings.embedding_model,
                revision=settings.embedding_revision,
                device=device,
                batch_size=settings.embedding_batch_size,
            )
            if embedder.dimension != settings.embedding_dimension:
                raise SystemExit(
                    f"{embedder.model_name} produces {embedder.dimension}-dimensional vectors, "
                    f"but EMBEDDING_DIMENSION is {settings.embedding_dimension}"
                )
            logger.info(
                "embedder ready", extra={"model": embedder.model_name, "device": embedder.device}
            )
            dense = DenseIndexer(
                store, index, chunker, embedder, batch_size=settings.ingest_batch_size
            )
            report = await dense.run(limit=args.limit, shard=args.shard)
        _print({args.pass_name: dataclasses.asdict(report), "points": await index.count()})
    finally:
        await index.close()
        await store.close()


def _refuse_slow_device(device: str, args: argparse.Namespace) -> None:
    """Keep a laptop from starting hours of embedding by accident."""
    limit = args.limit
    too_big = limit is None or limit > _SLOW_DEVICE_DOCUMENT_LIMIT
    if not device.startswith("cuda") and too_big and not args.allow_slow_device:
        raise SystemExit(
            f"refusing to embed more than {_SLOW_DEVICE_DOCUMENT_LIMIT} documents on '{device}': "
            "the whole corpus takes hours here. Compute the vectors on a GPU instead "
            "(docs/kaggle-embedding.md) and run `load-vectors`, or pass --allow-slow-device."
        )


async def _load_vectors(args: argparse.Namespace, settings: Settings) -> None:
    store, index = await _open(settings)
    expected = ExpectedPipeline(
        model=settings.embedding_model,
        revision=settings.embedding_revision,
        dimension=settings.embedding_dimension,
        fingerprint=pipeline_fingerprint(_chunker(settings).config),
    )
    try:
        report = await load_vectors(store, index, args.dir or settings.vectors_dir, expected)
        _print(
            {
                "load_vectors": dataclasses.asdict(report),
                "points_without_dense": await index.count_missing_dense(),
            }
        )
    finally:
        await index.close()
        await store.close()


async def _dense_graph(args: argparse.Namespace, settings: Settings) -> None:
    _, index = await _open(settings)
    try:
        await index.set_dense_graph(args.state == "on")
    finally:
        await index.close()
    _print({"dense_graph": args.state})


async def _status(args: argparse.Namespace, settings: Settings) -> None:
    store, index = await _open(settings)
    try:
        await store.ensure_schema()
        progress = await store.progress()
        _print(
            {
                "documents": dataclasses.asdict(progress),
                "points": await index.count(),
                "points_without_dense": await index.count_missing_dense(),
            }
        )
    finally:
        await index.close()
        await store.close()


async def _profile(args: argparse.Namespace, settings: Settings) -> None:
    result = await asyncio.to_thread(
        profile_corpus, args.path or settings.dataset_path, _chunker(settings), limit=args.limit
    )
    report = _profile_report(result, settings)
    if args.tokenizer:
        report["tokens"] = _token_report(result, args.tokenizer)
    report["bm25_avg_len"] = round(mean_document_length(result.sample_texts), 1)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _print(report)


def _profile_report(result: CorpusProfile, settings: Settings) -> dict[str, Any]:
    def summarize(profile: SourceProfile) -> dict[str, Any]:
        documents = max(profile.documents, 1)
        return {
            "documents": profile.documents,
            "megachars": round(profile.chars / 1e6, 1),
            "chunks": profile.chunks,
            "chunks_per_document": round(profile.chunks / documents, 2),
            "single_chunk_documents_pct": round(
                100 * profile.single_chunk_documents / documents, 1
            ),
            "parents": profile.parents,
            "chunk_chars_p50": profile.percentile(0.50),
            "chunk_chars_p90": profile.percentile(0.90),
            "chunk_chars_p99": profile.percentile(0.99),
            "indexed_megachars": round(profile.indexed_chars / 1e6, 1),
        }

    total = result.total()
    return {
        "config": {
            "chunk_target_chars": settings.chunk_target_chars,
            "chunk_max_chars": settings.chunk_max_chars,
            "chunk_overlap_chars": settings.chunk_overlap_chars,
            "chunk_parent_chars": settings.chunk_parent_chars,
        },
        "total": summarize(total),
        "by_source": {name: summarize(p) for name, p in sorted(result.sources.items())},
        "cleaning": _cleaning(result.cleaning),
        "cpu_seconds": round(result.seconds, 1),
        "documents_per_second": round(total.documents / result.seconds, 1) if result.seconds else 0,
        "megachars_per_second": round(total.chars / 1e6 / result.seconds, 2)
        if result.seconds
        else 0,
    }


def _cleaning(stats: CleaningStats) -> dict[str, int]:
    return {
        "documents": stats.documents,
        "decoded_lists": stats.decoded_lists,
        "unescaped_newlines": stats.unescaped_newlines,
        "control_chars": stats.control_chars,
        "titles_repaired": stats.titles_repaired,
    }


def _token_report(result: CorpusProfile, tokenizer_name: str) -> dict[str, Any]:
    """Count tokens on the sampled chunks and extrapolate to the corpus."""
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_pretrained(tokenizer_name)
    tokenizer.no_truncation()
    tokenizer.no_padding()
    texts = result.sample_texts
    lengths = [len(encoding.ids) for encoding in tokenizer.encode_batch(texts)]
    sampled_chars = sum(len(text) for text in texts)
    tokens_per_char = sum(lengths) / sampled_chars
    total = result.total()
    return {
        "tokenizer": tokenizer_name,
        "sampled_chunks": len(texts),
        "tokens_per_char": round(tokens_per_char, 4),
        "chars_per_token": round(1 / tokens_per_char, 2),
        "mean_tokens_per_chunk": round(sum(lengths) / len(lengths), 1),
        "max_tokens_per_chunk": max(lengths),
        "chunks_over_limit_pct": round(
            100 * sum(length > _MAX_EMBEDDING_TOKENS for length in lengths) / len(lengths), 3
        ),
        "estimated_total_tokens_millions": round(total.indexed_chars * tokens_per_char / 1e6, 1),
    }


def _print(value: dict[str, Any]) -> None:
    json.dump(value, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
