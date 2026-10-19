"""Embed the corpus into vector shards: `python -m rag_engine.workers.embed_corpus`.

Runs on a machine with a GPU (a Kaggle notebook, see docs/kaggle-embedding.md); it needs neither
PostgreSQL nor Qdrant. With several GPUs it starts one process per GPU, each handling a
contiguous range of shards. Output goes to `--out`, plus a `run-info-<rank>.json` per worker with
the throughput actually achieved.
"""

import argparse
import dataclasses
import json
import logging
import multiprocessing
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from rag_engine.indexing.embedders import DEFAULT_DIMENSION, DEFAULT_MODEL, DEFAULT_REVISION
from rag_engine.ingestion.parquet_source import count_rows
from rag_engine.observability.logging import configure_logging
from rag_engine.pipelines.offline_embedding import (
    EmbeddingJob,
    ShardReport,
    run_worker,
    shards_in_range,
)

logger = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    configure_logging("INFO")
    job = EmbeddingJob(
        parquet=args.parquet,
        out_dir=args.out,
        model=args.model,
        revision=args.revision or None,
        dimension=args.dimension,
        docs_per_shard=args.docs_per_shard,
        limit_docs=args.limit_docs,
    )
    total = shards_in_range(job, count_rows(job.parquet))
    first = args.first_shard
    last = total - 1 if args.last_shard is None else min(args.last_shard, total - 1)
    shards = list(range(first, last + 1))
    devices = _devices(args.device)
    logger.info(
        "embedding job",
        extra={"shards": len(shards), "first": first, "last": last, "devices": devices},
    )

    started = time.perf_counter()
    if len(devices) == 1:
        _worker(0, devices[0], job, args, shards)
        failed = False
    else:
        failed = _run_parallel(devices, job, args, shards)
    logger.info("embedding job finished", extra={"seconds": round(time.perf_counter() - started)})
    return 1 if failed else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="embed_corpus", description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--dimension", type=int, default=DEFAULT_DIMENSION)
    parser.add_argument("--docs-per-shard", type=int, default=5000)
    parser.add_argument("--first-shard", type=int, default=0)
    parser.add_argument("--last-shard", type=int, help="inclusive; default: through the last")
    parser.add_argument("--limit-docs", type=int, help="stop after N documents (smoke test)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, mps, cuda, or a comma-separated list: one worker each",
    )
    return parser


def _devices(requested: str) -> list[str]:
    if requested != "auto":
        return requested.split(",")
    import torch

    count = torch.cuda.device_count()
    return [f"cuda:{index}" for index in range(count)] if count else ["cpu"]


def _split(shards: list[int], parts: int) -> list[list[int]]:
    """Consecutive, near-equal ranges, one per worker."""
    size, extra = divmod(len(shards), parts)
    ranges, start = [], 0
    for index in range(parts):
        stop = start + size + (1 if index < extra else 0)
        ranges.append(shards[start:stop])
        start = stop
    return ranges


def _run_parallel(
    devices: list[str], job: EmbeddingJob, args: argparse.Namespace, shards: list[int]
) -> bool:
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=_worker, args=(rank, device, job, args, part))
        for rank, (device, part) in enumerate(
            zip(devices, _split(shards, len(devices)), strict=True)
        )
        if part
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    failed = False
    for rank, process in enumerate(processes):
        if process.exitcode != 0:
            failed = True
            # A negative code is the signal that killed it; -9 usually means out of memory.
            logger.error(
                "worker exited abnormally", extra={"rank": rank, "exitcode": process.exitcode}
            )
    return failed


def _worker(
    rank: int, device: str, job: EmbeddingJob, args: argparse.Namespace, shards: list[int]
) -> None:
    """Embed `shards` on `device`. Exits the process with status 1 if anything goes wrong."""
    from rag_engine.indexing.embedders import SentenceTransformerEmbedder

    configure_logging("INFO")
    try:
        embedder = SentenceTransformerEmbedder(
            job.model, revision=job.revision, device=device, batch_size=args.batch_size
        )
        if embedder.dimension != job.dimension:
            raise SystemExit(
                f"{job.model} has {embedder.dimension} dimensions, not {job.dimension}"
            )
        started = time.perf_counter()
        reports = run_worker(job, embedder, shards)
    except Exception:
        logger.exception("worker failed", extra={"rank": rank, "device": device})
        sys.exit(1)
    _write_run_info(job, rank, device, reports, time.perf_counter() - started)


def _write_run_info(
    job: EmbeddingJob, rank: int, device: str, reports: list[ShardReport], seconds: float
) -> None:
    done = [report for report in reports if not report.skipped]
    chunks = sum(report.chunks for report in done)
    characters = sum(report.characters for report in done)
    info = {
        "rank": rank,
        "device": device,
        "model": job.model,
        "revision": job.revision,
        "shards_written": len(done),
        "shards_skipped": len(reports) - len(done),
        "documents": sum(report.documents for report in done),
        "chunks": chunks,
        "characters": characters,
        "seconds": round(seconds, 1),
        "chunks_per_second": round(chunks / seconds, 1) if seconds else None,
        "shards": [dataclasses.asdict(report) for report in reports],
    }
    (job.out_dir / f"run-info-{rank}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
