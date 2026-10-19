"""Retrieval evaluation commands.

collect   search every question and keep the first chunks of both retrievers (needs Qdrant)
rerank    score the fused candidates with a cross-encoder (needs only the candidates file)
export-pool  keep just the fused pool of each question, to score on another machine
report    compare retrievers, fusion settings and rerankers offline and write the ablation
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from rag_engine.evaluation.candidates import collect_candidates, load_candidates
from rag_engine.evaluation.questions import Question, load_questions, with_gold_documents
from rag_engine.evaluation.report import RerankerRun, build_report
from rag_engine.evaluation.rerank_scores import (
    POOL_DEPTH,
    export_pools,
    load_scores,
    score_pools,
    scoring_speed,
)
from rag_engine.retrieval.cross_encoder import CrossEncoderReranker

if TYPE_CHECKING:
    from rag_engine.core.config import Settings

DEFAULT_QUESTIONS = Path("data/raw/questions/test.parquet")
DEFAULT_CANDIDATES = Path("eval/cache/candidates.jsonl.gz")


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    args.handler(args)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rag_engine.evaluation.cli", description=__doc__
    )
    commands = parser.add_subparsers(dest="command", required=True)

    collect = commands.add_parser("collect", help="search every question and store the candidates")
    collect.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    collect.add_argument("--out", type=Path, default=DEFAULT_CANDIDATES)
    collect.add_argument("--depth", type=int, default=100, help="chunks kept per retriever")
    collect.add_argument("--concurrency", type=int, default=2)
    collect.add_argument("--limit", type=int, help="only the first N questions with gold documents")
    collect.set_defaults(handler=_collect)

    rerank = commands.add_parser("rerank", help="score the fused candidates with a cross-encoder")
    rerank.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    rerank.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    rerank.add_argument("--model", required=True, help="a sentence-transformers cross-encoder")
    rerank.add_argument("--out", type=Path, required=True)
    rerank.add_argument("--pool", type=int, default=POOL_DEPTH, help="fused chunks scored")
    rerank.add_argument("--device", default="auto")
    rerank.add_argument("--batch-size", type=int, default=32)
    rerank.add_argument("--limit", type=int, help="only the first N questions")
    rerank.set_defaults(handler=_rerank)

    export = commands.add_parser(
        "export-pool", help="write a compact candidates file with just the pool to rerank"
    )
    export.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    export.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    export.add_argument("--out", type=Path, default=Path("eval/cache/pool.jsonl.gz"))
    export.add_argument("--pool", type=int, default=POOL_DEPTH)
    export.set_defaults(handler=_export_pool)

    report = commands.add_parser("report", help="write the retrieval ablation")
    report.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    report.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    report.add_argument(
        "--rerank",
        action="append",
        default=[],
        metavar="LABEL=SCORES_FILE",
        help="a scores file written by `rerank`, with the label to show in the report",
    )
    report.add_argument("--out-md", type=Path, default=Path("eval/reports/retrieval-ablation.md"))
    report.add_argument(
        "--out-json", type=Path, default=Path("eval/reports/retrieval-ablation.json")
    )
    report.set_defaults(handler=_report)
    return parser


def _questions(path: Path, limit: int | None) -> list[Question]:
    questions = with_gold_documents(load_questions(path))
    return questions[:limit] if limit else questions


def _collect(args: argparse.Namespace) -> None:
    # Imported here: only this command needs the data-service stack, and `rerank` has to run where
    # that stack is not installed (a Kaggle notebook).
    from rag_engine.core.config import get_settings

    asyncio.run(_collect_async(args, get_settings()))


async def _collect_async(args: argparse.Namespace, settings: "Settings") -> None:
    from qdrant_client import AsyncQdrantClient

    from rag_engine.adapters.qdrant_index import QdrantChunkIndex
    from rag_engine.indexing.embedders import SentenceTransformerEmbedder
    from rag_engine.indexing.sparse import Bm25SparseEncoder

    questions = _questions(args.questions, args.limit)
    embedder = SentenceTransformerEmbedder(
        settings.embedding_model,
        revision=settings.embedding_revision,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
    )
    sparse = Bm25SparseEncoder(avg_len=settings.bm25_avg_len)
    index = QdrantChunkIndex(
        AsyncQdrantClient(
            url=settings.qdrant_url,
            timeout=settings.qdrant_timeout_seconds,
            prefer_grpc=settings.qdrant_prefer_grpc,
            grpc_port=settings.qdrant_grpc_port,
        ),
        settings.qdrant_collection,
        search_ef=settings.qdrant_hnsw_ef,
    )
    try:
        searched = await collect_candidates(
            questions,
            index,
            embedder,
            sparse,
            args.out,
            depth=args.depth,
            concurrency=args.concurrency,
            on_progress=_progress("searched"),
        )
    finally:
        await index.close()
    print(f"\nsearched {searched} questions; candidates are in {args.out}")


def _rerank(args: argparse.Namespace) -> None:
    questions = _questions(args.questions, args.limit)
    candidates = load_candidates(args.candidates)
    reranker = CrossEncoderReranker(args.model, device=args.device, batch_size=args.batch_size)
    scored = score_pools(
        questions, candidates, reranker, args.out, pool=args.pool, on_progress=_progress("scored")
    )
    model, ms_per_pair, device = scoring_speed(args.out)
    print(f"\nscored {scored} questions with {model} on {device}: {ms_per_pair:.1f} ms per pair")


def _export_pool(args: argparse.Namespace) -> None:
    questions = _questions(args.questions, None)
    count = export_pools(questions, load_candidates(args.candidates), args.out, pool=args.pool)
    print(f"wrote the pools of {count} questions to {args.out}")


def _report(args: argparse.Namespace) -> None:
    runs = []
    for item in args.rerank:
        label, _, path = item.partition("=")
        if not path:
            sys.exit(f"--rerank expects LABEL=SCORES_FILE, got {item!r}")
        model, ms_per_pair, device = scoring_speed(Path(path))
        runs.append(RerankerRun(label, model, load_scores(Path(path)), ms_per_pair, device))
    questions = with_gold_documents(load_questions(args.questions))
    report = build_report(questions, load_candidates(args.candidates), runs)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(report.markdown, encoding="utf-8")
    args.out_json.write_text(json.dumps(report.data, indent=2), encoding="utf-8")
    print(report.markdown)


def _progress(verb: str):  # type: ignore[no-untyped-def]
    def show(done: int, total: int) -> None:
        if done % 25 == 0 or done == total:
            print(f"{verb} {done}/{total}", flush=True)

    return show


if __name__ == "__main__":
    main()
