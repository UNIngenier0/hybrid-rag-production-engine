"""Retrieval evaluation commands.

collect   search every question and keep the first chunks of both retrievers (needs Qdrant)
rerank    score the fused candidates with a cross-encoder (needs only the candidates file)
export-pool  keep just the fused pool of each question, to score on another machine
answer    generate the answers of a run from the stored candidates (needs PostgreSQL and a key)
grep-answer  the single-attempt grep baseline: one pattern, one search, the same answer step
compare   compare systems on the answers they gave, from the evaluator's results files
gate      replay the retrieval fixture and check the committed answer comparison (CI)
verify-check  plant wrong citations and values in real answers to test the citation check
build-gate-fixture  write the fixture the gate replays, from the stored candidates
report    compare retrievers, fusion settings and rerankers offline and write the ablation
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rag_engine.evaluation.candidates import collect_candidates, load_candidates
from rag_engine.evaluation.questions import (
    Question,
    load_questions,
    stratified_subset,
    with_gold_documents,
)
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
DEFAULT_GATE_FIXTURE = Path("eval/baselines/gate-fixture.jsonl.gz")
DEFAULT_GATE_BASELINE = Path("eval/baselines/retrieval.json")


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

    answer = commands.add_parser("answer", help="generate the answers of a run with the model")
    answer.add_argument("--run", required=True, help="names the folder eval/runs/<run>")
    answer.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    answer.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    answer.add_argument("--rerank", metavar="SCORES_FILE", type=Path, help="reranker scores")
    answer.add_argument("--mode", choices=["hybrid", "hybrid+rerank"], default="hybrid+rerank")
    answer.add_argument("--chunks", type=int, default=5, help="chunks shown to the model")
    answer.add_argument("--per-type", type=int, default=12, help="questions of each type")
    answer.add_argument("--seed", type=int, default=0)
    answer.add_argument("--limit", type=int, help="answer only the first N of the selection")
    answer.add_argument("--model", help="defaults to LLM_MODEL (Haiku)")
    answer.add_argument("--budget-usd", type=float, default=2.0, help="stop when this is spent")
    answer.add_argument("--concurrency", type=int, default=4)
    answer.add_argument("--documents", choices=["cited", "retrieved"], default="cited")
    answer.add_argument("--runs-dir", type=Path, default=Path("eval/runs"))
    answer.set_defaults(handler=_answer)

    grep = commands.add_parser("grep-answer", help="answer with the single-attempt grep baseline")
    grep.add_argument("--run", required=True, help="names the folder eval/runs/<run>")
    grep.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    grep.add_argument("--questions-from", type=Path, required=True, help="a questions.txt of IDs")
    grep.add_argument("--corpus", type=Path, required=True, help="the sources folder of documents")
    grep.add_argument("--files", type=int, default=5, help="files kept, as many as the RAG chunks")
    grep.add_argument("--model", help="defaults to LLM_MODEL (Haiku)")
    grep.add_argument("--budget-usd", type=float, default=1.0, help="stop when this is spent")
    grep.add_argument("--concurrency", type=int, default=3)
    grep.add_argument("--runs-dir", type=Path, default=Path("eval/runs"))
    grep.set_defaults(handler=_grep_answer)

    comparison = commands.add_parser("compare", help="compare systems on their answers")
    comparison.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="NAME=RUN_DIR",
        help="a system and its run folder (the first one is the baseline)",
    )
    comparison.add_argument("--judge", default="claude-haiku-5-5", help="whose results to read")
    comparison.add_argument(
        "--allow-differences",
        action="store_true",
        help="write the report even if the fairness checks fail (they stay in the report)",
    )
    comparison.add_argument(
        "--out-md", type=Path, default=Path("eval/reports/answer-comparison.md")
    )
    comparison.add_argument(
        "--out-json", type=Path, default=Path("eval/reports/answer-comparison.json")
    )
    comparison.set_defaults(handler=_compare)

    check = commands.add_parser(
        "verify-check", help="test the citation check by planting wrong citations and values"
    )
    check.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    check.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    check.add_argument("--rerank", metavar="SCORES_FILE", type=Path, required=True)
    check.add_argument("--per-type", type=int, default=12, help="questions of each type")
    check.add_argument("--seed", type=int, default=0)
    check.add_argument("--limit", type=int, help="only the first N of the selection")
    check.add_argument("--chunks", type=int, default=5)
    check.add_argument("--model", help="writes the answers; defaults to LLM_MODEL")
    check.add_argument("--judge-model", help="checks the citations; defaults to the same model")
    check.add_argument("--budget-usd", type=float, default=1.0)
    check.add_argument("--concurrency", type=int, default=4)
    check.add_argument("--out-md", type=Path, default=Path("eval/reports/citation-verification.md"))
    check.add_argument(
        "--out-json", type=Path, default=Path("eval/reports/citation-verification.json")
    )
    check.add_argument(
        "--sample-out", type=Path, help="write verified statements with their passages to read"
    )
    check.set_defaults(handler=_verify_check)

    fixture = commands.add_parser(
        "build-gate-fixture", help="write the fixture the regression gate replays"
    )
    fixture.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    fixture.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    fixture.add_argument("--rerank", metavar="SCORES_FILE", type=Path, required=True)
    fixture.add_argument("--out", type=Path, default=DEFAULT_GATE_FIXTURE)
    fixture.set_defaults(handler=_build_gate_fixture)

    gate = commands.add_parser("gate", help="fail when retrieval or the published numbers regress")
    gate.add_argument("--fixture", type=Path, default=DEFAULT_GATE_FIXTURE)
    gate.add_argument("--baseline", type=Path, default=DEFAULT_GATE_BASELINE)
    gate.add_argument(
        "--update-baseline",
        action="store_true",
        help="write the current numbers as the new baseline (after reviewing them)",
    )
    gate.set_defaults(handler=_gate)

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


def _answer(args: argparse.Namespace) -> None:
    from rag_engine.core.config import get_settings

    asyncio.run(_answer_async(args, get_settings()))


async def _answer_async(args: argparse.Namespace, settings: "Settings") -> None:
    from rag_engine.adapters.anthropic_llm import AnthropicClient
    from rag_engine.adapters.postgres_documents import PostgresDocumentStore
    from rag_engine.evaluation.answering import answer_questions
    from rag_engine.evaluation.audit import write_manifest
    from rag_engine.evaluation.stored import StoredRetriever
    from rag_engine.generation.answer import AnswerGenerator
    from rag_engine.generation.cost import CostMeter, MeteredLLM

    if settings.anthropic_api_key is None:
        sys.exit("ANTHROPIC_API_KEY is not set; put it in .env")
    everything = load_questions(args.questions)
    selection = stratified_subset(with_gold_documents(everything), args.per_type, seed=args.seed)
    selection = selection[: args.limit] if args.limit else selection
    run_dir = _write_selection(args.runs_dir / args.run, selection)
    model = args.model or settings.llm_model
    write_manifest(
        run_dir,
        {
            "system": "rag",
            "model": model,
            "mode": args.mode,
            "passages": args.chunks,
            "documents": args.documents,
            "selection": {"per_type": args.per_type, "seed": args.seed, "limit": args.limit},
        },
    )
    scores = load_scores(args.rerank) if args.rerank else None
    store = await PostgresDocumentStore.connect(settings.database_url.get_secret_value())
    meter = CostMeter(args.budget_usd)
    client = AnthropicClient(
        settings.anthropic_api_key.get_secret_value(), args.model or settings.llm_model
    )
    try:
        generator = AnswerGenerator(
            StoredRetriever(everything, load_candidates(args.candidates), scores, store),
            MeteredLLM(client, meter),
            mode=args.mode,
            chunks=args.chunks,
            max_tokens=settings.llm_max_tokens,
        )
        summary = await answer_questions(
            selection,
            generator,
            run_dir,
            documents=args.documents,
            concurrency=args.concurrency,
            on_progress=_progress("answered"),
        )
    finally:
        await client.aclose()
        await store.close()
    print(
        f"\n{summary}\nspent ${meter.spent_usd:.4f} on {meter.calls} calls "
        f"({meter.input_tokens} tokens in, {meter.output_tokens} out); run folder: {run_dir}"
    )


def _verify_check(args: argparse.Namespace) -> None:
    from rag_engine.core.config import get_settings

    asyncio.run(_verify_check_async(args, get_settings()))


async def _verify_check_async(args: argparse.Namespace, settings: "Settings") -> None:
    from dataclasses import asdict

    from rag_engine.adapters.anthropic_llm import AnthropicClient
    from rag_engine.adapters.postgres_documents import PostgresDocumentStore
    from rag_engine.evaluation.stored import StoredRetriever
    from rag_engine.evaluation.verification_check import check_verification, render
    from rag_engine.generation.answer import AnswerGenerator
    from rag_engine.generation.cost import CostMeter, MeteredLLM
    from rag_engine.generation.query import QueryService
    from rag_engine.generation.verify import CitationVerifier

    if settings.anthropic_api_key is None:
        sys.exit("ANTHROPIC_API_KEY is not set; put it in .env")
    everything = load_questions(args.questions)
    selection = stratified_subset(with_gold_documents(everything), args.per_type, seed=args.seed)
    selection = selection[: args.limit] if args.limit else selection
    key = settings.anthropic_api_key.get_secret_value()
    meter = CostMeter(args.budget_usd)
    writer = AnthropicClient(key, args.model or settings.llm_model)
    judge = AnthropicClient(key, args.judge_model or args.model or settings.llm_model)
    store = await PostgresDocumentStore.connect(settings.database_url.get_secret_value())
    try:
        retriever = StoredRetriever(
            everything, load_candidates(args.candidates), load_scores(args.rerank), store
        )
        generator = AnswerGenerator(
            retriever,
            MeteredLLM(writer, meter),
            mode="hybrid+rerank",
            chunks=args.chunks,
            max_tokens=settings.llm_max_tokens,
        )
        verifier = CitationVerifier(MeteredLLM(judge, meter), max_tokens=settings.llm_max_tokens)
        report = await check_verification(
            selection,
            QueryService(generator, verifier),
            verifier,
            seed=args.seed,
            concurrency=args.concurrency,
        )
    finally:
        await writer.aclose()
        await judge.aclose()
        await store.close()
    report.cost_usd = meter.spent_usd  # everything the check spent, planted damage included
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render(report), encoding="utf-8")
    summary = {
        "answers": report.answers,
        "failed": report.failed,
        "answers_all_verified": report.answers_all_verified,
        "judge_errors": dict(report.judge_errors),
        "statuses": dict(report.statuses),
        "planted": [asdict(outcome) for outcome in report.outcomes],
        "cost_usd": round(report.cost_usd, 4),
    }
    args.out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.sample_out:
        review = {"verified": report.sample, "flagged": report.flagged}
        args.sample_out.write_text(json.dumps(review, indent=2), encoding="utf-8")
    print(render(report))


def _grep_answer(args: argparse.Namespace) -> None:
    from rag_engine.core.config import get_settings

    asyncio.run(_grep_answer_async(args, get_settings()))


async def _grep_answer_async(args: argparse.Namespace, settings: "Settings") -> None:
    from rag_engine.adapters.anthropic_llm import AnthropicClient
    from rag_engine.evaluation.answering import answer_questions
    from rag_engine.evaluation.audit import write_manifest
    from rag_engine.evaluation.grep_baseline import SEARCH_TIMEOUT_SECONDS, GrepRetriever
    from rag_engine.generation.answer import AnswerGenerator
    from rag_engine.generation.cost import CostMeter, MeteredLLM

    if settings.anthropic_api_key is None:
        sys.exit("ANTHROPIC_API_KEY is not set; put it in .env")
    wanted = [line.strip() for line in args.questions_from.read_text().splitlines() if line.strip()]
    by_id = {question.question_id: question for question in load_questions(args.questions)}
    selection = [by_id[question_id] for question_id in wanted]
    run_dir = _write_selection(args.runs_dir / args.run, selection)
    write_manifest(
        run_dir,
        {
            "system": "grep-oneshot",
            "model": args.model or settings.llm_model,
            "passages": args.files,
            "documents": "cited",
            "window_chars": 3200,
            "search_timeout_seconds": SEARCH_TIMEOUT_SECONDS,
        },
    )
    meter = CostMeter(args.budget_usd)
    client = AnthropicClient(
        settings.anthropic_api_key.get_secret_value(),
        args.model or settings.llm_model,
        base_url=settings.anthropic_base_url,
    )
    llm = MeteredLLM(client, meter)
    retriever = GrepRetriever(llm, args.corpus, files=args.files)
    try:
        generator = AnswerGenerator(
            retriever, llm, mode="hybrid", chunks=args.files, max_tokens=settings.llm_max_tokens
        )
        summary = await answer_questions(
            selection,
            generator,
            run_dir,
            documents="cited",
            concurrency=args.concurrency,
            on_progress=_progress("answered"),
        )
    finally:
        await client.aclose()
    by_text = {question.question: question.question_id for question in selection}
    _append_searches(run_dir / "searches.jsonl", retriever.searches, by_text)
    unusable = sum(1 for plan in retriever.plans.values() if plan is None)
    timed_out = sum(1 for search in retriever.searches.values() if search["timed_out"])
    print(
        f"\n{summary}\nspent ${meter.spent_usd:.4f} on {meter.calls} calls; "
        f"{unusable} questions got no usable pattern, {timed_out} searches were abandoned; "
        f"run folder: {run_dir}"
    )


def _compare(args: argparse.Namespace) -> None:
    from rag_engine.evaluation.audit import audit_runs
    from rag_engine.evaluation.compare import compare
    from rag_engine.evaluation.runs import load_arms

    specs = []
    for item in args.arm:
        name, _, folder = item.partition("=")
        if not folder:
            sys.exit(f"--arm expects NAME=RUN_DIR, got {item!r}")
        specs.append((name, Path(folder)))
    audit = audit_runs(specs, args.judge)
    if not audit.passed and not args.allow_differences:
        print(audit.markdown())
        sys.exit(
            "the systems are not comparable like for like; fix that or pass --allow-differences"
        )
    comparison = compare(load_arms(specs, args.judge), baseline=specs[0][0], audit=audit)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(comparison.markdown, encoding="utf-8")
    args.out_json.write_text(json.dumps(comparison.data, indent=2), encoding="utf-8")
    print(comparison.markdown)


def _append_searches(path: Path, searches: dict[str, dict[str, Any]], ids: dict[str, str]) -> None:
    """Keep what each search did, appended so a resumed run adds to the earlier one."""
    with path.open("a", encoding="utf-8") as handle:
        for query, search in searches.items():
            row = {"question_id": ids[query], **search}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_gate_fixture(args: argparse.Namespace) -> None:
    from rag_engine.evaluation.gate import build_fixture

    model, _, _ = scoring_speed(args.rerank)
    count = build_fixture(
        _questions(args.questions, None),
        load_candidates(args.candidates),
        load_scores(args.rerank),
        args.out,
        reranker=model,
    )
    print(f"wrote {count} questions to {args.out}")


def _gate(args: argparse.Namespace) -> None:
    from rag_engine.evaluation.gate import (
        check_answer_report,
        load_fixture,
        make_baseline,
        measure,
        run_gate,
        shipped_pipeline,
    )

    if args.update_baseline:
        meta, cases = load_fixture(args.fixture)
        pipeline = shipped_pipeline()
        baseline = make_baseline(args.fixture, meta, pipeline, measure(cases, pipeline))
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
        print(f"baseline written to {args.baseline}")
        return
    retrieval = run_gate(args.fixture, args.baseline)
    print(retrieval.text)
    problems = check_answer_report()
    print("Answer comparison: " + ("consistent and like for like." if not problems else ""))
    for problem in problems:
        print(f"- {problem}")
    if not retrieval.passed or problems:
        sys.exit(1)


def _write_selection(run_dir: Path, selection: list[Question]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    ids = "\n".join(question.question_id for question in selection)
    (run_dir / "questions.txt").write_text(ids + "\n", encoding="utf-8")
    return run_dir


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
