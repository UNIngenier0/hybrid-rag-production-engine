"""Generate the answers of a run and write them in the benchmark's file format.

A run is a folder with `answers.jsonl` (what the benchmark's evaluator reads: `question_id`,
`answer`, `document_ids`) and `trace.jsonl` (what each answer cost and which documents it was built
from). Both are appended as answers finish, so an interrupted or budget-limited run continues
where it stopped.
"""

import asyncio
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from rag_engine.core.errors import BudgetExceededError, LLMError
from rag_engine.evaluation.questions import Question
from rag_engine.generation.answer import AnswerGenerator, GeneratedAnswer

# Which documents an answer reports: the ones its citations point at, or every document it saw.
DocumentPolicy = Literal["cited", "retrieved"]


@dataclass(frozen=True, slots=True)
class RunSummary:
    answered: int
    failed: int
    skipped: int  # already in the run folder
    stopped_by_budget: bool


async def answer_questions(
    questions: Sequence[Question],
    generator: AnswerGenerator,
    run_dir: Path,
    *,
    documents: DocumentPolicy = "cited",
    concurrency: int = 4,
    on_progress: Callable[[int, int], None] | None = None,
) -> RunSummary:
    _ensure_directory(run_dir)
    answers_file, trace_file = run_dir / "answers.jsonl", run_dir / "trace.jsonl"
    done = _question_ids(answers_file)
    pending = [question for question in questions if question.question_id not in done]
    semaphore = asyncio.Semaphore(concurrency)
    outcome = {"answered": 0, "failed": 0, "budget": False}

    async def work(question: Question) -> None:
        if outcome["budget"]:
            return
        async with semaphore:
            if outcome["budget"]:
                return
            try:
                generated = await generator.answer(question.question)
            except BudgetExceededError:
                outcome["budget"] = True
                return
            except LLMError as error:
                outcome["failed"] += 1
                _append(trace_file, {"question_id": question.question_id, "error": str(error)})
                return
        _append(answers_file, _answer_row(question, generated, documents))
        _append(trace_file, _trace_row(question, generated))
        outcome["answered"] += 1
        if on_progress is not None:
            on_progress(outcome["answered"], len(pending))

    await asyncio.gather(*(work(question) for question in pending))
    return RunSummary(
        answered=outcome["answered"],
        failed=outcome["failed"],
        skipped=len(done),
        stopped_by_budget=bool(outcome["budget"]),
    )


def _answer_row(
    question: Question, generated: GeneratedAnswer, documents: DocumentPolicy
) -> dict[str, Any]:
    ids = generated.cited_doc_ids if documents == "cited" else generated.retrieved_doc_ids
    return {"question_id": question.question_id, "answer": generated.text, "document_ids": ids}


def _trace_row(question: Question, generated: GeneratedAnswer) -> dict[str, Any]:
    return {
        "question_id": question.question_id,
        "question_type": question.question_type,
        "raw_answer": generated.raw_text,
        "cited_doc_ids": generated.cited_doc_ids,
        "retrieved_doc_ids": generated.retrieved_doc_ids,
        "unresolved_markers": generated.unresolved_markers,
        "input_tokens": generated.usage.input_tokens,
        "output_tokens": generated.usage.output_tokens,
        "model": generated.model,
    }


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _append(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _question_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as handle:
        return {json.loads(line)["question_id"] for line in handle if line.strip()}
