import json
from pathlib import Path

from fakes import FakeLLM, FakeRetriever, make_hit
from rag_engine.core.errors import BudgetExceededError, LLMError
from rag_engine.core.models import LLMReply, LLMUsage
from rag_engine.evaluation.answering import answer_questions
from rag_engine.evaluation.questions import Question
from rag_engine.generation.answer import AnswerGenerator
from rag_engine.generation.cost import CostMeter, MeteredLLM


def question(question_id: str) -> Question:
    return Question(question_id, "basic", f"question {question_id}?", ("d",), (), "", ())


def generator(llm=None, hits=None) -> AnswerGenerator:  # type: ignore[no-untyped-def]
    hits = hits or [
        make_hit("a", doc_id="d1", context="c1"),
        make_hit("b", doc_id="d2", context="c2"),
    ]
    return AnswerGenerator(FakeRetriever(hits), llm or FakeLLM("The answer [1]."), chunks=2)


def rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


async def test_each_answer_is_written_in_the_benchmarks_format(tmp_path: Path) -> None:
    summary = await answer_questions([question("q1"), question("q2")], generator(), tmp_path)

    written = sorted(rows(tmp_path / "answers.jsonl"), key=lambda row: str(row["question_id"]))
    assert (summary.answered, summary.failed, summary.skipped) == (2, 0, 0)
    assert written == [
        {"question_id": "q1", "answer": "The answer.", "document_ids": ["d1"]},
        {"question_id": "q2", "answer": "The answer.", "document_ids": ["d1"]},
    ]


async def test_the_trace_keeps_what_the_answer_rested_on_and_cost(tmp_path: Path) -> None:
    await answer_questions([question("q1")], generator(), tmp_path)

    (trace,) = rows(tmp_path / "trace.jsonl")
    assert trace["raw_answer"] == "The answer [1]."
    assert trace["cited_doc_ids"] == ["d1"]
    assert trace["retrieved_doc_ids"] == ["d1", "d2"]
    assert (trace["input_tokens"], trace["output_tokens"]) == (1000, 100)
    assert trace["question_type"] == "basic"


async def test_the_documents_reported_can_be_everything_the_model_saw(tmp_path: Path) -> None:
    await answer_questions([question("q1")], generator(), tmp_path, documents="retrieved")

    (answer,) = rows(tmp_path / "answers.jsonl")
    assert answer["document_ids"] == ["d1", "d2"]


async def test_a_second_run_continues_with_the_questions_left(tmp_path: Path) -> None:
    await answer_questions([question("q1")], generator(), tmp_path)
    llm = FakeLLM("Another [2].")

    summary = await answer_questions([question("q1"), question("q2")], generator(llm), tmp_path)

    assert (summary.answered, summary.skipped) == (1, 1)
    assert len(llm.prompts) == 1  # only q2 was sent to the model
    assert {row["question_id"] for row in rows(tmp_path / "answers.jsonl")} == {"q1", "q2"}


class Failing:
    model = "claude-haiku-5-5"

    def __init__(self, error: Exception, succeed_after: int = 0) -> None:
        self.error = error
        self.calls = 0
        self.succeed_after = succeed_after

    async def complete(self, system, user, *, max_tokens=1024, temperature=None):  # type: ignore[no-untyped-def]
        self.calls += 1
        if self.calls > self.succeed_after:
            raise self.error
        return LLMReply("ok [1]", LLMUsage(10, 5), self.model)


async def test_a_failed_question_is_recorded_and_the_run_goes_on(tmp_path: Path) -> None:
    llm = Failing(LLMError("the API answered 400"), succeed_after=1)

    summary = await answer_questions(
        [question("q1"), question("q2")], generator(llm), tmp_path, concurrency=1
    )

    assert (summary.answered, summary.failed) == (1, 1)
    trace = rows(tmp_path / "trace.jsonl")
    assert any("error" in row and "400" in str(row["error"]) for row in trace)
    assert len(rows(tmp_path / "answers.jsonl")) == 1


async def test_a_spent_budget_ends_the_run_without_losing_what_was_answered(tmp_path: Path) -> None:
    meter = CostMeter(budget_usd=0.0003)
    llm = MeteredLLM(FakeLLM("Fine [1].", usage=LLMUsage(2000, 200)), meter)  # $0.0003 a call

    summary = await answer_questions(
        [question(f"q{i}") for i in range(5)], generator(llm), tmp_path, concurrency=1
    )

    assert summary.stopped_by_budget
    assert summary.answered == 1
    assert len(rows(tmp_path / "answers.jsonl")) == 1


async def test_a_budget_error_is_not_reported_as_a_failed_question(tmp_path: Path) -> None:
    llm = Failing(BudgetExceededError("spent"))

    summary = await answer_questions([question("q1")], generator(llm), tmp_path)

    assert (summary.answered, summary.failed, summary.stopped_by_budget) == (0, 0, True)


async def test_a_declined_question_is_answered_with_the_refusal_and_reports_no_documents(
    tmp_path: Path,
) -> None:
    from rag_engine.generation.abstain import MESSAGE, AbstentionPolicy

    llm = FakeLLM("never used")
    weak = AnswerGenerator(
        FakeRetriever([make_hit("a", doc_id="d1", score=-5.0)]),
        llm,
        mode="hybrid+rerank",
        abstention=AbstentionPolicy(0.0),
    )

    await answer_questions([question("q1")], weak, tmp_path)

    (answer,) = rows(tmp_path / "answers.jsonl")
    (trace,) = rows(tmp_path / "trace.jsonl")
    assert answer == {"question_id": "q1", "answer": MESSAGE, "document_ids": []}
    assert trace["abstained"] is True
    assert trace["abstention"] == {"reason": "weak_evidence", "top_score": -5.0, "threshold": 0.0}
    assert (trace["input_tokens"], trace["output_tokens"]) == (0, 0)
    assert llm.prompts == []


async def test_an_answered_question_is_not_marked_as_declined_in_the_trace(
    tmp_path: Path,
) -> None:
    await answer_questions([question("q1")], generator(), tmp_path)

    (trace,) = rows(tmp_path / "trace.jsonl")
    assert (trace["abstained"], trace["abstention"]) == (False, None)
