import json
import random
import re

from fakes import FakeRetriever, make_hit
from rag_engine.core.models import LLMReply, LLMUsage
from rag_engine.evaluation.questions import Question
from rag_engine.evaluation.verification_check import (
    Probe,
    change_number,
    check_verification,
    move_citation,
    render,
)
from rag_engine.generation.answer import AnswerGenerator
from rag_engine.generation.prompt import SYSTEM_PROMPT, ContextBlock
from rag_engine.generation.query import QueryService
from rag_engine.generation.verify import CitationVerifier, Claim

LIMITS = "The upload limit is 10 MiB per file. Larger files are rejected with HTTP 413."
OTHER = "Nothing about limits is written here, only a note on the retry schedule."


def blocks() -> tuple[ContextBlock, ...]:
    return (
        ContextBlock(1, "d1", "Limits", "confluence", LIMITS),
        ContextBlock(2, "d2", "Retries", "slack", OTHER),
        ContextBlock(3, "d3", "More", "slack", "The retry schedule doubles every attempt."),
    )


def probe(
    markers: tuple[int, ...] = (1,), text: str = "The upload limit is 10 MiB per file."
) -> Probe:
    return Probe("q1", Claim(text, markers, "verified"), blocks())


def test_a_citation_is_moved_to_another_passage_of_the_same_question() -> None:
    damaged, ambiguous = move_citation(probe(), random.Random(0))

    assert damaged is not None
    statement, markers = damaged
    assert statement == "The upload limit is 10 MiB per file."
    assert markers[0] in (2, 3)
    assert not ambiguous  # neither passage holds the 10


def test_a_moved_citation_is_ambiguous_when_the_other_passage_holds_every_value() -> None:
    twin = Probe(
        "q1",
        Claim("The limit is 10 MiB.", (1,), "verified"),
        (
            ContextBlock(1, "d1", "A", "slack", "The limit is 10 MiB."),
            ContextBlock(2, "d2", "B", "slack", "Also the limit: 10 MiB, confirmed."),
        ),
    )

    damaged, ambiguous = move_citation(twin, random.Random(0))

    assert damaged is not None and ambiguous


def test_statements_with_two_citations_or_nowhere_to_move_are_left_alone() -> None:
    assert move_citation(probe(markers=(1, 2)), random.Random(0)) == (None, False)
    alone = Probe("q1", Claim("The limit is 10 MiB.", (1,), "verified"), blocks()[:1])
    assert move_citation(alone, random.Random(0)) == (None, False)


def test_a_number_is_replaced_by_one_the_cited_passage_does_not_contain() -> None:
    damaged, _ = change_number(probe(), random.Random(0))

    assert damaged is not None
    statement, markers = damaged
    assert "10" not in statement.split() and markers == (1,)
    assert re.search(r"\d+", statement) and statement != "The upload limit is 10 MiB per file."


def test_a_statement_without_numbers_cannot_have_one_changed() -> None:
    assert change_number(probe(text="Larger files are rejected."), random.Random(0)) == (
        None,
        False,
    )


class RuleJudge:
    """Writes a fixed answer, and judges by how many of a statement's words its passages hold."""

    model = "claude-haiku-5-5"

    async def complete(self, system: str, user: str, **kwargs: object) -> LLMReply:
        usage = LLMUsage(100, 10)
        if system == SYSTEM_PROMPT:
            answer = "The upload limit is 10 MiB per file [1]. Larger files are rejected [1]."
            return LLMReply(answer, usage, self.model)
        verdicts = []
        for number, chunk in enumerate(user.split("\n\n-----\n\n"), start=1):
            statement = re.search(r"Statement \d+: (.*)", chunk).group(1)  # type: ignore[union-attr]
            passages = chunk.split("Cited passages:")[1]
            passage = passages.lower()
            words = [w for w in re.findall(r"[a-z]{4,}", statement.lower())]
            held = sum(w in passage for w in words) / max(len(words), 1)
            quote = re.search(r"\)\n(.{20,60})", passages).group(1)  # type: ignore[union-attr]
            verdicts.append(
                {"claim": number, "support": "yes" if held >= 0.6 else "no", "quote": quote}
            )
        return LLMReply(json.dumps({"verdicts": verdicts}), usage, self.model)


async def test_the_check_plants_damage_and_reports_what_was_caught() -> None:
    judge = RuleJudge()
    hits = [make_hit("a", doc_id="d1", context=LIMITS), make_hit("b", doc_id="d2", context=OTHER)]
    generator = AnswerGenerator(FakeRetriever(hits), judge, chunks=2)
    verifier = CitationVerifier(judge)
    questions = [
        Question(f"q{i}", "basic", f"question {i}?", ("d1",), (), "", ()) for i in range(3)
    ]

    report = await check_verification(questions, QueryService(generator, verifier), verifier)

    assert report.answers == 3 and report.answers_all_verified == 3
    assert report.statuses["verified"] == 6
    moved, changed = report.outcomes
    assert (moved.kind, moved.planted, moved.flagged) == ("wrong citation", 6, 6)
    assert moved.judge_flagged == 6  # the judge alone saw the passage was the wrong one
    assert (changed.kind, changed.planted, changed.flagged) == ("wrong number", 3, 3)
    assert changed.judge_flagged == 0  # this judge missed it: the plain check caught it
    assert report.sample and set(report.sample[0]) >= {"statement", "evidence", "passages"}
    assert report.flagged == []  # everything verified, so there is nothing to read by hand
    text = render(report)
    assert "wrong citation" in text and "wrong number" in text and "100%" in text
