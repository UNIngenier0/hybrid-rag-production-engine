from fakes import FakeLLM, FakeRetriever, make_hit
from rag_engine.generation.answer import AnswerGenerator
from rag_engine.generation.citations import cited_numbers, resolve, strip_markers
from rag_engine.generation.prompt import SYSTEM_PROMPT, build_context, render_prompt


def test_markers_are_read_in_order_of_first_appearance() -> None:
    assert cited_numbers("A [2]. B [1][3] and [2] again, then [4, 1].") == [2, 1, 3, 4]


def test_text_without_markers_cites_nothing() -> None:
    assert cited_numbers("No sources here, but an [x] and a [] and [1.5].") == []


def test_stripping_removes_the_markers_and_closes_the_gaps() -> None:
    assert strip_markers("The limit is 10 MiB [1]. Total is 50 MiB [2][3].") == (
        "The limit is 10 MiB. Total is 50 MiB."
    )
    assert strip_markers("Both agree [1, 2], so use it.") == "Both agree, so use it."


def test_stripping_keeps_line_breaks() -> None:
    assert strip_markers("- one [1]\n- two [2]") == "- one\n- two"


def test_numbers_map_to_documents_and_unknown_numbers_are_reported() -> None:
    documents, unresolved = resolve([2, 1, 9, 2], ["doc-a", "doc-b", "doc-c"])

    assert documents == ["doc-b", "doc-a"]
    assert unresolved == [9]


def test_two_passages_of_one_document_cite_that_document_once() -> None:
    documents, _ = resolve([1, 2], ["doc-a", "doc-a"])

    assert documents == ["doc-a"]


def test_passages_are_numbered_in_rank_order_with_the_parent_text() -> None:
    hits = [
        make_hit("a", doc_id="d1", context="parent of a", title="Runbook"),
        make_hit("b", doc_id="d2", text="just a chunk", context=None, source_type="slack"),
    ]

    blocks = build_context(hits)

    assert [(b.number, b.doc_id, b.text) for b in blocks] == [
        (1, "d1", "parent of a"),
        (2, "d2", "just a chunk"),
    ]


def test_chunks_that_share_a_parent_passage_are_shown_once() -> None:
    shared = {"doc_id": "d1", "parent_start": 0, "parent_end": 100, "context": "the parent"}
    hits = [
        make_hit("a", **shared),
        make_hit("b", **shared),
        make_hit("c", doc_id="d2", context="x"),
    ]

    blocks = build_context(hits)

    assert [b.doc_id for b in blocks] == ["d1", "d2"]
    assert [b.number for b in blocks] == [1, 2]


def test_the_prompt_lists_every_passage_and_ends_with_the_question() -> None:
    blocks = build_context([make_hit("a", doc_id="d1", title="Runbook", context="Rotate keys.")])

    prompt = render_prompt("How do I rotate keys?", blocks)

    assert prompt.startswith("Passages:")
    assert "[1] Runbook (confluence)\nRotate keys." in prompt
    assert prompt.endswith("Question: How do I rotate keys?")


def test_the_instructions_ask_for_citations_and_an_honest_not_found() -> None:
    assert "square brackets" in SYSTEM_PROMPT
    assert "could not find it" in SYSTEM_PROMPT


async def test_the_answer_reports_the_documents_its_citations_point_at() -> None:
    hits = [
        make_hit("a", doc_id="d1", context="Limit is 10 MiB."),
        make_hit("b", doc_id="d2", context="Total is 50 MiB."),
        make_hit("c", doc_id="d3", context="Unrelated."),
    ]
    retriever = FakeRetriever(hits)
    llm = FakeLLM("Per file 10 MiB [1], total 50 MiB [2][7].")

    answer = await AnswerGenerator(retriever, llm, chunks=3).answer("What are the limits?")

    assert answer.text == "Per file 10 MiB, total 50 MiB."
    assert answer.raw_text == "Per file 10 MiB [1], total 50 MiB [2][7]."
    assert answer.cited_doc_ids == ["d1", "d2"]
    assert answer.retrieved_doc_ids == ["d1", "d2", "d3"]
    assert answer.unresolved_markers == [7]
    assert (answer.usage.input_tokens, answer.model) == (1000, "claude-haiku-5-5")


async def test_the_retriever_is_asked_for_passages_with_their_parent_context() -> None:
    retriever = FakeRetriever([make_hit("a")])

    await AnswerGenerator(retriever, FakeLLM("x [1]"), mode="hybrid", chunks=4).answer("question?")

    assert retriever.calls == [
        {"query": "question?", "mode": "hybrid", "limit": 4, "include_context": True}
    ]


async def test_the_model_is_shown_the_numbered_passages_and_the_question() -> None:
    llm = FakeLLM("done [1]")

    await AnswerGenerator(FakeRetriever([make_hit("a", context="Fact one.")]), llm).answer("Why?")

    system, user = llm.prompts[0]
    assert system == SYSTEM_PROMPT
    assert "[1] Title (confluence)\nFact one." in user
    assert user.endswith("Question: Why?")


async def test_an_answer_with_no_citations_reports_no_cited_documents() -> None:
    answer = await AnswerGenerator(
        FakeRetriever([make_hit("a", doc_id="d1")]),
        FakeLLM("I could not find it in the documents."),
    ).answer("q?")

    assert answer.cited_doc_ids == []
    assert answer.retrieved_doc_ids == ["d1"]


async def test_no_temperature_is_sent_by_default_because_newer_models_refuse_one() -> None:
    llm = FakeLLM("x [1]")

    await AnswerGenerator(FakeRetriever([make_hit("a")]), llm).answer("q?")

    assert llm.temperatures == [None]


def test_the_instructions_keep_markers_out_of_the_sentence() -> None:
    assert "end of the sentence" in SYSTEM_PROMPT
    assert "Never use a marker as a word" in SYSTEM_PROMPT


async def test_a_reply_with_no_text_is_a_failure_not_an_empty_answer() -> None:
    import pytest

    from rag_engine.core.errors import LLMError

    with pytest.raises(LLMError, match="no text"):
        await AnswerGenerator(FakeRetriever([make_hit("a")]), FakeLLM("  \n")).answer("q?")
