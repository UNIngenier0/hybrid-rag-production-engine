import json
import subprocess
from pathlib import Path

import pytest

from fakes import FakeLLM
from rag_engine.evaluation import grep_baseline
from rag_engine.evaluation.grep_baseline import (
    FileMatch,
    GrepRetriever,
    SearchPlan,
    densest_window,
    load_document,
    parse_plan,
    plan_search,
    search_corpus,
)
from rag_engine.generation.answer import AnswerGenerator


def write_doc(
    corpus: Path, source: str, name: str, doc_id: str, text: str, title: str = "T"
) -> Path:
    path = corpus / source / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "channel": title,
                "messages": text,
                "title_field_name": "channel",
                "content_field_names": "['messages']",
                "dataset_doc_uuid": doc_id,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_a_plan_is_read_from_json_even_inside_a_fence_or_prose() -> None:
    plan = parse_plan(
        'Here you go:\n```json\n{"pattern": "mtu|clamp", "sources": ["slack", "nope"]}\n```'
    )

    assert plan == SearchPlan("mtu|clamp", ("slack",))  # unknown sources are dropped


@pytest.mark.parametrize("text", ["no json here", '{"sources": []}', '{"pattern": "  "}'])
def test_a_plan_without_a_pattern_is_refused(text: str) -> None:
    with pytest.raises(ValueError):
        parse_plan(text)


async def test_the_model_is_asked_for_one_pattern_about_the_question() -> None:
    llm = FakeLLM('{"pattern": "kv cache", "sources": []}')

    plan = await plan_search(llm, "How big is the KV cache?")

    assert plan.pattern == "kv cache"
    system, user = llm.prompts[0]
    assert "exactly one try" in system
    assert user == "Question: How big is the KV cache?"


def test_the_python_search_counts_matches_and_puts_the_richest_file_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(grep_baseline.shutil, "which", lambda name: None)
    one = write_doc(tmp_path, "slack", "a", "dsid_a", "mtu once")
    three = write_doc(tmp_path, "jira", "b", "dsid_b", "MTU mtu mtu")
    write_doc(tmp_path, "slack", "c", "dsid_c", "nothing relevant")

    found = search_corpus(SearchPlan("mtu", ()), tmp_path)

    assert found == [FileMatch(three, 3), FileMatch(one, 1)]


def test_the_search_can_be_limited_to_some_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(grep_baseline.shutil, "which", lambda name: None)
    write_doc(tmp_path, "slack", "a", "dsid_a", "mtu")
    jira = write_doc(tmp_path, "jira", "b", "dsid_b", "mtu")

    found = search_corpus(SearchPlan("mtu", ("jira",)), tmp_path)

    assert found == [FileMatch(jira, 1)]


def test_ripgrep_output_is_read_as_path_and_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(grep_baseline.shutil, "which", lambda name: "/usr/bin/rg")
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        out = f"{tmp_path}/slack/a.json:2\n{tmp_path}/jira/b.json:9\nnot a count line\n"
        return subprocess.CompletedProcess(command, 0, out, "")

    monkeypatch.setattr(grep_baseline.subprocess, "run", fake_run)

    found = search_corpus(SearchPlan("mtu", ()), tmp_path)

    assert [(m.path.name, m.matches) for m in found] == [("b.json", 9), ("a.json", 2)]
    assert "--ignore-case" in calls[0] and "--regexp" in calls[0]


def test_a_document_is_read_with_its_id_title_source_and_content(tmp_path: Path) -> None:
    path = write_doc(tmp_path, "slack", "a", "dsid_a", "the thread", title="eng-sre")

    document = load_document(path, tmp_path)

    assert (document.doc_id, document.title, document.source_type) == ("dsid_a", "eng-sre", "slack")
    assert document.text == "the thread"


def test_content_fields_that_are_lists_are_joined(tmp_path: Path) -> None:
    path = tmp_path / "jira" / "x.json"
    path.parent.mkdir()
    path.write_text(
        json.dumps(
            {
                "summary": "S",
                "comments": ["one", "two"],
                "description": "D",
                "title_field_name": "summary",
                "content_field_names": ["description", "comments"],
                "dataset_doc_uuid": "dsid_x",
            }
        ),
        encoding="utf-8",
    )

    assert load_document(path, tmp_path).text == "description:\nD\n\ncomments:\none\ntwo"


def test_a_short_text_is_returned_whole_and_a_long_one_around_its_matches() -> None:
    assert densest_window("short mtu text", "mtu", size=100) == "short mtu text"
    text = "x" * 5000 + "mtu mtu mtu clamp" + "y" * 5000

    window = densest_window(text, "mtu", size=300)

    assert len(window) == 300
    assert window.count("mtu") == 3


def test_text_without_matches_or_with_a_bad_pattern_starts_at_the_top() -> None:
    text = "start " + "z" * 5000

    assert densest_window(text, "absent", size=100).startswith("start")
    assert densest_window(text, "(unclosed", size=100).startswith("start")


async def test_the_retriever_plans_once_searches_once_and_returns_the_richest_files(
    tmp_path: Path,
) -> None:
    docs = [
        write_doc(tmp_path, "slack", f"d{i}", f"dsid_{i}", f"text {i} mtu", title=f"T{i}")
        for i in range(4)
    ]
    searches: list[SearchPlan] = []

    def searcher(plan: SearchPlan, corpus: Path) -> list[FileMatch]:
        searches.append(plan)
        return [FileMatch(docs[3], 9), FileMatch(docs[1], 5), FileMatch(docs[0], 1)]

    llm = FakeLLM('{"pattern": "mtu", "sources": ["slack"]}')
    retriever = GrepRetriever(llm, tmp_path, files=2, searcher=searcher)

    result = await retriever.retrieve("what about mtu?")

    assert [hit.doc_id for hit in result.hits] == ["dsid_3", "dsid_1"]
    assert result.hits[0].source_type == "slack" and result.hits[0].title == "T3"
    assert len(searches) == 1 and len(llm.prompts) == 1
    assert retriever.plans["what about mtu?"] == SearchPlan("mtu", ("slack",))


async def test_a_plan_the_model_cannot_write_spends_the_single_try() -> None:
    retriever = GrepRetriever(FakeLLM("I cannot do that."), Path("/nowhere"))

    result = await retriever.retrieve("anything?")

    assert result.hits == []
    assert retriever.plans["anything?"] is None


async def test_the_answer_step_is_the_same_one_every_system_uses(tmp_path: Path) -> None:
    doc = write_doc(tmp_path, "slack", "a", "dsid_a", "The clamp is 1500 bytes.")
    llm = FakeLLM('{"pattern": "clamp", "sources": []}', "The clamp is 1500 bytes [1].")
    retriever = GrepRetriever(llm, tmp_path, searcher=lambda plan, corpus: [FileMatch(doc, 1)])

    answer = await AnswerGenerator(retriever, llm, mode="hybrid", chunks=5).answer("Clamp?")

    assert answer.text == "The clamp is 1500 bytes."
    assert answer.cited_doc_ids == ["dsid_a"]
    assert "The clamp is 1500 bytes." in llm.prompts[1][1]


async def test_a_search_that_never_finishes_counts_as_finding_nothing(tmp_path: Path) -> None:
    def searcher(plan: SearchPlan, corpus: Path) -> list[FileMatch]:
        raise subprocess.TimeoutExpired("rg", 300)

    llm = FakeLLM('{"pattern": "redwood", "sources": []}')
    retriever = GrepRetriever(llm, tmp_path, searcher=searcher)

    result = await retriever.retrieve("what is redwood?")

    assert result.hits == []
    assert retriever.plans["what is redwood?"] == SearchPlan("redwood", ())
