import pytest

from fakes import FakePassageStore, make_hit
from rag_engine.core.models import SearchFilters
from rag_engine.evaluation.candidates import QuestionCandidates
from rag_engine.evaluation.questions import Question
from rag_engine.evaluation.stored import StoredRetriever


def question(question_id: str, text: str) -> Question:
    return Question(question_id, "basic", text, ("d",), (), "", ())


def build(scores=None, passages=None, **options) -> StoredRetriever:  # type: ignore[no-untyped-def]
    candidates = {
        "q1": QuestionCandidates(
            "q1",
            dense=[make_hit("a"), make_hit("b"), make_hit("c")],
            bm25=[make_hit("c"), make_hit("d")],
        )
    }
    return StoredRetriever(
        [question("q1", "how does it work?")], candidates, scores, passages, **options
    )


def ids(result) -> list[str]:  # type: ignore[no-untyped-def]
    return [hit.point_id for hit in result.hits]


async def test_hybrid_replays_the_fusion_of_the_stored_lists() -> None:
    result = await build().retrieve("how does it work?", mode="hybrid", limit=10)

    assert ids(result)[0] == "c"  # found by both retrievers
    assert set(ids(result)) == {"a", "b", "c", "d"}


async def test_single_retriever_modes_replay_one_list() -> None:
    retriever = build()

    assert ids(await retriever.retrieve("how does it work?", mode="dense")) == ["a", "b", "c"]
    assert ids(await retriever.retrieve("how does it work?", mode="bm25")) == ["c", "d"]


async def test_reranking_uses_the_stored_scores() -> None:
    scores = {"q1": {"a": 1.0, "b": 9.0, "c": 5.0, "d": 3.0}}

    result = await build(scores).retrieve("how does it work?", mode="hybrid+rerank", limit=3)

    assert ids(result) == ["b", "c", "d"]


async def test_reranking_without_scores_is_an_error() -> None:
    with pytest.raises(ValueError, match="scores"):
        await build().retrieve("how does it work?", mode="hybrid+rerank")


async def test_the_limit_caps_the_hits() -> None:
    result = await build().retrieve("how does it work?", mode="hybrid", limit=2)

    assert len(result.hits) == 2


async def test_parent_passages_come_from_the_passage_store() -> None:
    result = await build(passages=FakePassageStore()).retrieve(
        "how does it work?", mode="dense", include_context=True
    )

    assert result.hits[0].context == "doc-a[0:40]"


async def test_a_question_that_was_not_collected_is_an_error() -> None:
    with pytest.raises(KeyError):
        await build().retrieve("something else?")


async def test_stored_candidates_cannot_be_filtered() -> None:
    with pytest.raises(ValueError, match="filtered"):
        await build().retrieve("how does it work?", filters=SearchFilters(source_types=("slack",)))
