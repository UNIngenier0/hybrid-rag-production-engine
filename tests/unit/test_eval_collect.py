from pathlib import Path

from fakes import FakeEmbedder, FakeSearcher, FakeSparseEncoder, make_hit
from rag_engine.evaluation.candidates import collect_candidates, load_candidates
from rag_engine.evaluation.questions import Question


def question(question_id: str, text: str = "what is it?") -> Question:
    return Question(question_id, "basic", text, ("d",), (), "", ())


async def test_every_question_is_searched_by_both_retrievers(tmp_path: Path) -> None:
    searcher = FakeSearcher(dense=[make_hit("a"), make_hit("b")], sparse=[make_hit("c")])
    out = tmp_path / "candidates.jsonl.gz"

    searched = await collect_candidates(
        [question("q1"), question("q2")],
        searcher,
        FakeEmbedder(),
        FakeSparseEncoder(),
        out,
        depth=10,
    )

    candidates = load_candidates(out)
    assert searched == 2
    assert set(candidates) == {"q1", "q2"}
    assert [hit.point_id for hit in candidates["q1"].dense] == ["a", "b"]
    assert [hit.point_id for hit in candidates["q1"].bm25] == ["c"]
    assert sorted(kind for kind, _, _ in searcher.calls) == ["dense", "dense", "sparse", "sparse"]


async def test_the_depth_limits_how_many_chunks_are_kept(tmp_path: Path) -> None:
    hits = [make_hit(str(i)) for i in range(20)]
    out = tmp_path / "candidates.jsonl.gz"

    await collect_candidates(
        [question("q1")],
        FakeSearcher(dense=hits, sparse=hits),
        FakeEmbedder(),
        FakeSparseEncoder(),
        out,
        depth=7,
    )

    candidates = load_candidates(out)["q1"]
    assert len(candidates.dense) == len(candidates.bm25) == 7


async def test_a_second_run_only_searches_what_is_missing(tmp_path: Path) -> None:
    out = tmp_path / "candidates.jsonl.gz"
    first = FakeSearcher(dense=[make_hit("a")])
    await collect_candidates(
        [question("q1")], first, FakeEmbedder(), FakeSparseEncoder(), out, depth=5
    )
    second = FakeSearcher(dense=[make_hit("a")])

    searched = await collect_candidates(
        [question("q1"), question("q2")], second, FakeEmbedder(), FakeSparseEncoder(), out, depth=5
    )

    assert searched == 1
    assert len(second.calls) == 2  # q2 only: one dense and one sparse search
    assert set(load_candidates(out)) == {"q1", "q2"}


async def test_the_question_text_is_what_gets_embedded(tmp_path: Path) -> None:
    searcher = FakeSearcher()
    embedder = FakeEmbedder()

    await collect_candidates(
        [question("q1", "rotate the signing key")],
        searcher,
        embedder,
        FakeSparseEncoder(),
        tmp_path / "c.jsonl.gz",
        depth=5,
    )

    assert searcher.dense_vectors == [embedder.embed_query("rotate the signing key").tolist()]
    assert searcher.sparse_vectors == [FakeSparseEncoder().encode_query("rotate the signing key")]


async def test_progress_is_reported(tmp_path: Path) -> None:
    seen: list[tuple[int, int]] = []

    await collect_candidates(
        [question("q1"), question("q2"), question("q3")],
        FakeSearcher(),
        FakeEmbedder(),
        FakeSparseEncoder(),
        tmp_path / "c.jsonl.gz",
        depth=5,
        on_progress=lambda done, total: seen.append((done, total)),
    )

    assert seen == [(1, 3), (2, 3), (3, 3)]
