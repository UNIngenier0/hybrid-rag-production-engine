from pathlib import Path

from fakes import FakeReranker, make_hit
from rag_engine.evaluation.candidates import QuestionCandidates, load_candidates
from rag_engine.evaluation.questions import Question
from rag_engine.evaluation.rerank_scores import (
    export_pools,
    load_scores,
    score_pools,
    scoring_speed,
)
from rag_engine.evaluation.variants import fused_ranking


def question(question_id: str) -> Question:
    return Question(question_id, "basic", f"query {question_id}", ("d",), (), "", ())


def candidates(question_id: str) -> QuestionCandidates:
    hits = [make_hit(f"{question_id}{i}") for i in range(6)]
    return QuestionCandidates(question_id, hits, hits[::-1])


def test_the_fused_pool_of_every_question_is_scored(tmp_path: Path) -> None:
    out = tmp_path / "scores.jsonl.gz"
    reranker = FakeReranker({"text of q10": 7.0})

    scored = score_pools(
        [question("q1")], {"q1": candidates("q1")}, reranker, out, pool=4, on_progress=None
    )

    scores = load_scores(out)["q1"]
    assert scored == 1
    assert len(scores) == 4
    assert scores["q10"] == 7.0
    assert reranker.calls[0][0] == "query q1"
    assert len(reranker.calls[0][1]) == 4


def test_questions_already_scored_are_skipped_on_the_next_run(tmp_path: Path) -> None:
    out = tmp_path / "scores.jsonl.gz"
    both = {"q1": candidates("q1"), "q2": candidates("q2")}
    score_pools([question("q1")], both, FakeReranker({}), out, pool=3)
    second = FakeReranker({})

    scored = score_pools([question("q1"), question("q2")], both, second, out, pool=3)

    assert scored == 1
    assert [call[0] for call in second.calls] == ["query q2"]
    assert set(load_scores(out)) == {"q1", "q2"}


def test_questions_without_candidates_are_left_out(tmp_path: Path) -> None:
    out = tmp_path / "scores.jsonl.gz"

    scored = score_pools([question("q1")], {}, FakeReranker({}), out)

    assert scored == 0
    assert load_scores(out) == {}


def test_the_scoring_speed_is_recorded_per_pair(tmp_path: Path) -> None:
    out = tmp_path / "scores.jsonl.gz"

    score_pools([question("q1")], {"q1": candidates("q1")}, FakeReranker({}), out, pool=5)

    model, ms_per_pair, device = scoring_speed(out)
    assert model == "fake-reranker"
    assert ms_per_pair >= 0.0
    assert device == "unknown"  # the fake has no device


def test_progress_is_reported(tmp_path: Path) -> None:
    seen: list[tuple[int, int]] = []
    both = {"q1": candidates("q1"), "q2": candidates("q2")}

    score_pools(
        [question("q1"), question("q2")],
        both,
        FakeReranker({}),
        tmp_path / "s.jsonl.gz",
        on_progress=lambda done, total: seen.append((done, total)),
    )

    assert seen == [(1, 2), (2, 2)]


def test_an_exported_pool_is_scored_exactly_like_the_full_candidates(tmp_path: Path) -> None:
    both = {"q1": candidates("q1"), "q2": candidates("q2")}
    asked = [question("q1"), question("q2")]
    pool_file = tmp_path / "pool.jsonl.gz"

    exported = export_pools(asked, both, pool_file, pool=4)
    from_pool = tmp_path / "from-pool.jsonl.gz"
    from_full = tmp_path / "from-full.jsonl.gz"
    reranker = FakeReranker({"text of q11": 3.0, "text of q23": 5.0})
    score_pools(asked, load_candidates(pool_file), reranker, from_pool, pool=4)
    score_pools(asked, both, reranker, from_full, pool=4)

    assert exported == 2
    assert load_scores(from_pool) == load_scores(from_full)
    assert [h.point_id for h in fused_ranking(load_candidates(pool_file)["q1"])] == [
        h.point_id for h in fused_ranking(both["q1"])[:4]
    ]


def test_exporting_again_replaces_the_file(tmp_path: Path) -> None:
    pool_file = tmp_path / "pool.jsonl.gz"
    both = {"q1": candidates("q1"), "q2": candidates("q2")}

    export_pools([question("q1"), question("q2")], both, pool_file)
    export_pools([question("q1")], both, pool_file)

    assert set(load_candidates(pool_file)) == {"q1"}
