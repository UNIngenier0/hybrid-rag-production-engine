from fakes import make_hit
from rag_engine.evaluation.candidates import QuestionCandidates
from rag_engine.evaluation.questions import Question
from rag_engine.evaluation.report import BASELINE, RerankerRun, build_report
from rag_engine.evaluation.variants import fused_ranking


def question(question_id: str, question_type: str, gold: str) -> Question:
    return Question(question_id, question_type, "q?", (gold,), (), "", ())


def scenario() -> tuple[list[Question], dict[str, QuestionCandidates]]:
    """Two questions. In both the gold chunk is found by BM25 only, far down the fused list."""
    questions = [question("q1", "basic", "doc-gold1"), question("q2", "semantic", "doc-gold2")]
    candidates = {}
    for q, gold in (("q1", "doc-gold1"), ("q2", "doc-gold2")):
        noise = [make_hit(f"{q}n{i}", doc_id=f"doc-{q}n{i}") for i in range(8)]
        candidates[q] = QuestionCandidates(
            q, noise, [*noise[:7], make_hit(f"{q}gold", doc_id=gold)]
        )
    return questions, candidates


def perfect_reranker(candidates: dict[str, QuestionCandidates]) -> RerankerRun:
    scores = {
        q: {
            hit.point_id: (10.0 if "gold" in hit.point_id else 0.0) for hit in fused_ranking(c)[:50]
        }
        for q, c in candidates.items()
    }
    return RerankerRun("oracle", "fake/oracle", scores, ms_per_pair=2.5)


def test_the_report_compares_every_retriever_and_fusion_setting() -> None:
    questions, candidates = scenario()

    report = build_report(questions, candidates)

    assert "# Retrieval ablation" in report.markdown
    assert "2 questions with gold documents" in report.markdown
    for name in ("dense", "bm25", BASELINE, "hybrid, BM25 weight 3", "hybrid, depth 100"):
        assert f"| {name} |" in report.markdown
    assert "## 5 chunks returned" in report.markdown
    assert "## 10 chunks returned" in report.markdown
    assert "Candidate pools" in report.markdown


def test_numbers_in_the_report_are_the_measured_ones() -> None:
    questions, candidates = scenario()

    report = build_report(questions, candidates)

    dense = report.data["systems"]["dense"]["5"]["overall"]
    bm25 = report.data["systems"]["bm25"]["10"]["overall"]
    assert dense["recall"] == 0.0  # the gold chunk is not in the dense list
    assert bm25["recall"] == 1.0  # BM25 has it at rank 8, inside ten chunks
    assert report.data["systems"]["bm25"]["5"]["overall"]["recall"] == 0.0
    assert report.data["questions"] == 2


def test_a_good_reranker_shows_up_as_a_significant_gain() -> None:
    questions, candidates = scenario()

    report = build_report(questions, candidates, [perfect_reranker(candidates)])

    assert report.data["systems"]["rerank oracle, pool 30"]["5"]["overall"]["recall"] == 1.0
    assert report.data["systems"][BASELINE]["5"]["overall"]["recall"] == 0.0
    assert "| rerank oracle, pool 30 |" in report.markdown
    assert "Reranker cost" in report.markdown
    assert "fake/oracle" in report.markdown
    assert "75 ms" in report.markdown  # 30 passages at 2.5 ms each


def test_questions_without_gold_or_candidates_or_scores_are_not_scored() -> None:
    questions, candidates = scenario()
    questions.append(Question("q3", "info_not_found", "?", (), (), "", ()))  # no gold documents
    questions.append(question("q4", "basic", "doc-x"))  # never collected
    run = perfect_reranker(candidates)
    partial = RerankerRun("oracle", "fake/oracle", {"q1": run.scores["q1"]}, 1.0)

    report = build_report(questions, candidates, [partial])

    assert report.data["questions"] == 1  # only q1 has gold, candidates and scores


def test_the_type_table_lists_the_best_reranker() -> None:
    questions, candidates = scenario()

    report = build_report(questions, candidates, [perfect_reranker(candidates)])

    type_table = report.markdown.split("## Recall by question type, 5 chunks")[1]
    assert "basic (1)" in type_table and "semantic (1)" in type_table
    assert "| rerank oracle, pool" in type_table
