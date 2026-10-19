from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from rag_engine.evaluation.questions import (
    Question,
    load_questions,
    stratified_subset,
    with_gold_documents,
)


def write_questions(path: Path, rows: list[dict[str, object]]) -> Path:
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def make_question(question_id: str, question_type: str, gold: tuple[str, ...] = ("d",)) -> Question:
    return Question(question_id, question_type, "q?", gold, (), "", ())


def test_lists_are_read_from_their_text_form(tmp_path: Path) -> None:
    path = write_questions(
        tmp_path / "q.parquet",
        [
            {
                "question_id": "q1",
                "question_type": "basic",
                "question": "What?",
                "expected_doc_ids": "['dsid_a', 'dsid_b']",
                "source_types": "['slack']",
                "gold_answer": "An answer",
                "answer_facts": "['one fact']",
            }
        ],
    )

    (question,) = load_questions(path)

    assert question.gold_doc_ids == ("dsid_a", "dsid_b")
    assert question.source_types == ("slack",)
    assert question.answer_facts == ("one fact",)
    assert question.gold_answer == "An answer"


def test_lists_are_read_from_real_lists(tmp_path: Path) -> None:
    path = write_questions(
        tmp_path / "q.parquet",
        [
            {
                "question_id": "q2",
                "question_type": "semantic",
                "question": "Why?",
                "expected_doc_ids": ["dsid_c"],
                "source_types": ["jira", "github"],
                "gold_answer": None,
                "answer_facts": ["a", "b"],
            }
        ],
    )

    (question,) = load_questions(path)

    assert question.gold_doc_ids == ("dsid_c",)
    assert question.source_types == ("jira", "github")
    assert question.answer_facts == ("a", "b")
    assert question.gold_answer == ""


def test_questions_without_gold_documents_are_left_out_of_retrieval_scoring(tmp_path: Path) -> None:
    path = write_questions(
        tmp_path / "q.parquet",
        [
            {
                "question_id": "q1",
                "question_type": "basic",
                "question": "a",
                "expected_doc_ids": "['d']",
            },
            {
                "question_id": "q2",
                "question_type": "info_not_found",
                "question": "b",
                "expected_doc_ids": "[]",
            },
            {
                "question_id": "q3",
                "question_type": "high_level",
                "question": "c",
                "expected_doc_ids": None,
            },
        ],
    )

    scored = with_gold_documents(load_questions(path))

    assert [q.question_id for q in scored] == ["q1"]


def test_the_subset_takes_up_to_n_of_every_type_and_is_repeatable() -> None:
    questions = [make_question(f"a{i}", "basic") for i in range(20)]
    questions += [make_question(f"b{i}", "semantic") for i in range(3)]

    first = stratified_subset(questions, 5, seed=1)
    again = stratified_subset(questions, 5, seed=1)

    assert first == again
    assert [q.question_type for q in first].count("basic") == 5
    assert [q.question_type for q in first].count("semantic") == 3  # fewer than asked: all of them
    assert first == sorted(first, key=lambda q: q.question_id)


def test_a_different_seed_picks_different_questions() -> None:
    questions = [make_question(f"a{i:02d}", "basic") for i in range(40)]

    assert stratified_subset(questions, 5, seed=1) != stratified_subset(questions, 5, seed=2)
