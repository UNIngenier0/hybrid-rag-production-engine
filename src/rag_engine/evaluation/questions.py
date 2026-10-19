"""EnterpriseRAG-Bench questions and the way they are sampled."""

import ast
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


@dataclass(frozen=True, slots=True)
class Question:
    """One benchmark question. `gold_doc_ids` is empty for questions without source documents,
    such as the ones whose right answer is that the information is not in the corpus."""

    question_id: str
    question_type: str
    question: str
    gold_doc_ids: tuple[str, ...]
    source_types: tuple[str, ...]
    gold_answer: str
    answer_facts: tuple[str, ...]


def load_questions(path: Path) -> list[Question]:
    rows = pq.read_table(path).to_pylist()
    return [
        Question(
            question_id=row["question_id"],
            question_type=row["question_type"],
            question=row["question"],
            gold_doc_ids=_as_tuple(row.get("expected_doc_ids")),
            source_types=_as_tuple(row.get("source_types")),
            gold_answer=row.get("gold_answer") or "",
            answer_facts=_as_tuple(row.get("answer_facts")),
        )
        for row in rows
    ]


def with_gold_documents(questions: Sequence[Question]) -> list[Question]:
    """The questions that retrieval can be scored on."""
    return [question for question in questions if question.gold_doc_ids]


def stratified_subset(
    questions: Sequence[Question], per_type: int, *, seed: int = 0
) -> list[Question]:
    """Up to `per_type` questions of every type, chosen the same way for the same seed."""
    by_type: dict[str, list[Question]] = defaultdict(list)
    for question in questions:
        by_type[question.question_type].append(question)
    rng = random.Random(seed)
    chosen: list[Question] = []
    for question_type in sorted(by_type):
        group = sorted(by_type[question_type], key=lambda question: question.question_id)
        chosen.extend(rng.sample(group, min(per_type, len(group))))
    return sorted(chosen, key=lambda question: question.question_id)


def _as_tuple(value: Any) -> tuple[str, ...]:
    """The dataset stores lists either as real lists or as their Python text form."""
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ()
        items = ast.literal_eval(text) if text.startswith("[") else [text]
        return tuple(str(item) for item in items)
    return tuple(str(item) for item in value)
