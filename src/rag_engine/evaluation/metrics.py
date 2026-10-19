"""Retrieval metrics at the level the benchmark scores: documents, not chunks.

A retriever returns chunks, several of which may come from one document. The benchmark's gold
labels are documents, so a ranking of chunks is first reduced to the distinct documents in the
order their best chunk appears.
"""

import math
from collections import defaultdict
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class QuestionScore:
    recall: float  # share of the gold documents that were found
    hit: bool  # at least one gold document was found
    all_gold: bool  # every gold document was found
    reciprocal_rank: float  # 1 / rank of the first gold document, 0 when none was found
    ndcg: float  # binary-relevance nDCG over the first `ndcg_k` documents


@dataclass(frozen=True, slots=True)
class Summary:
    n: int
    recall: float
    hit: float
    all_gold: float
    mrr: float
    ndcg: float


@dataclass(frozen=True, slots=True)
class PairedDifference:
    """Mean of `a - b` over the same questions, with a 95% interval (normal approximation)."""

    mean: float
    low: float
    high: float
    n: int

    @property
    def significant(self) -> bool:
        return self.low > 0 or self.high < 0


def distinct_documents(doc_ids: Iterable[str]) -> list[str]:
    """Documents in the order of their first chunk."""
    return list(dict.fromkeys(doc_ids))


def score_question(
    gold: Collection[str], ranked_docs: Sequence[str], *, ndcg_k: int = 10
) -> QuestionScore:
    """Score one ranking. `ranked_docs` must list each document once, best first."""
    gold_set = set(gold)
    if not gold_set:
        raise ValueError("a question without gold documents cannot be scored")
    found = gold_set.intersection(ranked_docs)
    first = next((rank for rank, doc in enumerate(ranked_docs, start=1) if doc in gold_set), None)
    gain = sum(
        1 / math.log2(rank + 1)
        for rank, doc in enumerate(ranked_docs[:ndcg_k], start=1)
        if doc in gold_set
    )
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(gold_set), ndcg_k) + 1))
    return QuestionScore(
        recall=len(found) / len(gold_set),
        hit=bool(found),
        all_gold=len(found) == len(gold_set),
        reciprocal_rank=0.0 if first is None else 1 / first,
        ndcg=gain / ideal,
    )


def summarize(scores: Sequence[QuestionScore]) -> Summary:
    n = len(scores)
    if n == 0:
        return Summary(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return Summary(
        n=n,
        recall=sum(s.recall for s in scores) / n,
        hit=sum(s.hit for s in scores) / n,
        all_gold=sum(s.all_gold for s in scores) / n,
        mrr=sum(s.reciprocal_rank for s in scores) / n,
        ndcg=sum(s.ndcg for s in scores) / n,
    )


def summarize_by_type(
    scores: Sequence[tuple[str, QuestionScore]],
) -> dict[str, Summary]:
    grouped: dict[str, list[QuestionScore]] = defaultdict(list)
    for question_type, score in scores:
        grouped[question_type].append(score)
    return {question_type: summarize(group) for question_type, group in sorted(grouped.items())}


def paired_difference(a: Sequence[float], b: Sequence[float]) -> PairedDifference:
    if len(a) != len(b):
        raise ValueError("both systems must be scored on the same questions")
    n = len(a)
    if n == 0:
        return PairedDifference(0.0, 0.0, 0.0, 0)
    differences = [x - y for x, y in zip(a, b, strict=True)]
    mean = sum(differences) / n
    if n < 2:
        return PairedDifference(mean, mean, mean, n)
    variance = sum((d - mean) ** 2 for d in differences) / (n - 1)
    margin = 1.96 * math.sqrt(variance / n)
    return PairedDifference(mean, mean - margin, mean + margin, n)
