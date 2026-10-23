from fakes import FakeLLM, FakeRetriever, make_hit
from rag_engine.core.models import ChunkHit
from rag_engine.generation.abstain import MESSAGE, AbstentionPolicy
from rag_engine.generation.answer import AnswerGenerator
from rag_engine.generation.query import QueryService
from rag_engine.generation.verify import CitationVerifier


def scored(*scores: float) -> list[ChunkHit]:
    return [make_hit(f"p{i}", doc_id=f"d{i}", score=score) for i, score in enumerate(scores)]


def test_a_strong_best_passage_is_answered() -> None:
    assert AbstentionPolicy(0.0).check(scored(2.5, -1.0), "hybrid+rerank") is None


def test_a_weak_best_passage_declines_and_says_by_how_much() -> None:
    declined = AbstentionPolicy(0.0).check(scored(-2.0, -5.0), "hybrid+rerank")

    assert declined is not None
    assert (declined.reason, declined.top_score, declined.threshold) == ("weak_evidence", -2.0, 0.0)
    assert declined.message == MESSAGE


def test_the_threshold_itself_is_enough_evidence() -> None:
    assert AbstentionPolicy(1.0).check(scored(1.0), "hybrid+rerank") is None


def test_the_best_score_counts_wherever_it_stands_in_the_list() -> None:
    assert AbstentionPolicy(0.0).check(scored(-3.0, 4.0), "hybrid+rerank") is None


def test_without_a_threshold_nothing_declines_on_score() -> None:
    assert AbstentionPolicy().check(scored(-99.0), "hybrid+rerank") is None


def test_scores_that_are_not_reranker_scores_are_never_compared_with_the_threshold() -> None:
    for mode in ("dense", "bm25", "hybrid"):
        assert AbstentionPolicy(5.0).check(scored(0.01), mode) is None  # type: ignore[arg-type]


def test_nothing_retrieved_declines_in_every_mode_and_with_or_without_a_threshold() -> None:
    for policy in (AbstentionPolicy(), AbstentionPolicy(0.0)):
        for mode in ("dense", "bm25", "hybrid", "hybrid+rerank"):
            declined = policy.check([], mode)  # type: ignore[arg-type]
            assert declined is not None and declined.reason == "no_passages"
            assert declined.top_score is None


async def test_a_declined_question_costs_no_model_call() -> None:
    llm = FakeLLM("never used")
    generator = AnswerGenerator(
        FakeRetriever(scored(-4.0, -6.0)),
        llm,
        mode="hybrid+rerank",
        abstention=AbstentionPolicy(0.0),
    )

    answer = await generator.answer("What is the secret?")

    assert llm.prompts == []
    assert answer.abstention is not None and answer.abstention.reason == "weak_evidence"
    assert answer.text == MESSAGE and answer.raw_text == MESSAGE
    assert answer.cited_doc_ids == []
    assert answer.retrieved_doc_ids == ["d0", "d1"]
    assert (answer.usage.input_tokens, answer.usage.output_tokens) == (0, 0)


async def test_a_question_with_enough_evidence_is_answered_as_before() -> None:
    llm = FakeLLM("The limit is 10 MiB [1].")
    generator = AnswerGenerator(
        FakeRetriever(scored(3.0)), llm, mode="hybrid+rerank", abstention=AbstentionPolicy(0.0)
    )

    answer = await generator.answer("What is the limit?")

    assert answer.abstention is None
    assert answer.cited_doc_ids == ["d0"]
    assert len(llm.prompts) == 1


async def test_a_query_that_declines_is_not_verified_and_costs_nothing() -> None:
    llm = FakeLLM("never used")
    generator = AnswerGenerator(
        FakeRetriever(scored(-4.0)), llm, mode="hybrid+rerank", abstention=AbstentionPolicy(0.0)
    )

    result = await QueryService(generator, CitationVerifier(llm)).query("What is the secret?")

    assert llm.prompts == []
    assert result.verification.method == "none" and result.verification.claims == []
    assert (result.input_tokens, result.output_tokens, result.cost_usd) == (0, 0, 0.0)
    assert result.answer.abstention is not None


async def test_a_query_that_declines_has_the_abstained_status() -> None:
    llm = FakeLLM("never used")
    generator = AnswerGenerator(
        FakeRetriever(scored(-4.0)), llm, mode="hybrid+rerank", abstention=AbstentionPolicy(0.0)
    )

    result = await QueryService(generator, CitationVerifier(llm)).query("What is the secret?")

    assert (result.coverage.status, result.coverage.gaps) == ("abstained", ())
