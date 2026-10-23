import json

import httpx
import pytest
from pydantic import SecretStr

from fakes import FakeLLM, FakeRetriever, make_hit
from rag_engine.api.main import create_app
from rag_engine.core.config import Settings
from rag_engine.core.errors import BudgetExceededError, LLMError, RetrievalUnavailableError
from rag_engine.core.models import LLMUsage, SearchFilters
from rag_engine.generation.answer import AnswerGenerator
from rag_engine.generation.query import QueryService
from rag_engine.generation.verify import CitationVerifier

PASSAGE = "The upload limit is 10 MiB per file. Larger files are rejected with HTTP 413."
ANSWER = "The upload limit is 10 MiB per file [1]. Larger files are rejected [1]."
VERDICTS = json.dumps(
    {
        "verdicts": [
            {"claim": 1, "support": "yes", "quote": "The upload limit is 10 MiB per file."},
            {"claim": 2, "support": "yes", "quote": "Larger files are rejected with HTTP 413."},
        ]
    }
)


def hits() -> list:  # type: ignore[type-arg]
    return [
        make_hit("a", doc_id="d1", title="Limits", context=PASSAGE, source_type="confluence"),
        make_hit("b", doc_id="d2", title="Other", context="Nothing about limits."),
    ]


def service(llm: FakeLLM, retriever: FakeRetriever | None = None, **kwargs: object) -> QueryService:
    retriever = retriever or FakeRetriever(hits())
    generator = AnswerGenerator(retriever, llm, mode="hybrid", chunks=2)
    return QueryService(generator, CitationVerifier(llm), **kwargs)  # type: ignore[arg-type]


async def test_a_question_gets_a_cited_answer_whose_citations_are_verified() -> None:
    llm = FakeLLM(ANSWER, VERDICTS)

    result = await service(llm).query("What is the upload limit?")

    assert result.answer.text == "The upload limit is 10 MiB per file. Larger files are rejected."
    assert result.answer.cited_doc_ids == ["d1"]
    assert result.verification.all_verified
    assert [claim.status for claim in result.verification.claims] == ["verified", "verified"]
    assert len(llm.prompts) == 2  # one call to answer, one to judge


async def test_cost_and_tokens_add_up_the_answer_and_the_judge() -> None:
    llm = FakeLLM(ANSWER, VERDICTS, usage=LLMUsage(1000, 100))

    result = await service(llm).query("What is the upload limit?")

    assert (result.input_tokens, result.output_tokens) == (2000, 200)
    assert result.cost_usd == pytest.approx(2 * (1000 * 0.10 + 100 * 0.50) / 1_000_000)


async def test_a_model_without_a_known_price_has_no_cost_but_still_answers() -> None:
    llm = FakeLLM(ANSWER, VERDICTS, model="some-new-model")

    result = await service(llm).query("What is the upload limit?")

    assert result.cost_usd is None
    assert result.verification.all_verified


async def test_every_stage_is_timed() -> None:
    result = await service(FakeLLM(ANSWER, VERDICTS)).query("What is the upload limit?")

    assert {"retrieval_total", "generation", "verification", "total"} <= set(result.timings_ms)


async def test_the_judge_is_skipped_when_the_request_does_not_want_it() -> None:
    llm = FakeLLM(ANSWER, VERDICTS)

    result = await service(llm).query("What is the upload limit?", verify=False)

    assert result.verification.method == "values"
    assert len(llm.prompts) == 1
    assert not result.verification.all_verified


async def test_the_options_of_a_request_reach_the_retriever() -> None:
    retriever = FakeRetriever(hits())
    filters = SearchFilters(source_types=("slack",))

    await service(FakeLLM(ANSWER, VERDICTS), retriever).query(
        "What is the upload limit?", mode="dense", chunks=1, filters=filters
    )

    call = retriever.calls[0]
    assert (call["mode"], call["limit"], call["filters"]) == ("dense", 1, filters)
    assert call["include_context"] is True


# --- over HTTP


def client(
    settings: Settings, query_service: QueryService | None, retriever: FakeRetriever | None = None
) -> httpx.AsyncClient:
    app = create_app(
        settings,
        probes=[],
        retriever=retriever or FakeRetriever(hits()),
        query_service=query_service,
    )
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_the_endpoint_returns_the_answer_the_citations_and_what_was_found(
    settings: Settings,
) -> None:
    async with client(settings, service(FakeLLM(ANSWER, VERDICTS))) as http:
        response = await http.post("/v1/query", json={"query": "What is the upload limit?"})

    body = response.json()
    assert response.status_code == 200, body
    assert body["answer"] == "The upload limit is 10 MiB per file. Larger files are rejected."
    assert body["answer_with_markers"] == ANSWER
    assert body["document_ids"] == ["d1"]
    assert body["citations"] == [
        {
            "marker": 1,
            "doc_id": "d1",
            "title": "Limits",
            "source_type": "confluence",
            "verified": True,
            "claims": [0, 1],
        }
    ]
    assert [c["status"] for c in body["claims"]] == ["verified", "verified"]
    assert body["claims"][0]["evidence"] == "The upload limit is 10 MiB per file."
    assert body["verification"] == {
        "method": "judge",
        "model": "claude-haiku-5-5",
        "statements": 2,
        "verified": 2,
        "all_verified": True,
        "error": None,
    }
    assert body["usage"]["input_tokens"] == 2000 and body["usage"]["cost_usd"] > 0
    assert "total" in body["timings_ms"]


async def test_a_citation_of_an_unsupported_statement_is_not_marked_verified(
    settings: Settings,
) -> None:
    verdicts = json.dumps(
        {
            "verdicts": [
                {"claim": 1, "support": "no", "quote": ""},
                {"claim": 2, "support": "no", "quote": ""},
            ]
        }
    )
    async with client(settings, service(FakeLLM(ANSWER, verdicts))) as http:
        body = (await http.post("/v1/query", json={"query": "What is the upload limit?"})).json()

    assert body["citations"][0]["verified"] is False
    assert body["verification"]["all_verified"] is False
    assert [c["status"] for c in body["claims"]] == ["unsupported", "unsupported"]


async def test_without_the_judge_citations_are_neither_verified_nor_rejected(
    settings: Settings,
) -> None:
    async with client(settings, service(FakeLLM(ANSWER))) as http:
        body = (
            await http.post(
                "/v1/query", json={"query": "What is the upload limit?", "verify": False}
            )
        ).json()

    assert body["verification"]["method"] == "values"
    assert body["citations"][0]["verified"] is None
    assert body["verification"]["verified"] == 0


async def test_the_server_default_for_verification_applies_when_the_request_is_silent(
    settings: Settings,
) -> None:
    quiet = settings.model_copy(update={"verify_citations": False})
    llm = FakeLLM(ANSWER, VERDICTS)
    async with client(quiet, service(llm)) as http:
        body = (await http.post("/v1/query", json={"query": "What is the upload limit?"})).json()

    assert body["verification"]["method"] == "values"
    assert len(llm.prompts) == 1


async def test_without_a_model_key_the_endpoint_says_how_to_enable_it(settings: Settings) -> None:
    async with client(settings, None) as http:
        response = await http.post("/v1/query", json={"query": "What is the upload limit?"})

    assert response.status_code == 503
    assert "ANTHROPIC_API_KEY" in response.json()["detail"]


async def test_the_service_is_built_when_a_key_is_configured(settings: Settings) -> None:
    keyed = settings.model_copy(update={"anthropic_api_key": SecretStr("sk-ant-test-key")})
    app = create_app(keyed, probes=[], retriever=FakeRetriever(hits()))

    assert app.state.query_service is not None


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        ({"query": "  "}, "empty query"),
        ({"query": "x", "chunks": 11}, "too many chunks"),
        ({"query": "x", "chunks": 0}, "no chunks"),
        ({"query": "x", "mode": "fuzzy"}, "unknown mode"),
    ],
)
async def test_invalid_requests_are_rejected(
    settings: Settings, body: dict[str, object], reason: str
) -> None:
    async with client(settings, service(FakeLLM(ANSWER, VERDICTS))) as http:
        response = await http.post("/v1/query", json=body)

    assert response.status_code == 422, reason


async def test_the_options_in_the_request_body_reach_the_retriever(settings: Settings) -> None:
    retriever = FakeRetriever(hits())
    body = {
        "query": "What is the upload limit?",
        "mode": "bm25",
        "chunks": 1,
        "filters": {"source_types": ["confluence"]},
    }
    async with client(settings, service(FakeLLM(ANSWER, VERDICTS), retriever), retriever) as http:
        response = await http.post("/v1/query", json=body)

    assert response.status_code == 200
    call = retriever.calls[0]
    assert (call["mode"], call["limit"]) == ("bm25", 1)
    assert call["filters"] == SearchFilters(source_types=("confluence",))


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (RetrievalUnavailableError("models"), 503),
        (LLMError("model down"), 502),
        (BudgetExceededError("spent"), 503),
    ],
)
async def test_failures_map_to_a_status_without_leaking_their_cause(
    settings: Settings, failure: Exception, expected: int
) -> None:
    llm = FakeLLM(ANSWER, VERDICTS)
    retriever = FakeRetriever(
        hits(), error=failure if isinstance(failure, RetrievalUnavailableError) else None
    )
    if not isinstance(failure, RetrievalUnavailableError):

        async def broken(*args: object, **kwargs: object) -> None:
            raise failure

        llm.complete = broken  # type: ignore[method-assign]
    async with client(settings, service(llm, retriever), retriever) as http:
        response = await http.post("/v1/query", json={"query": "What is the upload limit?"})

    assert response.status_code == expected
    assert "model down" not in response.text and "models" not in response.json()["detail"]


async def test_the_endpoint_says_when_it_declined_and_why(settings: Settings) -> None:
    from rag_engine.generation.abstain import AbstentionPolicy

    llm = FakeLLM("never used")
    weak = FakeRetriever([make_hit("a", doc_id="d1", score=-3.0)])
    generator = AnswerGenerator(weak, llm, mode="hybrid+rerank", abstention=AbstentionPolicy(0.0))
    declining = QueryService(generator, CitationVerifier(llm))

    async with client(settings, declining) as http:
        response = await http.post("/v1/query", json={"query": "What is the secret?"})

    body = response.json()
    assert response.status_code == 200, body
    assert body["abstained"] is True
    assert (body["status"], body["gaps"]) == ("abstained", [])
    assert body["abstention"] == {"reason": "weak_evidence", "top_score": -3.0, "threshold": 0.0}
    assert body["claims"] == [] and body["citations"] == [] and body["document_ids"] == []
    assert body["verification"]["method"] == "none"
    assert body["usage"]["cost_usd"] == 0.0
    assert llm.prompts == []


async def test_an_answered_question_is_not_marked_as_declined(settings: Settings) -> None:
    async with client(settings, service(FakeLLM(ANSWER, VERDICTS))) as http:
        response = await http.post("/v1/query", json={"query": "What is the upload limit?"})

    body = response.json()
    assert (body["abstained"], body["abstention"]) == (False, None)
    assert (body["status"], body["gaps"]) == ("answered", [])


async def test_the_endpoint_reports_what_the_answer_says_the_documents_lack(
    settings: Settings,
) -> None:
    said = "The upload limit is 10 MiB per file [1]. The passages do not give the retry policy."
    verdict = json.dumps(
        {
            "verdicts": [
                {"claim": 1, "support": "yes", "quote": "The upload limit is 10 MiB per file."}
            ]
        }
    )

    async with client(settings, service(FakeLLM(said, verdict))) as http:
        response = await http.post("/v1/query", json={"query": "What is the upload limit?"})

    body = response.json()
    assert response.status_code == 200, body
    assert body["status"] == "partial"
    assert body["gaps"] == ["The passages do not give the retry policy."]
    assert body["abstained"] is False
