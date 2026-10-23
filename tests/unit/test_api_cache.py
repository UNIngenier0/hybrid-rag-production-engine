import json

import httpx

from fakes import FakeCache, FakeLLM, FakeRetriever, make_hit
from rag_engine.api.main import create_app
from rag_engine.api.query_cache import cache_key
from rag_engine.api.schemas import QueryRequest
from rag_engine.core.config import Settings
from rag_engine.generation.answer import AnswerGenerator
from rag_engine.generation.query import QueryService
from rag_engine.generation.verify import CitationVerifier

PASSAGE = "The upload limit is 10 MiB per file."
ANSWER = "The upload limit is 10 MiB per file [1]."
VERDICT = json.dumps(
    {"verdicts": [{"claim": 1, "support": "yes", "quote": "The upload limit is 10 MiB per file."}]}
)
QUESTION = {"query": "What is the upload limit?"}


def service(llm: FakeLLM) -> QueryService:
    retriever = FakeRetriever([make_hit("a", doc_id="d1", title="Limits", context=PASSAGE)])
    return QueryService(
        AnswerGenerator(retriever, llm, mode="hybrid", chunks=1), CitationVerifier(llm)
    )


def client(
    settings: Settings, llm: FakeLLM, cache: FakeCache, **changes: object
) -> httpx.AsyncClient:
    cached = settings.model_copy(update={"query_cache_ttl_seconds": 600, **changes})
    app = create_app(
        cached,
        probes=[],
        retriever=FakeRetriever([]),
        query_service=service(llm),
        response_cache=cache,
    )
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def request(**fields: object) -> QueryRequest:
    return QueryRequest(**{"query": "What is the upload limit?", **fields})  # type: ignore[arg-type]


async def test_an_identical_question_is_answered_from_the_cache_without_the_model(
    settings: Settings,
) -> None:
    llm, cache = FakeLLM(ANSWER, VERDICT), FakeCache()
    async with client(settings, llm, cache) as http:
        first = (await http.post("/v1/query", json=QUESTION)).json()
        second = (await http.post("/v1/query", json=QUESTION)).json()

    assert (first["cached"], second["cached"]) == (False, True)
    assert second["answer"] == first["answer"] and second["claims"] == first["claims"]
    assert len(llm.prompts) == 2  # one answer and one judgement, for the first request only
    assert second["usage"]["input_tokens"] == 0 and second["usage"]["cost_usd"] == 0.0
    assert set(second["timings_ms"]) == {"total"}
    assert list(cache.ttls.values()) == [600]


async def test_questions_that_differ_only_in_whitespace_share_an_answer(settings: Settings) -> None:
    llm, cache = FakeLLM(ANSWER, VERDICT), FakeCache()
    async with client(settings, llm, cache) as http:
        await http.post("/v1/query", json=QUESTION)
        again = (
            await http.post("/v1/query", json={"query": "  What is   the upload limit?  "})
        ).json()

    assert again["cached"] is True


async def test_a_different_option_is_a_different_question(settings: Settings) -> None:
    llm, cache = FakeLLM(ANSWER, VERDICT), FakeCache()
    async with client(settings, llm, cache) as http:
        await http.post("/v1/query", json=QUESTION)
        without_judge = (await http.post("/v1/query", json={**QUESTION, "verify": False})).json()

    assert without_judge["cached"] is False
    assert len(cache.values) == 2


async def test_a_request_can_ask_for_a_fresh_answer_that_is_not_stored(
    settings: Settings,
) -> None:
    llm, cache = FakeLLM(ANSWER, VERDICT), FakeCache()
    async with client(settings, llm, cache) as http:
        first = (await http.post("/v1/query", json={**QUESTION, "cache": False})).json()
        second = (await http.post("/v1/query", json={**QUESTION, "cache": False})).json()

    assert (first["cached"], second["cached"]) == (False, False)
    assert cache.values == {}
    answering = [system for system, _ in llm.prompts if system.startswith("You answer")]
    assert len(answering) == 2  # the model wrote both answers


async def test_a_fresh_answer_request_does_not_read_a_stored_one(settings: Settings) -> None:
    llm, cache = FakeLLM(ANSWER, VERDICT), FakeCache()
    async with client(settings, llm, cache) as http:
        await http.post("/v1/query", json=QUESTION)  # stores
        fresh = (await http.post("/v1/query", json={**QUESTION, "cache": False})).json()

    assert fresh["cached"] is False


async def test_with_a_ttl_of_zero_nothing_is_cached(settings: Settings) -> None:
    llm, cache = FakeLLM(ANSWER, VERDICT), FakeCache()
    async with client(settings, llm, cache, query_cache_ttl_seconds=0) as http:
        first = (await http.post("/v1/query", json=QUESTION)).json()
        second = (await http.post("/v1/query", json=QUESTION)).json()

    assert (first["cached"], second["cached"]) == (False, False)
    assert cache.values == {}


async def test_an_answer_whose_citations_could_not_be_checked_is_not_kept(
    settings: Settings,
) -> None:
    llm, cache = FakeLLM(ANSWER, "this is not json"), FakeCache()
    async with client(settings, llm, cache) as http:
        body = (await http.post("/v1/query", json=QUESTION)).json()

    assert body["verification"]["error"]
    assert cache.values == {}


async def test_if_the_cache_is_down_the_question_is_still_answered(settings: Settings) -> None:
    llm, cache = FakeLLM(ANSWER, VERDICT), FakeCache(error=ConnectionError("redis is down"))
    async with client(settings, llm, cache) as http:
        response = await http.post("/v1/query", json=QUESTION)
        text = (await http.get("/metrics")).text

    assert response.status_code == 200 and response.json()["cached"] is False
    assert 'rag_query_cache_total{result="error"}' in text


async def test_an_unreadable_cached_value_is_ignored(settings: Settings) -> None:
    llm, cache = FakeLLM(ANSWER, VERDICT), FakeCache()
    cache.values[
        cache_key(settings.model_copy(update={"query_cache_ttl_seconds": 600}), request(), True)
    ] = "{broken"
    async with client(settings, llm, cache) as http:
        body = (await http.post("/v1/query", json=QUESTION)).json()

    assert body["cached"] is False and body["answer"]


async def test_hits_and_misses_are_counted_and_cached_answers_cost_nothing(
    settings: Settings,
) -> None:
    llm, cache = FakeLLM(ANSWER, VERDICT), FakeCache()
    async with client(settings, llm, cache) as http:
        await http.post("/v1/query", json=QUESTION)
        await http.post("/v1/query", json=QUESTION)
        text = (await http.get("/metrics")).text

    assert 'rag_query_cache_total{result="miss"} 1.0' in text
    assert 'rag_query_cache_total{result="hit"} 1.0' in text
    assert 'rag_queries_total{status="answered"} 1.0' in text  # the hit is not counted again


def test_the_key_changes_with_everything_that_changes_the_answer(settings: Settings) -> None:
    base = cache_key(settings, request(), True)

    assert cache_key(settings, request(), True) == base
    assert cache_key(settings, request(query="  What is   the upload limit? "), True) == base
    assert cache_key(settings, request(), False) != base
    assert cache_key(settings, request(mode="dense"), True) != base
    assert cache_key(settings, request(chunks=3), True) != base
    assert cache_key(settings, request(filters={"source_types": ["slack"]}), True) != base
    for change in (
        {"llm_model": "another-model"},
        {"verification_model": "another-judge"},
        {"reranker_model": "some/reranker"},
        {"abstain_threshold": 0.3},
        {"query_cache_namespace": "v2"},
    ):
        assert cache_key(settings.model_copy(update=change), request(), True) != base, change


def test_filters_in_a_different_order_share_a_key(settings: Settings) -> None:
    one = request(filters={"doc_ids": ["a", "b"]})
    two = request(filters={"doc_ids": ["b", "a"]})

    assert cache_key(settings, one, True) == cache_key(settings, two, True)


def test_an_omitted_number_of_passages_keys_the_same_as_the_server_default(
    settings: Settings,
) -> None:
    assert cache_key(settings, request(), True) == cache_key(
        settings, request(chunks=settings.query_chunks), True
    )
