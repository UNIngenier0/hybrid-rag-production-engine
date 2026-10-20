import json

import pytest

from fakes import FakeLLM
from rag_engine.core.errors import LLMError
from rag_engine.core.models import LLMReply, LLMUsage
from rag_engine.generation.prompt import ContextBlock
from rag_engine.generation.verify import (
    CitationVerifier,
    Verification,
    missing_values,
    quote_in_passages,
    split_statements,
    values_of,
)

PASSAGE_1 = "The upload limit is 10 MiB per file. Larger files are rejected with HTTP 413."
PASSAGE_2 = "Requests to gen-infer-01 time out after 30 seconds. See limits.yaml for the rest."
BLOCKS = [
    ContextBlock(1, "d1", "Limits", "confluence", PASSAGE_1),
    ContextBlock(2, "d2", "Runbook", "slack", PASSAGE_2),
]


def verdicts(*rows: tuple[int, str, str]) -> str:
    return json.dumps({"verdicts": [{"claim": n, "support": s, "quote": q} for n, s, q in rows]})


def statuses(verification: Verification) -> list[str]:
    return [claim.status for claim in verification.claims]


class FailingLLM:
    model = "judge-model"

    async def complete(self, system: str, user: str, **kwargs: object) -> LLMReply:
        raise LLMError("down")


# --- splitting an answer into statements


def test_a_marker_belongs_to_the_sentence_it_ends() -> None:
    found = split_statements("The limit is 10 MiB [2]. Both agree [1][3].")

    assert [(s.text, s.markers) for s in found] == [
        ("The limit is 10 MiB.", (2,)),
        ("Both agree.", (1, 3)),
    ]


def test_a_marker_after_the_full_stop_still_belongs_to_the_sentence_before_it() -> None:
    found = split_statements("The limit is 10 MiB. [2] Another statement follows here [1].")

    assert [s.markers for s in found] == [(2,), (1,)]


def test_decimals_versions_file_names_and_abbreviations_do_not_end_a_sentence() -> None:
    found = split_statements("Latency is 1.5 s on v2.0, e.g. under load, per limits.yaml [1].")

    assert len(found) == 1
    assert found[0].markers == (1,)


def test_list_items_are_separate_statements_without_their_bullets() -> None:
    found = split_statements("- First item is this one [1]\n- Second item is that one [2]")

    assert [(s.text, s.markers) for s in found] == [
        ("First item is this one", (1,)),
        ("Second item is that one", (2,)),
    ]


def test_a_short_cited_fragment_is_kept_and_a_short_uncited_one_is_dropped() -> None:
    found = split_statements("Yes [1]. OK. Sure thing.")

    assert [(s.text, s.markers) for s in found] == [("Yes.", (1,))]


# --- the values a statement commits to


def test_numbers_are_compared_by_value_not_by_spelling() -> None:
    numbers, _ = values_of("It allows 1,500 requests, 03:40, 0.50 and 10 MiB.")

    assert numbers == ["1500", "3", "40", "0.5", "10"]


def test_identifiers_are_found_in_code_spans_hyphenated_names_and_file_names() -> None:
    _, identifiers = values_of("Set `max_conn` for gen-infer-01 and NCCL_DEBUG in limits.yaml.")

    assert identifiers == ["max_conn", "gen-infer-01", "NCCL_DEBUG", "limits.yaml"]


def test_a_value_missing_from_every_cited_passage_is_reported() -> None:
    absent = missing_values(
        "The limit is 20 MiB for gen-infer-01 and gen-infer-07.", [PASSAGE_1, PASSAGE_2]
    )

    assert absent == ["20", "gen-infer-07"]


def test_values_may_come_from_any_of_the_cited_passages_and_ignore_case_and_commas() -> None:
    assert (
        missing_values("GEN-INFER-01 allows 10 MiB and 30 seconds.", [PASSAGE_1, PASSAGE_2]) == []
    )
    assert missing_values("There are 10000 of them.", ["Up to 10,000 of them exist."]) == []


# --- the judge's evidence has to be real


def test_a_quote_is_found_despite_case_and_whitespace() -> None:
    assert quote_in_passages("the upload  limit is\n10 mib per file", [PASSAGE_1])


def test_a_quote_with_a_small_slip_is_accepted_and_an_invented_one_is_not() -> None:
    assert quote_in_passages("The upload limit is 10 MiB per files.", [PASSAGE_1])
    assert not quote_in_passages("Uploads above 25 GiB are throttled nightly.", [PASSAGE_1])
    assert not quote_in_passages("limit", [PASSAGE_1])


# --- the verifier


async def test_each_statement_gets_the_verdict_it_earns() -> None:
    answer = (
        "The upload limit is 10 MiB per file [1]. Requests time out after 90 seconds [2]. "
        "Retries are limited to three per hour. The proxy is configured in file seven [9]. "
        "I could not find the retry policy in the documents."
    )
    judge = FakeLLM(
        verdicts(
            (1, "yes", "The upload limit is 10 MiB per file."),
            (2, "no", ""),
        )
    )

    result = await CitationVerifier(judge).verify(answer, BLOCKS)

    assert statuses(result) == ["verified", "unsupported", "uncited", "unresolved", "no_claim"]
    assert result.claims[0].quote == "The upload limit is 10 MiB per file."
    assert result.method == "judge"
    assert not result.all_verified
    assert [c.status for c in result.problems] == ["unsupported", "uncited", "unresolved"]


async def test_the_judge_only_sees_the_passages_a_statement_cites() -> None:
    judge = FakeLLM(verdicts((1, "yes", "The upload limit is 10 MiB per file.")))

    await CitationVerifier(judge).verify("The upload limit is 10 MiB per file [1].", BLOCKS)

    prompt = judge.prompts[0][1]
    assert "upload limit is 10 MiB" in prompt
    assert "gen-infer-01" not in prompt  # passage 2 was not cited


async def test_evidence_the_judge_made_up_does_not_verify_anything() -> None:
    judge = FakeLLM(verdicts((1, "yes", "Files up to 25 GiB are accepted overnight.")))

    result = await CitationVerifier(judge).verify(
        "The upload limit is 10 MiB per file [1].", BLOCKS
    )

    assert statuses(result) == ["unsupported"]
    assert "not in the cited passages" in (result.claims[0].reason or "")


async def test_a_yes_without_evidence_is_not_enough() -> None:
    judge = FakeLLM(verdicts((1, "yes", "")))

    result = await CitationVerifier(judge).verify(
        "The upload limit is 10 MiB per file [1].", BLOCKS
    )

    assert statuses(result) == ["unsupported"]


async def test_a_value_the_passage_lacks_downgrades_a_judge_yes_to_partial() -> None:
    judge = FakeLLM(verdicts((1, "yes", "The upload limit is 10 MiB per file.")))

    result = await CitationVerifier(judge).verify(
        "The upload limit is 20 MiB per file [1].", BLOCKS
    )

    assert statuses(result) == ["partially_supported"]
    assert "20" in (result.claims[0].reason or "")


async def test_a_partial_verdict_is_reported_as_partial() -> None:
    judge = FakeLLM(verdicts((1, "partial", "Larger files are rejected with HTTP 413.")))

    result = await CitationVerifier(judge).verify(
        "Larger files are rejected with HTTP 413 and logged [1].", BLOCKS
    )

    assert statuses(result) == ["partially_supported"]


async def test_a_judge_that_fails_leaves_the_statements_unchecked_not_verified() -> None:
    result = await CitationVerifier(FailingLLM()).verify(
        "The upload limit is 10 MiB per file [1].", BLOCKS
    )

    assert statuses(result) == ["unchecked"]
    assert result.error is not None and "LLMError" in result.error
    assert not result.all_verified


async def test_a_judge_that_returns_garbage_leaves_the_statements_unchecked() -> None:
    result = await CitationVerifier(FakeLLM("I think it is fine.")).verify(
        "The upload limit is 10 MiB per file [1].", BLOCKS
    )

    assert statuses(result) == ["unchecked"]
    assert result.error == "the judge did not return usable verdicts"


async def test_a_missing_verdict_leaves_that_statement_unchecked() -> None:
    judge = FakeLLM(verdicts((1, "yes", "The upload limit is 10 MiB per file.")))
    answer = "The upload limit is 10 MiB per file [1]. Requests time out after 30 seconds [2]."

    result = await CitationVerifier(judge).verify(answer, BLOCKS)

    assert statuses(result) == ["verified", "unchecked"]


async def test_without_a_judge_nothing_is_ever_verified() -> None:
    answer = (
        "The upload limit is 10 MiB per file [1]. Requests time out after 90 seconds [2]. "
        "Something else is also true here."
    )

    result = await CitationVerifier(None).verify(answer, BLOCKS)

    assert statuses(result) == ["unchecked", "ungrounded", "uncited"]
    assert result.method == "values"
    assert result.usage is None
    assert "90" in (result.claims[1].reason or "")


async def test_the_judge_can_be_switched_off_for_one_request() -> None:
    judge = FakeLLM(verdicts((1, "yes", "The upload limit is 10 MiB per file.")))

    result = await CitationVerifier(judge).verify(
        "The upload limit is 10 MiB per file [1].", BLOCKS, use_judge=False
    )

    assert statuses(result) == ["unchecked"]
    assert judge.prompts == []


async def test_an_answer_that_only_says_nothing_was_found_has_nothing_to_verify() -> None:
    judge = FakeLLM()

    result = await CitationVerifier(judge).verify("I could not find this in the documents.", BLOCKS)

    assert statuses(result) == ["no_claim"]
    assert not result.all_verified  # nothing was verified
    assert judge.prompts == []  # and no judge call was spent


async def test_usage_of_the_judge_is_reported() -> None:
    judge = FakeLLM(
        verdicts((1, "yes", "The upload limit is 10 MiB per file.")),
        usage=LLMUsage(input_tokens=700, output_tokens=40),
        model="judge-model",
    )

    result = await CitationVerifier(judge).verify(
        "The upload limit is 10 MiB per file [1].", BLOCKS
    )

    assert result.model == "judge-model"
    assert result.usage == LLMUsage(700, 40)
    assert result.all_verified


@pytest.mark.parametrize("reply", ['{"verdicts": "no"}', '{"verdicts": [1, 2]}', "{}", "[]"])
async def test_unexpected_json_shapes_are_survived(reply: str) -> None:
    result = await CitationVerifier(FakeLLM(reply)).verify(
        "The upload limit is 10 MiB per file [1].", BLOCKS
    )

    assert statuses(result) == ["unchecked"]


@pytest.mark.parametrize(
    "sentence",
    [
        "I could not find this in the documents.",
        "The PR does not give a separate per-request total for that path.",
        "The documents do not say which limit is current.",
        "No passage mentions a retry policy.",
        "The rollout date is not specified anywhere.",
    ],
)
async def test_an_uncited_statement_about_what_the_documents_lack_is_not_a_missing_citation(
    sentence: str,
) -> None:
    result = await CitationVerifier(None).verify(sentence, BLOCKS)

    assert statuses(result) == ["no_claim"], sentence


async def test_the_same_statement_with_a_citation_is_checked_like_any_other() -> None:
    judge = FakeLLM(verdicts((1, "no", "")))

    result = await CitationVerifier(judge).verify(
        "The documents do not give a total for uploads [1].", BLOCKS
    )

    assert statuses(result) == ["unsupported"]


@pytest.mark.parametrize(
    ("sentence", "kind"),
    [
        ("**Routing and execution**", "a heading"),
        ("The current automated rollback thresholds are:", "a heading"),
        ("The passages do not explicitly say that sharing avoids a review.", "lack something"),
        ("That engineering Slack thread is not in the passages.", "lack something"),
        ("Passage 5 describes a different incident.", "the passages themselves"),
        ("The documents conflict, and none of them addresses the SKU.", "lack something"),
    ],
)
async def test_headings_and_comments_on_the_sources_are_not_missing_citations(
    sentence: str, kind: str
) -> None:
    result = await CitationVerifier(None).verify(sentence, BLOCKS)

    assert statuses(result) == ["no_claim"], sentence
    assert kind in (result.claims[0].reason or "")


@pytest.mark.parametrize(
    "sentence",
    [
        "The root cause was a deduplication bug in the usage exporter.",
        "Open an internal-support Jira ticket for this request.",
        "The flag is `runtime.bandwidth_fanout.enabled`.",
    ],
)
async def test_a_factual_or_procedural_statement_without_a_citation_is_flagged(
    sentence: str,
) -> None:
    result = await CitationVerifier(None).verify(sentence, BLOCKS)

    assert statuses(result) == ["uncited"]


def test_typographic_quotes_dashes_and_markdown_do_not_make_a_quote_fail() -> None:
    passage = (
        "We\u2019ve been improving the taxonomy \u2014 some resets are counted as **stream_error**."
    )

    assert quote_in_passages(
        "We've been improving the taxonomy - some resets are counted as stream_error", [passage]
    )


def test_a_quote_with_an_ellipsis_must_be_found_in_each_of_its_parts() -> None:
    passage = (
        "The alert fired at 14:02. Then the on-call was paged. "
        "The traffic shift was applied by hand."
    )

    assert quote_in_passages(
        "The alert fired at 14:02 ... The traffic shift was applied by hand", [passage]
    )
    assert not quote_in_passages(
        "The alert fired at 14:02 ... Customers were refunded in full", [passage]
    )


def test_a_time_inside_a_timestamp_of_the_passage_is_found() -> None:
    assert (
        missing_values(
            "Scheduled for 2026-03-26 at 09:00Z.", ['at "2026-03-26T09:00Z" by facilities']
        )
        == []
    )


def test_the_references_an_answer_makes_to_its_own_passages_are_not_values() -> None:
    numbers, _ = values_of("Passage 5 and passages 2 and 3 describe a 14-day window.")

    assert numbers == ["14"]


def test_only_single_word_code_spans_count_as_identifiers() -> None:
    _, identifiers = values_of("Run `rdwd demo start` and check `console_baseline`, then `ok`.")

    assert identifiers == ["console_baseline", "ok"]


def test_two_code_spans_are_not_read_as_one() -> None:
    _, identifiers = values_of("Use `alpha_one`, which is fast, and then `beta_two`.")

    assert identifiers == ["alpha_one", "beta_two"]


async def test_values_may_come_from_the_title_of_a_cited_passage() -> None:
    blocks = [ContextBlock(1, "d1", "Weekly status 2026-01-16", "confluence", "Overall: At Risk")]
    judge = FakeLLM(verdicts((1, "yes", "Overall: At Risk")))

    result = await CitationVerifier(judge).verify(
        "The 2026-01-16 update marked the project At Risk [1].", blocks
    )

    assert statuses(result) == ["verified"]


def test_escaped_quotes_left_in_a_passage_by_its_export_do_not_make_a_quote_fail() -> None:
    passage = (
        'error: no matches for kind \\"RedwoodTenant\\" in version \\"platform.redwood.ai/v1\\"'
    )

    assert quote_in_passages(
        'no matches for kind "RedwoodTenant" in version "platform.redwood.ai/v1"', [passage]
    )
