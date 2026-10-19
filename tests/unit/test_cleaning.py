import json

from rag_engine.ingestion.cleaning import (
    ELEMENT_SEPARATOR,
    decode_serialized_list,
    normalize_document,
)


def clean(content: str, title: str = "Title") -> tuple[str, str]:
    document, _ = normalize_document("dsid_1", "gmail", title, content)
    return document.title, document.content


def test_decodes_a_json_list_of_strings_into_separated_elements() -> None:
    _, content = clean(json.dumps(["From: a\nHi", "From: b\nHello"]))

    assert content == f"From: a\nHi{ELEMENT_SEPARATOR}From: b\nHello"


def test_decodes_a_python_repr_list() -> None:
    assert decode_serialized_list("['one', \"two\"]") == ["one", "two"]


def test_text_that_only_looks_like_a_list_is_left_alone() -> None:
    assert decode_serialized_list("[00:00] Maya: hello everyone") is None
    assert decode_serialized_list("[1, 2, 3]") is None
    assert decode_serialized_list("[]") is None
    assert decode_serialized_list("plain text") is None


def test_flattened_text_gets_its_newlines_back() -> None:
    flattened = "# Summary\\n\\nFirst paragraph.\\n\\n## Scope\\n\\nSecond paragraph."

    _, content = clean(flattened)

    assert "\\n" not in content
    assert content.splitlines()[0] == "# Summary"
    assert content.count("\n") == 6


def test_text_with_real_newlines_keeps_its_escaped_ones() -> None:
    log = "step one\nrun: echo a\\nb\\nc\nstep two\nmore text"

    _, content = clean(log)

    assert "a\\nb\\nc" in content


def test_a_single_escape_is_not_enough_to_call_text_flattened() -> None:
    # One escape alone could be a Windows path such as C:\new; two or more mean flattened text.
    assert "\\n" in clean(r"open C:\new_folder now")[1]
    assert "\\n" not in clean("line one\\nline two\\nline three")[1]


def test_nul_and_control_characters_are_replaced() -> None:
    document, flags = normalize_document("d", "slack", "t", "a\x00b\x07c\td\r\ne")

    assert document.content == "a b c\td\ne"
    assert flags.control_chars


def test_a_normal_title_is_kept() -> None:
    title, _ = clean("body text", title="Quarterly plan")

    assert title == "Quarterly plan"


def test_a_title_that_copies_the_body_becomes_its_first_line() -> None:
    body = "Title: Egress controls runbook\n\nOverview\n\n" + "More text. " * 50

    title, content = clean(body, title=body)

    assert title == "Egress controls runbook"
    assert content.startswith("Title: Egress controls runbook")


def test_an_empty_title_comes_from_the_content() -> None:
    title, _ = clean("alex: quick status check on the v2 rollout\n\nben: ok", title="")

    assert title == "alex: quick status check on the v2 rollout"


def test_a_very_long_single_line_title_is_shortened_at_a_word_boundary() -> None:
    long_title = "word " * 100

    title, _ = clean("body", title=long_title)

    assert 0 < len(title) <= 160
    assert not title.endswith(" ")


def test_the_content_hash_identifies_the_cleaned_text() -> None:
    first, _ = normalize_document("d", "slack", "t", "same text")
    again, _ = normalize_document("d", "slack", "other title", "same text")
    different, _ = normalize_document("d", "slack", "t", "other text")

    assert first.content_sha256 == again.content_sha256
    assert first.content_sha256 != different.content_sha256


def test_missing_fields_do_not_raise() -> None:
    document, _ = normalize_document("d", "slack", None, None)

    assert document.content == ""
    assert document.title == ""
