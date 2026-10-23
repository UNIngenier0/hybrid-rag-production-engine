import pytest

from rag_engine.generation.coverage import ABSTAINED, coverage


def test_an_answer_that_states_everything_is_answered() -> None:
    result = coverage("The limit is 10 MiB per file [1]. Larger files get HTTP 413 [2].")

    assert (result.status, result.gaps) == ("answered", ())


def test_an_answer_that_says_part_is_missing_is_partial_and_quotes_the_gap() -> None:
    result = coverage(
        "The retry limit is 3 [1]. The passages do not give the backoff interval. "
        "It applies to all regions [1]."
    )

    assert result.status == "partial"
    assert result.gaps == ("The passages do not give the backoff interval.",)


def test_an_answer_that_only_says_nothing_was_found_is_not_found() -> None:
    result = coverage("I could not find the contract address in the documents.")

    assert result.status == "not_found"
    assert len(result.gaps) == 1


def test_a_heading_does_not_turn_a_not_found_answer_into_a_partial_one() -> None:
    result = coverage("**What the documents say:**\nThe documents do not name the network.")

    assert result.status == "not_found"


def test_markers_do_not_hide_a_gap() -> None:
    result = coverage("The passages do not specify the window [2].")

    assert result.status == "not_found"


@pytest.mark.parametrize(
    "sentence",
    [
        "The documents do not mention a lunch bracket.",
        "The thread doesn't state a floor.",
        "I couldn't find the value.",
        "The policy is not described in the passages.",
        "No document gives the owner.",
        "None of the passages say who approved it.",
    ],
)
def test_the_usual_ways_of_saying_something_is_missing_are_recognised(sentence: str) -> None:
    assert coverage(f"The cap is 5 [1]. {sentence}").status == "partial"


def test_the_abstained_status_is_fixed() -> None:
    assert (ABSTAINED.status, ABSTAINED.gaps) == ("abstained", ())
