"""Reading the [n] markers an answer carries."""

import re
from collections.abc import Sequence

# [3], [3, 4] and runs such as [1][2]; a marker is one or more numbers in one pair of brackets.
_MARKER = re.compile(r"\[(\d{1,3}(?:\s*,\s*\d{1,3})*)\]")


def cited_numbers(text: str) -> list[int]:
    """Passage numbers the text cites, in order of first appearance, each once."""
    numbers: list[int] = []
    for match in _MARKER.finditer(text):
        for part in match.group(1).split(","):
            number = int(part)
            if number not in numbers:
                numbers.append(number)
    return numbers


def strip_markers(text: str) -> str:
    """The text without its markers, with the gaps they leave closed."""
    cleaned = _MARKER.sub("", text)
    cleaned = re.sub(r"[ \t]+([.,;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return "\n".join(line.rstrip() for line in cleaned.splitlines()).strip()


def resolve(numbers: Sequence[int], doc_ids: Sequence[str]) -> tuple[list[str], list[int]]:
    """Map passage numbers (1-based) to document IDs. Returns the distinct documents in citation
    order and the numbers that point at no passage."""
    documents: list[str] = []
    unresolved: list[int] = []
    for number in numbers:
        if 1 <= number <= len(doc_ids):
            if doc_ids[number - 1] not in documents:
                documents.append(doc_ids[number - 1])
        else:
            unresolved.append(number)
    return documents, unresolved
