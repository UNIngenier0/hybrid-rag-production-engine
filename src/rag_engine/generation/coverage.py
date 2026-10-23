"""Does the answer say that the documents lack part of what was asked?

A model told to answer only from the passages says so when they fall short ("the documents do not
give the default value"). That sentence is the most useful reliability signal an answer carries,
and reading it needs no model call: it is found in the statements of the answer itself.

Measured on the benchmark (docs/generation.md): the answers that say something is missing were
judged correct 15% of the time, the others 84%, and every one of the 20 questions whose
information is not in the corpus got such a sentence.
"""

from dataclasses import dataclass
from typing import Literal

from rag_engine.generation.verify import is_absence, is_heading, split_statements

Status = Literal["answered", "partial", "not_found", "abstained"]


@dataclass(frozen=True, slots=True)
class Coverage:
    status: Status
    gaps: tuple[str, ...]  # the statements that say the documents lack something


def coverage(raw_text: str) -> Coverage:
    """`answered`: nothing is said to be missing. `partial`: some of the answer is stated and
    some is said to be missing. `not_found`: the answer says only that nothing was found."""
    statements = [s.text for s in split_statements(raw_text) if not is_heading(s.text)]
    gaps = tuple(text for text in statements if is_absence(text))
    if not gaps:
        return Coverage("answered", ())
    return Coverage("not_found" if len(gaps) == len(statements) else "partial", gaps)


ABSTAINED = Coverage("abstained", ())
