"""Checking that each statement of an answer is backed by the passages it cites.

A model that is told to cite can still cite wrongly: a marker that points at a passage that says
something else, a number that appears in no source. Verification takes the answer apart into its
statements, and for each one asks three questions in order:

1. Does the marker point at a passage that was shown? (`unresolved` if not; `uncited` if there is
   no marker at all.)
2. Do the specific values in the statement (numbers, identifiers, file names) appear in the cited
   passages? This is a plain text check, so it cannot be argued with.
3. Does a judge model, given only the cited passages, say they support the statement, and can it
   point at the sentence that does? The quote it gives must be found in the passages: a judge that
   invents its evidence does not get to verify anything.

Only a statement that passes all three is `verified`. The judge is optional; without it the
statements that pass the first two checks are `unchecked`, never `verified`.
"""

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from typing import Literal

from rag_engine.core.errors import BudgetExceededError, LLMError
from rag_engine.core.models import LLMUsage
from rag_engine.core.ports import LLMClient
from rag_engine.generation.citations import cited_numbers, strip_markers
from rag_engine.generation.prompt import ContextBlock

Status = Literal[
    "verified",  # the judge found the statement in the cited passages and the quote is real
    "partially_supported",  # the cited passages support part of it, or lack one of its values
    "unsupported",  # the cited passages do not say it, or the judge's evidence is not in them
    "ungrounded",  # (no judge) a value of the statement appears in none of the cited passages
    "unchecked",  # no judge ran, or it gave no verdict; nothing found wrong in the cheap checks
    "unresolved",  # every marker points at a passage that was never shown
    "uncited",  # a statement of fact with no marker
    "no_claim",  # a heading, or a statement about what the documents lack: nothing to verify
]
# Statuses that count as a problem with the answer.
PROBLEMS: frozenset[Status] = frozenset(
    {"partially_supported", "unsupported", "ungrounded", "unresolved", "uncited"}
)

JUDGE_PROMPT = """\
You check whether statements are supported by the passages they cite. Each statement comes with \
the passages it cites; judge it using only those passages, never outside knowledge.

Reply with JSON only, in this shape:
{"verdicts": [{"claim": 1, "support": "yes", "quote": "text copied from a passage"}]}

- "support" is "yes" when the cited passages state everything the statement says (a paraphrase is \
fine, but every specific value, name and condition must be there), "partial" when they support \
only part of it or leave out a specific value, and "no" when they do not support it or \
contradict it.
- "quote" is the sentence or phrase of a cited passage that supports the statement, copied \
character for character, at most 300 characters. Use an empty string when "support" is "no".
- Give one verdict for every numbered statement.\
"""

ABSENCE = re.compile(
    r"(?:could\s*n[o']t|cannot|can't|unable to|did\s*n[o']t|do(?:es)?\s*n[o']t)\s+(?:\w+ly\s+)?"
    r"(?:find|locate|give|say|state|specify|mention|describe|provide|list|include|contain|cover"
    r"|define|show|address|name)"
    r"|\bnot (?:found|mentioned|stated|covered|present|described|specified|given|provided"
    r"|documented|in the (?:passages|documents|sources))\b"
    r"|\bno (?:passage|document|information|mention|source)s?\b"
    r"|\bnone of (?:them|the (?:passages|documents|sources))\b",
    re.IGNORECASE,
)
_ABOUT_PASSAGES = re.compile(
    r"\b(?:passages?|documents?|sources?)\s+\d"
    r"|\b(?:passages|documents|sources)\s+(?:conflict|disagree|differ|contradict|don't|do not)\b",
    re.IGNORECASE,
)
_HEADING = re.compile(r"\*{2}[^*]{1,80}\*{2}:?|.*:")
_MARKERS = r"(?:\s*\[\d{1,3}(?:\s*,\s*\d{1,3})*\])*"
_FRAGMENT = re.compile(rf".+?(?:[.!?]+(?=\s|$)|\n|$){_MARKERS}", re.DOTALL)
_ABBREVIATIONS = ("e.g.", "i.e.", "vs.", "etc.", "approx.", "inc.", "mr.", "ms.", "dr.", "fig.")
_DOT = chr(
    0xE000
)  # a private-use character that stands in for the full stops that do not end a sentence
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")
# In a passage a number may sit inside a timestamp or a name (T09:00Z, p95): all of it is digits.
_ANY_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
# "passage 5", "documents 2 and 3": the numbers of the answer's own references are not values.
_REFERENCE = re.compile(
    r"\b(?:passages?|documents?|sources?)\s+\d+(?:\s*(?:,|and|&)\s*\d+)*", re.IGNORECASE
)
_IDENTIFIERS = (
    re.compile(r"\b[A-Za-z][A-Za-z_-]*\d[A-Za-z0-9_-]*\b"),  # gen-infer-01, p95, qst_0099
    re.compile(r"\b[A-Za-z]+(?:_[A-Za-z0-9]+)+\b"),  # NCCL_DEBUG, max_connections
    re.compile(r"\b[\w-]+\.(?:json|ya?ml|py|md|txt|csv|sql|sh|toml|conf)\b", re.IGNORECASE),
)


@dataclass(frozen=True, slots=True)
class Claim:
    text: str  # the statement without its markers
    markers: tuple[int, ...]
    status: Status
    reason: str | None = None
    quote: str | None = None  # the evidence the judge pointed at, found in the cited passage
    judged: str | None = None  # what the judge said before the other checks had their say


@dataclass(frozen=True, slots=True)
class Verification:
    claims: list[Claim]
    method: Literal["judge", "values", "none"]  # "none": there was no answer to verify
    model: str | None = None
    usage: LLMUsage | None = None
    error: str | None = None  # why the judge could not run, if it was meant to

    @property
    def checked(self) -> list[Claim]:
        return [claim for claim in self.claims if claim.status != "no_claim"]

    @property
    def verified(self) -> int:
        return sum(1 for claim in self.claims if claim.status == "verified")

    @property
    def problems(self) -> list[Claim]:
        return [claim for claim in self.claims if claim.status in PROBLEMS]

    @property
    def all_verified(self) -> bool:
        """True when there is something to verify and every statement of it was."""
        return bool(self.checked) and all(claim.status == "verified" for claim in self.checked)


def visible_text(block: ContextBlock) -> str:
    """Everything the answer was shown of a passage: its title as well as its text."""
    return f"{block.title}\n{block.text}"


@dataclass(frozen=True, slots=True)
class Statement:
    text: str
    markers: tuple[int, ...]


def split_statements(raw_text: str) -> list[Statement]:
    """The statements of an answer written with [n] markers, each with the markers that cite it.

    A marker belongs to the sentence it ends, whether it stands before the full stop or after it.
    """
    text = raw_text
    for abbreviation in _ABBREVIATIONS:
        text = re.sub(re.escape(abbreviation), abbreviation.replace(".", _DOT), text, flags=re.I)
    pieces = [match.group(0).strip() for match in _FRAGMENT.finditer(text)]
    sentences: list[tuple[str, list[int]]] = []
    for piece in pieces:
        piece = piece.replace(_DOT, ".")
        markers = cited_numbers(piece)
        words = strip_markers(piece)
        if not re.search(r"[A-Za-z0-9]", words):
            if markers and sentences:  # markers left alone after a full stop
                sentences[-1][1].extend(n for n in markers if n not in sentences[-1][1])
            continue
        sentences.append((words, markers))
    statements = []
    for words, markers in sentences:
        words = _BULLET.sub("", words).strip()
        # A cited fragment is a statement however short; an uncited one needs a few words to be
        # more than a heading or a pleasantry.
        if markers or len(words.split()) >= 3:
            statements.append(Statement(words, tuple(markers)))
    return statements


def code_spans(text: str) -> list[str]:
    """What sits between backticks, without whole commands: a placeholder or an option in a
    command varies from one document to the next, and the judge reads those."""
    spans = text.split("`")[1::2] if text.count("`") % 2 == 0 else []
    return [
        span.strip() for span in spans if 2 <= len(span.strip()) <= 80 and " " not in span.strip()
    ]


def values_of(statement: str) -> tuple[list[str], list[str]]:
    """The numbers (normalised) and the identifiers a statement commits to. The digits inside an
    identifier belong to it and are not counted again as numbers."""
    identifiers = code_spans(statement)
    remainder = _REFERENCE.sub(" ", statement.replace("`", " "))
    for pattern in _IDENTIFIERS:
        for found in pattern.finditer(remainder):
            token = found.group(0).strip()
            if token not in identifiers:
                identifiers.append(token)
        remainder = pattern.sub(" ", remainder)
    numbers = list(dict.fromkeys(filter(None, map(_normal_number, NUMBER.findall(remainder)))))
    return numbers, identifiers


def missing_values(statement: str, passages: Sequence[str]) -> list[str]:
    """Values of the statement that appear in none of the passages."""
    numbers, identifiers = values_of(statement)
    passage_numbers = {
        normal
        for passage in passages
        for normal in map(_normal_number, _ANY_NUMBER.findall(passage))
    }
    haystack = " ".join(_squash(passage) for passage in passages)
    missing = [number for number in numbers if number not in passage_numbers]
    missing += [token for token in identifiers if _squash(token) not in haystack]
    return missing


def quote_in_passages(quote: str, passages: Sequence[str], *, tolerance: float = 0.85) -> bool:
    """Whether the quote is in a passage, allowing for the small slips of copying by a model.

    Typographic quotes, dashes and markdown emphasis do not count as differences, and a quote with
    "..." in it is several quotes that must each be found.
    """
    pieces = [_squash(piece) for piece in re.split(r"\.{3}|\u2026", quote)]
    pieces = [piece for piece in pieces if len(piece) >= 8]
    if not pieces:
        return False
    texts = [_squash(passage) for passage in passages]
    return all(_piece_in(piece, texts, tolerance) for piece in pieces)


def _piece_in(wanted: str, texts: Sequence[str], tolerance: float) -> bool:
    for text in texts:
        if wanted in text:
            return True
        match = SequenceMatcher(None, text, wanted, autojunk=False).find_longest_match(
            0, len(text), 0, len(wanted)
        )
        if match.size >= tolerance * len(wanted):
            return True
    return False


class CitationVerifier:
    def __init__(self, judge: LLMClient | None = None, *, max_tokens: int = 4000) -> None:
        self._judge = judge
        self._max_tokens = max_tokens

    async def verify(
        self, raw_answer: str, blocks: Sequence[ContextBlock], *, use_judge: bool = True
    ) -> Verification:
        by_number = {block.number: block for block in blocks}
        pending: list[tuple[int, Statement, list[str]]] = []
        claims: list[Claim | None] = []
        for statement in split_statements(raw_answer):
            claim = self._cheap_checks(statement, by_number)
            claims.append(claim)
            if claim is None:
                cited = [visible_text(by_number[n]) for n in statement.markers if n in by_number]
                pending.append((len(claims) - 1, statement, cited))
        judge = self._judge if use_judge else None
        if judge is None:
            return Verification(
                [claim or self._unchecked(pending, index) for index, claim in enumerate(claims)],
                method="values",
            )
        return await self._judge_pending(judge, claims, pending, by_number)

    def _cheap_checks(
        self, statement: Statement, by_number: dict[int, ContextBlock]
    ) -> Claim | None:
        """The verdict the first two checks already settle, or None when a judge must decide."""
        markers = statement.markers
        if not markers:
            reason = _no_claim_reason(statement.text)
            if reason:
                return Claim(statement.text, markers, "no_claim", reason)
            return Claim(statement.text, markers, "uncited", "the statement has no citation")
        shown = [n for n in markers if n in by_number]
        if not shown:
            return Claim(
                statement.text, markers, "unresolved", f"cites {list(markers)}, never shown"
            )
        return None

    @staticmethod
    def _unchecked(pending: list[tuple[int, Statement, list[str]]], index: int) -> Claim:
        _, statement, cited = next(item for item in pending if item[0] == index)
        absent = missing_values(statement.text, cited)
        if absent:
            return Claim(
                statement.text,
                statement.markers,
                "ungrounded",
                f"not in the cited passages: {', '.join(absent)}",
            )
        return Claim(statement.text, statement.markers, "unchecked")

    async def _judge_pending(
        self,
        judge: LLMClient,
        claims: list[Claim | None],
        pending: list[tuple[int, Statement, list[str]]],
        by_number: dict[int, ContextBlock],
    ) -> Verification:
        verdicts: dict[int, tuple[str, str]] = {}
        usage: LLMUsage | None = None
        error: str | None = None
        if pending:
            try:
                reply = await judge.complete(
                    JUDGE_PROMPT,
                    _judge_input([item[1] for item in pending], by_number),
                    max_tokens=self._max_tokens,
                )
                usage = reply.usage
                verdicts = _parse_verdicts(reply.text, len(pending))
                if not verdicts:
                    error = "the judge did not return usable verdicts"
            except (LLMError, BudgetExceededError) as failure:
                error = f"the judge could not run: {type(failure).__name__}"
        done: list[Claim] = []
        for index, claim in enumerate(claims):
            if claim is not None:
                done.append(claim)
                continue
            position = next(i for i, item in enumerate(pending) if item[0] == index)
            _, statement, cited = pending[position]
            support, quote = verdicts.get(position + 1, ("", ""))
            if not support:
                done.append(self._unchecked(pending, index))
                continue
            done.append(_decide(statement, cited, support, quote))
        return Verification(done, method="judge", model=judge.model, usage=usage, error=error)


def is_absence(text: str) -> bool:
    """Whether a statement says that the documents do not hold something ("the passages do not
    give the default value")."""
    return ABSENCE.search(text) is not None


def is_heading(text: str) -> bool:
    return _HEADING.fullmatch(text.strip()) is not None


def _no_claim_reason(text: str) -> str | None:
    """Why an uncited statement asserts nothing that a passage could support, if that is so."""
    if is_heading(text):
        return "a heading"
    if is_absence(text):
        return "says the documents lack something"
    if _ABOUT_PASSAGES.search(text):
        return "comments on the passages themselves"
    return None


def _decide(statement: Statement, cited: Sequence[str], support: str, quote: str) -> Claim:
    """Combine the judge's verdict with what can be checked without it."""
    text, markers = statement.text, statement.markers

    def claim(status: Status, reason: str | None = None, evidence: str | None = None) -> Claim:
        return Claim(text, markers, status, reason, evidence, judged=support)

    if support == "no":
        return claim("unsupported", "the cited passages do not say it")
    if quote and not quote_in_passages(quote, cited):
        return claim("unsupported", "the evidence the judge gave is not in the cited passages")
    if support == "yes" and not quote:
        return claim("unsupported", "the judge gave no evidence for its verdict")
    absent = missing_values(text, cited)
    if support == "partial" or absent:
        reason = (
            f"not in the cited passages: {', '.join(absent)}"
            if absent
            else "the cited passages support only part of it"
        )
        return claim("partially_supported", reason, quote or None)
    return claim("verified", None, quote)


def _judge_input(statements: Sequence[Statement], by_number: dict[int, ContextBlock]) -> str:
    lines = []
    for number, statement in enumerate(statements, start=1):
        cited = [by_number[n] for n in statement.markers if n in by_number]
        passages = "\n\n".join(
            f"Passage [{block.number}] {block.title} ({block.source_type})\n{block.text}"
            for block in cited
        )
        lines.append(f"Statement {number}: {statement.text}\n\nCited passages:\n\n{passages}")
    return "\n\n-----\n\n".join(lines)


def _parse_verdicts(text: str, count: int) -> dict[int, tuple[str, str]]:
    found = re.search(r"\{.*\}", text, re.DOTALL)
    if not found:
        return {}
    try:
        rows = json.loads(found.group(0)).get("verdicts", [])
    except (ValueError, AttributeError):
        return {}
    verdicts: dict[int, tuple[str, str]] = {}
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        number, support = row.get("claim"), str(row.get("support", "")).lower().strip()
        if isinstance(number, int) and 1 <= number <= count and support in {"yes", "partial", "no"}:
            verdicts[number] = (support, str(row.get("quote") or ""))
    return verdicts


def _normal_number(token: str) -> str:
    try:
        return format(Decimal(token.replace(",", "")).normalize(), "f")
    except InvalidOperation:
        return ""


_TYPOGRAPHY = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u00a0": " ",
        "*": "",
        "`": "",
        "\\": "",  # passages exported as JSON keep the backslashes of escaped quotes
    }
)


def _squash(text: str) -> str:
    """The text as it is compared: lower case, plain quotes and dashes, no emphasis marks, single
    spaces."""
    return re.sub(r"\s+", " ", text.translate(_TYPOGRAPHY)).strip().lower()
