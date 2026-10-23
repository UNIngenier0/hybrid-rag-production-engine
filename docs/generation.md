# Answers and citation verification

`POST /v1/query` answers a question from the indexed documents and checks, for every statement of
the answer, that the passage it cites really says it. This page explains how an answer is made,
how a citation is verified, what the verifier was measured to catch, and what it does not do.

Conflict-aware answering and the RAGAS metrics are the rest of Phase 5 and are not here yet.

## The endpoint

```bash
curl -s localhost:8000/v1/query -H 'content-type: application/json' -d '{
  "query": "default wait time of the tool-call batching invoker"
}'
```

It needs `ANTHROPIC_API_KEY` in `.env` (without it the endpoint answers `503` and says so) and,
for good answers, the index and a reranker (`RERANKER_MODEL`). A real response, shortened:

```json
{
  "mode": "hybrid+rerank",
  "status": "answered",
  "gaps": [],
  "abstained": false,
  "abstention": null,
  "answer": "The default grace period is 10ms. ... The grace period is configurable through a per-request config and a runtime policy override.",
  "answer_with_markers": "The default grace period is 10ms [1][2]. ... a runtime policy override [1][2].",
  "citations": [
    {"marker": 1, "doc_id": "dsid_c786...", "title": "Coalescing Invoker: grace-period batching, ...",
     "source_type": "github", "verified": true, "claims": [0, 2]}
  ],
  "claims": [
    {"text": "The default grace period is 10ms.", "markers": [1, 2], "status": "verified",
     "reason": null,
     "evidence": "introduces a configurable grace period (default 10ms) to allow closely timed tool calls ..."}
  ],
  "verification": {"method": "judge", "model": "claude-haiku-5-5", "statements": 3,
                   "verified": 3, "all_verified": true, "error": null},
  "document_ids": ["dsid_c786..."],
  "usage": {"model": "claude-haiku-5-5", "input_tokens": 8997, "output_tokens": 1322, "cost_usd": 0.0016},
  "timings_ms": {"retrieval_total": 27298.1, "generation": 3404.6, "verification": 4622.0, "total": 54605.2}
}
```

(The timings are from the 6 GB development laptop with cold caches while other jobs ran: the
retrieval stages are explained in [retrieval.md](retrieval.md#latency-on-the-development-laptop).
Generation took 3 to 4 seconds and verification 2 to 6 in every call made.)

| Request field | Meaning |
|---|---|
| `query` | The question, up to 2,000 characters |
| `mode` | `dense`, `bm25`, `hybrid` or `hybrid+rerank`. Left out, the server uses `hybrid+rerank` when a reranker is configured and `hybrid` if not |
| `chunks` | Passages the answer is written from, 1 to 10 (`QUERY_CHUNKS`, 5 by default) |
| `filters` | `source_types` and `doc_ids`, as in `/v1/search` |
| `verify` | Run the judge that checks each citation (`VERIFY_CITATIONS`, on by default). With `false`, no statement is ever called `verified` |

Errors: `422` for an invalid request, `503` when retrieval is down or no key is set, `502` when the
model fails. The cause stays in the server logs.

## How an answer is written

The pipeline is the one the benchmark comparison measured: hybrid retrieval, a cross-encoder
reranking of the best 30 candidates, the best 5 passages shown to the model together with their
parent text, numbered `[1]` to `[5]`, and a prompt that asks for plain prose in which each
statement ends with the numbers of the passages that support it. The prompt is the same one the
evaluation uses (`generation/prompt.py`), so the numbers in [evaluation.md](evaluation.md) describe
this answer step. A stricter version of the prompt, asking for a marker on every sentence, was
tried and did not change the share of uncited statements beyond the run-to-run noise, so it was not
kept.

## How a citation is verified

The answer is split into statements (sentences and list items), each with the markers that cite
it. A statement is `verified` only if it passes three checks, in this order:

1. **The citation exists.** A marker that points at a passage that was never shown is `unresolved`;
   a statement of fact with no marker is `uncited`. Headings and statements saying the documents
   lack something ("the policy is not described") assert nothing a passage could support and are
   `no_claim`.
2. **The values are in the passage.** Numbers, identifiers, file names and code spans of the
   statement are looked for as plain text in the cited passages and in their titles (an answer
   may legitimately use a date from a title). This check cannot be argued with, and it is what
   catches a model that rounds, mixes up or invents a figure.
3. **A judge confirms it, with evidence.** A second model call, which sees only the cited
   passages, says whether they support the statement (`yes`, `partial`, `no`) and copies the
   sentence that does. That quote has to be found in the cited passage, allowing for typographic
   quotes, markdown and ellipses. A judge that invents its evidence verifies nothing.

| Status | Meaning |
|---|---|
| `verified` | The judge found the statement in the cited passages, its quote is real, and every value is there |
| `partially_supported` | The passages support part of it (a clause, a qualifier, a number is missing) |
| `unsupported` | The passages do not say it, or the evidence the judge gave is not in them |
| `uncited` | A statement of fact with no citation |
| `unresolved` | Every marker points at a passage that was not shown |
| `ungrounded`, `unchecked` | Without a judge (or when it failed): a value is missing from the passages, or nothing was found wrong but nothing was checked |
| `no_claim` | A heading, or a statement about what the documents lack: nothing to verify |

`all_verified` is true only when there is something to verify and every statement of it was. A
citation is marked `verified` when every statement that cites it was. If the judge cannot run, the
answer still comes back, with `verification.error` set and the statements `unchecked`: a failure of
the check never passes as success.

## What the verifier was measured to do

`python -m rag_engine.evaluation.cli verify-check` writes 96 real answers (12 questions of each of
the eight types with gold documents, the same selection as the answer comparison), verifies them,
and then damages the verified statements to see whether the verifier notices
([report](../eval/reports/citation-verification.md)). The last run, with the final verifier:

| Status | Statements | Share |
|---|---|---|
| verified | 442 | 66.3% |
| uncited | 119 | 17.8% |
| no_claim | 59 | 8.8% |
| partially_supported | 38 | 5.7% |
| unsupported | 9 | 1.3% |

24 of the 96 answers had every statement verified. These figures move between runs, because the
answers do: across five runs made while the verifier was being fixed (it changed a little between
them) the verified share was 60% to 66%, the uncited share 17% to 22%, and the answers with every
statement verified 16 to 24 of the 96.

| Damage planted in a verified statement | Planted | Caught by the judge alone | Caught by the whole check |
|---|---|---|---|
| Cite another passage of the same question | 354 | 87% | 99% |
| Change a number to one the passage does not contain | 184 | 96% | 98% |

For a moved citation, 30 of the 354 are ambiguous: the other passage also holds every value in the
statement, so it may support it. The value check is what lifts the judge from 87% to 99% on moved
citations, which is why it is there.

**Reading statements by hand.** Detection of planted damage does not say whether the verifier is
right about untouched statements, so the statements it flagged and passed were read next to their
passages while it was being built. On the version before the final fixes, 27 of 30 verified
statements were supported, 1 was not (a proposal reported as done) and 2 could not be settled. Most
`partially_supported` statements really did say more than their evidence (an invented "only", an
inferred cause, a second clause). The false alarms found this way were fixed one at a time:
timestamps such as `T09:00Z` that hid a number, dates taken from a title, the answer's own
"passage 5" read as a value, two code spans read as one, curly quotes, ellipses and escaped quotes
in a judge's evidence. In the last full read of the flagged `unsupported` statements, 11 of 16 were
rightly flagged, 3 were doubtful and 2 were false alarms, one of them (escaped quotes) fixed
afterwards. The final version was not read in full again, so treat the false-alarm rate of the
`unsupported` and `partially_supported` statuses as roughly one in six, not as measured.

**What this means.** The verifier is strict: it asks that the cited passage state the whole
statement, and about one answer in four passes completely. The largest group of problems is not a
wrong citation but a missing one (18% of statements carry no marker at all), which is the model not
following the instruction to cite. The share of `unsupported` statements, the ones whose citation
points at a passage that does not say them, is 1% to 3%.

## When the documents do not hold the answer

Every response says what the answer claims about its own completeness. `status` is one of:

| `status` | Meaning |
|---|---|
| `answered` | The answer does not say anything is missing |
| `partial` | The answer states part of what was asked and says the documents lack the rest. `gaps` quotes those sentences |
| `not_found` | The answer says only that nothing was found |
| `abstained` | No answer was written: the service declined before calling the model (see below) |

`partial` and `not_found` are read from the answer's own sentences ("the passages do not give the
default value", "I could not find the contract address"), with the same patterns the verifier uses to
recognise a statement about absence. It needs no model call, so it is there when `verify` is off.

**What it was measured to do** ([report](../eval/reports/abstention-scores.md), recomputed from the
committed runs and the regression-gate fixture in every build). The 470 questions that have gold
documents, answered by this pipeline and judged by the benchmark's evaluator:

| Answer says | Answers | Judged correct |
|---|---|---|
| nothing is missing (`answered`) | 261 | 219 (84%) |
| something is missing (`partial`, `not_found`) | 209 | 32 (15%) |

The flag marks 177 of the 219 wrong answers and 32 of the 251 right ones. Fourteen of those 32 were
read: every one was a real gap (a detail the answer says the documents lack) that the gold answer
did not need. So the flag is a reliability signal, not a correctness verdict.

The benchmark has 20 `info_not_found` questions, written so that related documents exist and the
asked-for detail does not. The same pipeline, with the prompt unchanged, acknowledges that the
information is missing in 20 of 20 (the evaluator's one fact for these questions, scored 100.0), and
every one of the 20 answers is `partial`. A prompt variant that explicitly asked for a caveat when
the passages cover only part of the question was tried on the same questions (with hybrid retrieval, before the
reranker scores were available) and gave the same 20 of 20, so it was not kept. The 10 `high_level` questions, which need many documents put together, are the weak
spot: 6 of 10 correct (48.3 combined), and only 4 of the 10 answers say something is missing.

**Declining before writing.** The service also declines without calling the model in two cases:
nothing was retrieved (always), and, only if `ABSTAIN_THRESHOLD` is set and the mode is
`hybrid+rerank`, the best reranked passage scores below it. The response then has `status:
"abstained"`, `abstained: true` and `abstention` with the reason, the best score and the threshold;
no tokens are spent.

The threshold is off by default because the score is a weak signal on this benchmark. It is the
cross-encoder's probability (0 to 1) for `bge-reranker-v2-m3`, and a value is only valid for the
reranker that produced it:

| Answerable questions refused | Threshold | `info_not_found` caught (of 20) | Wrong answers avoided (of 219) | Right answers given up (of 251) |
|---|---|---|---|---|
| 1% | 0.225 | 0 | 3 | 1 |
| 2% | 0.318 | 2 | 8 | 1 |
| 5% | 0.495 | 5 | 21 | 2 |
| 10% | 0.630 | 9 | 36 | 11 |
| 20% | 0.822 | 15 | 62 | 32 |

A random answerable question has a higher best score than a random `info_not_found` one 82% of the
time, and a correct answer has a higher one than a wrong answer 68% of the time: the related
documents that make these questions hard also score high, so a score cannot tell them apart well.
The model's own statement of what is missing is the stronger signal (it marks 177 of 219 wrong
answers, against 36 for a threshold that costs 11 right answers). A threshold near 0.3 is a cheap
safety net for questions with nothing relevant in the corpus (2% of answerable questions refused,
and 8 of the 9 were answered wrongly anyway), which this benchmark has no questions to exercise.

## Cost and speed

An answer costs about $0.001 to $0.002 with Claude Haiku 5.5 (the answer call plus the judge call),
and `verify-check` as a whole, with 96 answers and the planted-damage judging, cost $0.30. On the
development laptop generation took 3 to 4 seconds and verification 2 to 6 seconds, after the
retrieval; the judge runs after the answer, so it adds that time to every request. Setting
`VERIFY_CITATIONS=false` removes it. A stronger judge model (`VERIFICATION_MODEL`) would be stricter
and cost more.

| Setting | Default | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | none | Enables `/v1/query` |
| `LLM_MODEL` | `claude-haiku-5-5` | The model that writes the answer |
| `VERIFICATION_MODEL` | the same | The model that judges the citations |
| `VERIFY_CITATIONS` | `true` | Whether a request without `verify` runs the judge |
| `QUERY_CHUNKS` | 5 | Passages the answer is written from |
| `ABSTAIN_THRESHOLD` | none | Decline, without calling the model, when the best reranker score is below this |

## Limits

- **The judge is a small model of the same family as the writer.** It shares blind spots with it.
  Against planted damage it is good (87% to 96% alone); against subtle misreadings it is not
  measured, beyond the hand reading above.
- **Verification says the cited passage states the answer, not that the answer is right.** If the
  passage is wrong, or the right passage was not retrieved, a faithful statement still verifies.
  Answer correctness is what the [answer comparison](evaluation.md#answer-evaluation) measures.
- **Statements are sentences.** A sentence that makes several claims is judged as one, and the
  judge's quote usually covers only one of them. The value check covers the numbers and names in
  the rest, but not a clause that has none.
- **Absences are not verified.** "The documents do not give X" cannot be confirmed by reading the
  passages that were shown; with a citation it is judged like any other statement (and often comes
  back `unsupported`), without one it is `no_claim`.
- **No streaming, no conversation.** One question, one complete answer.
- **`status` reads wording.** An answer that lacks something but says so in a way the patterns do
  not know is `answered`, and a sentence that merely contains "does not include" about the subject
  itself counts as a gap. None of the 14 flagged correct answers that were read was a wrong flag,
  but the false-flag rate in general is not measured.
- **Broad questions.** Questions that need many documents put together (`high_level`) are limited by
  the five passages the answer is written from: 4 of 10 were wrong.
