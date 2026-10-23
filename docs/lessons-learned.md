# Lessons learned

What building this taught, in the order it mattered. Each item says what happened, what it cost or
saved, and what to do next time. It is a list of things that surprised the author or that a
reader building a similar system could use; the measurements are in [results.md](results.md).

## Measuring

**1. The first comparison was flattering and the audit said so.** The headline (50.6 against the
agent's 40.0) came from a 96-question sample. Re-running the pipeline on all 470 questions gave
48.9, a second run on the same 96 gave 46.3, and the sample turned out to favour the reranker
(+0.154 recall there, +0.055 on the rest). Nothing was wrong with the first number; it was one
draw. *Next time:* run the second sample and the repeat before writing the headline, and report
the interval next to it. The gap to the iterating agent stayed positive in every run, but its size
never settled.

**2. A fairness audit is worth more as code than as prose.** Nine checks (same questions, same
model, same judge, same evidence budget, costs measured, and so on) run on every comparison and
refuse to publish if one fails. During the end-to-end run a usage report showed zero calls for a
run that had clearly made some; it did not happen again in two further runs, so its cause is
unknown, and the audit now rejects any report with no calls and the proxy's log is kept next to
it. A check written after a surprise is a smaller thing than understanding the surprise, and this
page does not claim to have done the second.

**3. Replay beats re-running.** Collecting candidates from the index takes half an hour, the
reranker scores a GPU and a Kaggle queue; answers cost money. Storing what the retriever returned
and replaying it through the same fusion and reranking code made every later experiment minutes
and cents, made CI able to check retrieval quality without the index (a 0.9 MB fixture), and let a
test check that the replay matches the live service. The price: a fixture is only as current as
its last rebuild, and a change to the index itself needs the real services.

**4. Judge noise was small; generation noise was not.** Asked twice about the same answers, the
judge changed 2 verdicts of 96. The model writing the answers moved the score by 4 points between
two runs. Spend repetitions on the generation, not on the judging.

## Retrieval

**5. On identifier-heavy text, BM25 beat the dense model.** Document recall at five chunks was 0.60
for BM25 and 0.49 for dense search on a corpus full of flag names, pool IDs and region codes.
Fusing them added only two points, and a cross-encoder reranker six more. Leading with the dense
model, as most tutorials do, would have been the wrong default here.

**6. The reranker was the single largest gain and the single largest dependency.** It needs a GPU
for the latency to be reasonable (between 46 and 554 milliseconds for 30 passages on a T4,
depending on the model, against about eight seconds for the smallest on the laptop CPU). A design that cannot afford the GPU should plan for the lighter model
and a smaller pool, and say what that costs in recall.

**7. The weak categories are retrieval, not writing.** `semantic`, `project_related` and
`completeness` questions score 26, 7 and 28 because five passages cannot hold an answer that is
spread over many documents or described without its own words. The benchmark's grep agent wins
on exactly those. The next step is to retrieve again when the first pass is thin; no amount of
better prompting fixes it.

**8. Retrieval measured on a laptop misses the latency target, and said so only because it was
measured.** A cold query took 10 to 27 seconds on a 6 GB Docker VM with the index mostly on disk.
The target of one second was written before the measurement and is documented as missed, not
quietly relaxed.

## Grounding

**9. Checking citations is a pipeline of cheap, certain checks and one judge.** Does the cited
passage exist? Are the numbers and identifiers of the statement in it? Only then ask a model
whether it supports the statement, and make it copy the sentence, which is then looked up in the
passage. The plain-text checks lifted detection of a moved citation from 87% to 99%; a judge that
has to quote its evidence cannot verify with invented evidence.

**10. Hand-reading beat any metric for finding false alarms.** Planted-error tests said the
verifier was good; reading the statements it flagged found six separate causes of false alarms
(timestamps hiding a number, a date that appears only in a title, "passage 5" read as a value,
curly quotes in the judge's evidence, and others), each fixed once and tested. The remaining false
alarm rate is estimated at one in six, not measured.

**11. What the model says about itself is a better abstention signal than the retriever's score.**
Answers that state that something is missing were right 15% of the time; the rest 84%. A
threshold on the best reranker score separated right from wrong answers with probability 0.68 and
the benchmark's unanswerable questions, which are written to look answerable, with 0.82. The score
threshold ships off; the status that reads the answer's own wording ships on.

**12. Prompt variants that did not help were removed.** A prompt asking for a marker on every
sentence did not change the share of uncited statements beyond run-to-run noise, and one asking
for a caveat when the passages hold only part of the answer gave the same 20 of 20. Both were
dropped. A tuned prompt that cannot be shown to help is only more text to maintain.

## Operating it

**13. A load test found a bug the unit tests never could.** At four concurrent searches on the
laptop, a Qdrant query outlasted its 30-second deadline and the API answered `500` with a stack
trace instead of `503 retrieval unavailable`. Every unit test passed because none made a backend
slow. The adapters now translate timeouts and connection failures, and a test covers each. *Next
time:* test the failure of every dependency, not just its success.

**14. Fail open or fail closed is a decision to write down.** If Redis is down, the rate limiter lets
requests through and counts the failure, because refusing every caller over a counter outage turns
a cache problem into an outage. That has a cost (the limit is not enforced while Redis is down),
which is why the failure counter is the thing to alert on.

**15. Secrets need a policy before they need a tool.** One API key was pasted into a chat early
on. It lives only in a git-ignored `.env`, never in a log, a metric label or a commit, the
repository was scanned for key patterns before each commit, and the advice remains to revoke the
pasted key. Per-client keys carry a name so that logs and metrics identify the caller without ever
holding the secret.

## Process

**16. Limited hardware is a design input, not an excuse.** An 8 GB laptop could not embed three
million chunks (projected 80 to 125 hours), so the embeddings were computed on a free Kaggle GPU in
2.16 hours and shipped back as Parquet shards that describe themselves. Heavy jobs on borrowed
GPUs also meant the Kaggle queue sometimes took an hour for a minute of work; re-pushing with a
different accelerator unblocked one.

**17. Say what was not done.** Conflict-aware answering as a separate step, RAGAS metrics, a
hand-written golden set, document upload, tracing and TLS are in the original design and are not in
the code; the README says which. A reader can trust a "done" more when the "not done" list is next
to it.
