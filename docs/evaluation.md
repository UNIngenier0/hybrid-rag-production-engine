# Retrieval evaluation

How retrieval quality is measured, and how to repeat the measurement. The results are in
[`eval/reports/retrieval-ablation.md`](../eval/reports/retrieval-ablation.md); the findings and
what they decided are summarized at the end of this page.

## What is measured

The benchmark labels **documents**, not chunks: each question lists the documents that answer it.
A retriever returns chunks, several of which can come from one document, so every ranking is first
reduced to the distinct documents in the order their best chunk appears. Then, per question:

| Metric | Meaning |
|---|---|
| Recall | Share of the question's gold documents that were found |
| Any gold | At least one gold document was found |
| All gold | Every gold document was found |
| MRR | 1 / rank of the first gold document |
| nDCG@10 | Binary-relevance nDCG over the first 10 documents |

Everything is scored at a **chunk budget** (5 or 10 chunks returned), because the budget is what
reaches the LLM. 470 of the 500 questions have gold documents; the other 30 (information that is
not in the corpus, and high-level questions) cannot be scored by retrieval and belong to the answer
evaluation. Differences between systems are paired by question and reported with a 95% interval
(normal approximation), so a gain is only called real when the interval excludes zero.

## How it runs

Collecting candidates is the slow part, so it is done once and everything else is computed from the
stored lists. Every ranking in the report is rebuilt with the same fusion and reranking code the
API uses (a test checks that the offline ranking equals what the service returns).

```bash
# 1. Search every question and keep the first 100 chunks of both retrievers.
#    Needs Qdrant and the embedding model. About 27 minutes on the development laptop.
python -m rag_engine.evaluation.cli collect

# 2. Score the fused candidates with a cross-encoder. Needs only the candidates file.
#    On a CPU this is slow (see docs/evaluation.md#reranker-cost), so use a GPU:
python -m rag_engine.evaluation.cli export-pool          # the compact file to ship (11 MB)
uv run --group kaggle python scripts/kaggle/rerank.py run \
    --username <kaggle-user> --wheel dist/<wheel>        # five models in about 20 minutes

# 3. Write the ablation.
python -m rag_engine.evaluation.cli report \
    --rerank bge-base=eval/cache/rerank-BAAI_bge-reranker-base.jsonl.gz \
    --rerank minilm-12=eval/cache/rerank-cross-encoder_ms-marco-MiniLM-L-12-v2.jsonl.gz
```

All commands are resumable: collection and scoring skip the questions already stored. The caches
live in `eval/cache/` (git-ignored); the reports in `eval/reports/` are committed.

On the Docker setup, run them in the worker image or any environment with the project installed;
the Kaggle step needs the Kaggle CLI (`uv sync --group kaggle`) and the same account as the
embedding job ([kaggle-embedding.md](kaggle-embedding.md)).

## Limits to keep in mind

- **One run, one retriever configuration.** The candidates come from the index as it is (HNSW
  search effort 64, int8 vectors with rescoring). The exact dense search of the earlier ceiling
  study is slightly better, so the dense numbers here are a little lower than the exact ones.
- **Selection on the evaluation questions.** Fusion weights and the reranker are chosen on the same
  470 questions they are reported on. There is no held-out set, so small differences between the top
  settings are not conclusive; the split-half check in the findings shows how stable a choice is.
- **Retrieval is not the answer score.** Recall says whether the right documents reach the LLM, not
  whether the answer is right. The answer evaluation (Phase 4b) measures that.

## Findings

470 questions, recall at the chunks that reach the LLM (5), against the hybrid baseline of 0.608.
Intervals are paired 95% intervals; the full tables are in the report.

- **Reranking helps, and the model matters.** `bge-reranker-v2-m3` over the first 30 fused chunks
  raises recall from 0.608 to **0.684** (+0.076, interval +0.046 to +0.105), the share of questions
  with at least one gold document from 0.691 to 0.755, and the `semantic` category from 0.304 to
  0.424. That closes about half of the gap to the pool's ceiling (0.768 for 30 chunks). The
  base-size cross-encoders gain +0.03 to +0.05, and the small MiniLM models mostly do not beat the
  noise (+0.01 to +0.02 at 5 chunks, intervals that include zero).
- **A pool of 20 to 30 chunks is enough.** Scoring 50 gives no further gain at 5 chunks and costs
  almost twice as much.
- **Fusing deeper is worse.** Using the first 100 chunks of each list instead of 50 lowers recall
  at 5 chunks by 0.023: the extra candidates are mostly noise.
- **BM25 is the stronger retriever here** (0.601 against 0.491 for dense at 5 chunks), and a BM25
  weight of 1.5 in the fusion adds +0.014 at 5 chunks and +0.025 at 10 (both intervals exclude
  zero). Split by even and odd questions, 1.5 gains on both halves at 10 chunks (+0.028 and +0.027)
  and at 5 chunks the gain is smaller and noisier (+0.007 and +0.020), so 1.5 is the default
  rather than the larger weights. The reranker study used equal weights.
- **The approximate index costs some dense recall.** Fifty dense chunks from the index (HNSW search
  effort 64, int8 vectors) contain 0.645 of the gold documents; the exact search of the earlier
  ceiling study found 0.679.
- **What stays low** is a first-stage problem that a reranker cannot fix: `completeness` (0.27 to
  0.37 with the best reranker), `project_related` (0.38 to 0.41) and `semantic` (0.42). The gold
  documents are often not among the candidates, which points at query rewriting and keywords for
  the BM25 side.

### Reranker cost

On a Tesla T4 in half precision, scoring one (query, passage) pair takes 1.5 ms (MiniLM-L6), 2.5 ms
(MiniLM-L12), 6.0 ms (bge-reranker-base), 12.8 ms (mxbai-rerank-base) and 18.5 ms
(bge-reranker-v2-m3), so 30 passages take between 46 ms and 554 ms. A CPU is far slower: on two
cores of the development laptop MiniLM-L12 took 203 ms per pair, and a run of bge-reranker-base
did not finish 300 pairs in over five minutes, which puts 30 passages at tens of seconds. The
sub-second retrieval target therefore needs either a GPU or a machine with many more cores, and the
CPU figures on the second machine are still to be measured.

| Deployment | Reranker | Pool | Recall at 5 chunks |
|---|---|---|---|
| GPU available | `BAAI/bge-reranker-v2-m3` | 30 | 0.684 |
| Smaller GPU or fast CPU | `BAAI/bge-reranker-base` | 20 | 0.651 |
| No reranker | none (BM25 weight 1.5) | | 0.622 |

## Regression gate

CI cannot run the real index, so it replays a fixture instead (`eval/baselines/gate-fixture.jsonl.gz`,
0.9 MB): for each of the 470 questions with gold documents, the ranked chunks that dense search and
BM25 returned from the real index (their documents and ranks, no text) and the reranker's score for
the fused pool. The gate runs fusion, reranking and the metrics through the same code the service
uses, with the settings the service ships with (read from the defaults of `Settings`), and compares
four systems at 5 and 10 chunks with the committed baseline (`eval/baselines/retrieval.json`).

```bash
python -m rag_engine.evaluation.cli gate                    # what CI runs
python -m rag_engine.evaluation.cli gate --update-baseline  # after reviewing a deliberate change
python -m rag_engine.evaluation.cli build-gate-fixture --rerank <scores file>   # new candidates
```

- **What fails the build.** An overall metric (recall, any-gold hit, MRR, nDCG) that drops by more
  than 0.005, a recall by question type that drops by more than 0.02, a fixture that changed
  without a new baseline, or a shipped pipeline that differs from the one the baseline was
  measured with. Improvements never fail; the gate prints them so the baseline can be raised.
- **It is deterministic**, so any difference comes from a change in fusion, reranking, metrics or
  settings. Today it reproduces the ablation exactly (hybrid 0.622 and hybrid + rerank 0.684 at 5
  chunks).
- **A new pool needs new scores.** If a settings change makes the reranker see chunks the fixture
  has no score for, the gate stops with an error instead of guessing: score the new pool (on a GPU)
  and rebuild the fixture.
- **What it cannot see.** A change to the index itself (another embedding model or other chunking)
  and anything that needs the live services or the model API. The answer-side check is a different
  one: it verifies that the published comparison is consistent with the run folders it is built
  from, and that the systems in it were compared like for like (next section).

## Answer evaluation

Retrieval recall says whether the right documents reach the model; this part measures the answers
and compares the system with two grep baselines, using the benchmark's own evaluator.

### What is compared

All three systems use the same model (Claude Haiku 5.5) on the same 96 questions, and the same
judge scores them.

| System | How it finds documents | Then |
|---|---|---|
| **Hybrid + rerank (ours)** | Dense + BM25, fused, 30 candidates reranked by `bge-reranker-v2-m3`; the best 5 chunks reach the model with their parent passages | One answer step, with citations |
| **Iterative grep agent** | The benchmark's own agent: a loop that runs `grep`, `find`, `jq` and similar commands over the raw documents, reads files and picks documents, with a 10-minute budget per question | Writes the answer itself |
| **Single-attempt grep** | A model writes one regular expression, `ripgrep` runs it once, the five files with the most matches are kept (the part of each that holds the matches) | The same answer step as ours |

The 96 questions are 12 of each of the eight types that have gold documents, drawn with a fixed
seed (`eval/runs/*/questions.txt`). The two types without gold documents (information that is not in
the corpus, and high-level questions) are Phase 5, because they test abstention.

The evaluator is the benchmark's `metrics_based_eval`: *correctness* is a judge's yes or no on
whether the answer agrees with the gold answer, *completeness* is the share of gold facts the
answer contains (each checked on its own), *document recall* is the share of gold documents the
system names, and *invalid extra documents* counts the documents it names that are not gold. The
*combined* score is the benchmark's headline: completeness averaged over all questions, counted as
zero when the answer is not correct. Every system is scored against the original gold
(`--no-correction`) so they stay comparable, and the judge is Claude Haiku 5.5, not the GPT model of
the public leaderboard, so none of these numbers can be compared with that leaderboard.

### Results

| System | Correct | Completeness | Combined | Doc recall | Invalid extra docs | Combined vs ours | Cost of the answers |
|---|---|---|---|---|---|---|---|
| **Hybrid + rerank (ours)** | 56.2% | 61.8% | **50.6** | 72.3% | 0.95 | | $0.08 |
| Iterative grep agent | 42.7% | 49.8% | **40.0** | 38.2% | 0.50 | -10.7 (-20.6, -0.7) | $1.94 |
| Single-attempt grep | 5.2% | 15.8% | **4.8** | 9.0% | 1.72 | -45.8 (-55.5, -36.2) | $0.09 |

By question type, the combined score (12 questions each, so read the types as directions):

| System | basic | semantic | intra-doc | constrained | conflicting | misc. | project | completeness |
|---|---|---|---|---|---|---|---|---|
| Hybrid + rerank | 56.9 | 33.3 | 58.3 | 67.2 | 73.1 | 95.6 | 3.8 | 16.7 |
| Iterative grep agent | 33.3 | 23.6 | 58.3 | 23.1 | 29.8 | 100.0 | 9.9 | 41.5 |
| Single-attempt grep | 8.3 | 0.0 | 0.0 | 14.7 | 6.9 | 8.3 | 0.0 | 0.0 |

- **One pass of hybrid retrieval and reranking came out ahead of the iterating agent**, by 10.7
  points on the combined score in this run and at about a twenty-fifth of the cost: $0.0008
  against $0.0202 per question. It also found much more of the gold evidence (72% of the gold
  documents against 38%). How far ahead is less settled than the interval suggests: a second run
  of our pipeline on the same questions was ahead by 6.4 (interval -4.7 to 17.4), so the direction
  is consistent and the size is not ([fairness audit](#fairness-audit)).
- **The agent wins exactly where one pass cannot reach**: `completeness` (41.5 against 16.7) and
  `project_related` (9.9 against 3.8), the questions that need many documents put together. Both
  are low for everyone, and they are the case for a retrieve-check-retrieve-again step.
- **The agent is more precise**: it names fewer unrelated documents (0.50 against 0.95 per question)
  because it only selects what it read.
- **A single grep attempt is not a baseline worth fearing**: with no ranking beyond match counts, a
  common word brings back the longest threads, and for 74 of the 96 questions none of the five
  files held the answer (only 13 had any gold document among them).

### Fairness audit

A comparison is only worth reading if it is like for like. Two kinds of checks back this one.

**Checked by code on every comparison.** `compare` refuses to write a report unless these hold, and
CI re-checks them on the committed runs and verifies that the published numbers are what the run
folders say (`eval/reports/answer-comparison.md` ends with the table):

| Check | What it guarantees |
|---|---|
| Same questions | Every system answered the same 96, and each answered all of them |
| Every answer scored | Nothing was dropped between answering and judging |
| Same gold | The evaluator did not rewrite the gold answers for any system (`--no-correction`) |
| Same judge | One judge model scored every system, confirmed by the usage report of each judging run |
| Same answering model | Claude Haiku 5.5 behind every system, confirmed by each run's usage |
| Same evidence budget | Our RAG and the single-attempt grep each hand the model 5 passages or files |
| Same document rule | Both report the documents their answer cites |
| Cost measured | Each system has a usage report with calls in it, or token counts; a metering report with no calls cannot pass as a cost of zero |

What differs on purpose is recorded in the report: the agent decides for itself how much to read
(its cost is reported next to its score), and its answers end with a source list, so the evaluator
strips citations before judging, as the benchmark does by default.

**Measured by `scripts/analysis/fairness_audit.py`** (offline, from the stored candidates and runs):

- **The 96 questions flatter the reranker.** Reranking adds 0.154 document recall at 5 chunks on
  those 96 questions, against 0.055 on the other 374 (0.076 on all 470). The best reranker is the
  same on both groups, so the choice is not an artifact of the sample, but the sample's retrieval
  advantage is larger than average.

  | Questions | Hybrid | + `bge-reranker-v2-m3` | + `bge-reranker-base` | + `mxbai-rerank-base` |
  |---|---|---|---|---|
  | the 96 compared | 0.574 | 0.728 (+0.154) | 0.656 (+0.082) | 0.645 (+0.071) |
  | the other 374 | 0.617 | 0.673 (+0.055) | 0.637 (+0.020) | 0.658 (+0.040) |

- **Our pipeline on every question with gold documents scores 48.9** (53.4% correct), against 50.6
  on the 96. The same 96 answered a second time, with a new generation and a new judgment, scored
  46.3. One run on 96 questions therefore moves by about four points on its own.
- **The judge is steady.** Judging the same answers a second time changed the combined score by at
  most 0.8 points (50.6 to 50.4, 40.0 to 39.2, 4.8 to 4.8) and flipped 2, 2 and 0 of 96 verdicts. The
  noise is in the answers, not in the scoring.
- **The gap to the agent is not settled.** Ours minus the agent is +10.7 (+0.7 to +20.6) for the
  first run and +6.4 (-4.7 to +17.4) for the second. Leaving out the three questions used to try the
  prompts of each system gives +9.0 (-1.0 to +19.0). The agent ran once, and its own noise is
  unknown: in its pilot it answered one of those three questions correctly and in the full run it
  did not.
- **The single-attempt grep is not held back by the search limit.** Its search returned no file for
  13 of 96 questions (it never finished within the limit, or matched nothing). Had it answered all
  13 perfectly it would score 18.3, still far below. What does hold it back is the ranking: its
  patterns match thousands of files (4,874, 4,066 and 105,700 in a trial), and the five with the
  most matches are rarely the right ones. New runs record every pattern, the number of files it
  matched and whether the search was abandoned (`searches.jsonl`).
- **What was evaluated is what ships.** Reranking used a pool fused with equal BM25 weight, while
  the service's plain hybrid mode weighs BM25 1.5. The service now reranks the equal-weight pool
  (`RERANK_SPARSE_WEIGHT`), and the running service was checked against the replay on 32 questions
  ([retrieval.md](retrieval.md#checked-against-the-running-service)). Heavier BM25 showed no gain
  after reranking (-0.007, interval -0.026 to +0.011).

What can be said: against a single grep attempt the difference is large and robust. Against the
iterating agent at equal model strength, one pass of hybrid retrieval and reranking is ahead by
roughly 6 to 11 points at a twenty-fifth of the cost, but the 96 questions cannot separate that from
no difference, and the agent is better on `completeness` and `project_related` questions. Settling it
takes a second, disjoint sample of questions for every system, preferably on a faster machine (the
agent took 2 hours 12 minutes here).

### Reproduce it

```bash
scripts/benchmark/setup.sh --volume       # the benchmark's evaluator, agent and raw documents (~4 GB)

# Ours: needs the stored candidates and reranker scores (above) and PostgreSQL for the passages.
python -m rag_engine.evaluation.cli answer --run rag-haiku --per-type 12 \
    --rerank eval/cache/rerank-BAAI_bge-reranker-v2-m3.jsonl.gz
scripts/benchmark/evaluate.sh eval/runs/rag-haiku claude-haiku-5-5

# The rivals, on the same questions (every script meters its cost and stops at a budget):
scripts/benchmark/run_agent.sh eval/runs/grep-agent-haiku claude-haiku-5-5 6 6
STRIP_CITATIONS=1 scripts/benchmark/evaluate.sh eval/runs/grep-agent-haiku claude-haiku-5-5
scripts/benchmark/run_grep_oneshot.sh eval/runs/grep-oneshot-haiku eval/runs/rag-haiku
scripts/benchmark/evaluate.sh eval/runs/grep-oneshot-haiku claude-haiku-5-5

python -m rag_engine.evaluation.cli compare --arm "Hybrid + rerank (ours)=eval/runs/rag-haiku" \
    --arm "Iterative grep agent=eval/runs/grep-agent-haiku" \
    --arm "Single-attempt grep=eval/runs/grep-oneshot-haiku"
```

The runs behind the fairness audit (under $2 more, judging included):

```bash
python -m rag_engine.evaluation.cli answer --run rag-haiku-all --per-type 1000 \
    --rerank eval/cache/rerank-BAAI_bge-reranker-v2-m3.jsonl.gz    # every question with gold documents
scripts/benchmark/evaluate.sh eval/runs/rag-haiku-all claude-haiku-5-5 3
LABEL=repeat scripts/benchmark/evaluate.sh eval/runs/rag-haiku claude-haiku-5-5   # judge the same answers again
uv run python scripts/analysis/fairness_audit.py                                  # the audit tables
```

The benchmark's agent and evaluator call the Anthropic API on their own, with no cost report and no
cap. The scripts send their calls through a small metering proxy (`scripts/benchmark/meter_proxy.py`)
that holds the real key, adds up what every call costs and refuses new calls once the budget is spent.
The whole comparison (answers and judging, for the three systems) cost about $2.5, plus
roughly $0.20 of earlier trial runs.

### Limits to keep in mind

- **One small model.** Everything runs on Haiku 5.5, which is cheap but small. The benchmark's agent
  was designed around stronger models and would likely do better with one; this comparison says
  how the systems compare at equal model strength, not how good an agent can be.
- **One sample per question, 96 questions.** Newer models refuse a temperature setting, so a rerun
  does not give identical answers; a second run of our pipeline moved the combined score by four
  points (see the fairness audit). With 12 questions per type the per-type figures are only
  directional.
- **The judge is Claude Haiku 5.5**, a small model grading answers; it shares its blind spots with
  the system under test. Rescoring with a stronger judge (`evaluate.sh ... claude-sonnet-5-5`) costs
  a few dollars.
- **Our retrieval replays stored candidates**: the candidates collected from the index and the
  reranker scores computed on a GPU. The running service reproduces that replay (32 questions, see
  [retrieval.md](retrieval.md#checked-against-the-running-service)).
- **The single-attempt baseline's design matters.** Its ranking is by match count, the only ranking
  grep has. A smarter way to order files would score higher.
- **The cost of the first 28 single-attempt answers is estimated**, because a crash lost their usage
  report; the rest of the figures are metered.
