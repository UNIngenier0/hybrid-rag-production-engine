# Phase 8: iterative RAG, the fourth system

A complete work order for a coding assistant starting from a fresh clone on a new machine (the plan
assumes an ASUS laptop with 16 GB of RAM, Windows 11 and WSL2 Ubuntu). It says what to install, what
to build, in which order, how to measure it, when to stop and what to write down. Nothing here
depends on the conversation that produced it.

**Contents**

1. [How to start](#1-how-to-start)
2. [The goal and the evidence behind it](#2-the-goal-and-the-evidence-behind-it)
3. [Working agreements](#3-working-agreements)
4. [Stage 0: prepare the machine](#4-stage-0-prepare-the-machine)
5. [The experimental protocol](#5-the-experimental-protocol)
6. [Stage 1: deeper reranking and better evidence selection](#6-stage-1-deeper-reranking-and-better-evidence-selection)
7. [Stage 2: query reformulation](#7-stage-2-query-reformulation)
8. [Stage 3: gap-driven iteration](#8-stage-3-gap-driven-iteration)
9. [Stage 4: cascade and escalation](#9-stage-4-cascade-and-escalation)
10. [Stage 5: the live service](#10-stage-5-the-live-service)
11. [Stage 6: the final comparison](#11-stage-6-the-final-comparison)
12. [Stage 7: write-up, site and commit](#12-stage-7-write-up-site-and-commit)
13. [Known traps](#13-known-traps)
14. [Definition of done](#14-definition-of-done)
15. [Appendix: command reference and file map](#15-appendix-command-reference-and-file-map)

## 1. How to start

**For the person at the keyboard.** Open the Ubuntu (WSL2) terminal, clone the repository
(`git clone https://github.com/UNIngenier0/hybrid-rag-production-engine.git ~/hybrid-rag-production-engine`),
`cd` into it, start the coding assistant, and paste this as the first message:

> Read `docs/phase-8-iterative-rag.md` completely, then follow it from Stage 0. I speak Spanish;
> write every file and commit message in English. Ask me for what only I can provide (listed in
> section 4.1) before you start installing, tell me what you are about to run before any step that
> takes more than ten minutes or spends money, and stop at each gate in the document and report.

**For the assistant.** Read the whole file first, including sections 3 and 13. Then do Stage 0 and
report. Work stage by stage; each stage ends in a gate (a measured go/no-go). Do not start a stage
whose gate has not passed. Keep `docs/phase-8-log.md` (section 5.4) up to date as you go.

## 2. The goal and the evidence behind it

The repository compares three systems on EnterpriseRAG-Bench with the same model (Claude Haiku 5.5),
judge and questions: a single-pass hybrid RAG ("Hybrid + rerank (ours)"), the benchmark's iterative
grep agent and a single-attempt grep. Phase 8 adds a fourth, **"Iterative RAG (ours)"**: the same
index and models, but with deeper reranking, wider evidence selection, query reformulation and
follow-up searches driven by what the first answer says is missing, so that it can be compared
iterative against iterative with the grep agent.

**Hypothesis.** Most lost points come from evidence that never reaches the model, not from the
model. If more of it reaches the model, the score rises, at a cost far below the agent's.

**The evidence** (all computed from committed runs and the stored candidates; see
[results.md](results.md)). Baseline: 470 questions with gold documents, combined 48.9.

| Gold evidence in front of the model | Questions | Answered correctly |
|---|---|---|
| All gold documents shown | 295 (63%) | 80.7% (combined 74.6) |
| Some | 60 (13%) | 20.0% |
| None | 115 (24%) | 0.9% |

Of the 219 wrong answers: 114 (52%) had no gold document shown, 48 (22%) some, 57 (26%) all of them.
Recall of gold documents at each depth (first column is what the model sees today):

| Type | Top 5 chunks shown | Top 10 reranked | Pool of 30 reranked | Fused top 50 | All 200 candidates |
|---|---|---|---|---|---|
| all | 0.683 | 0.719 | 0.767 | 0.803 | 0.865 |
| semantic (125 q) | 0.424 | 0.440 | 0.480 | 0.544 | 0.672 |
| project_related (40 q) | 0.414 | 0.521 | 0.677 | 0.762 | 0.898 |
| completeness (20 q) | 0.370 | 0.508 | 0.619 | 0.656 | 0.708 |

Reading more distinct documents helps the multi-document types (12 distinct documents: 0.626 for
`project_related`, 0.524 for `completeness`) and barely `semantic` (0.456). The model sees 4.13
distinct documents on average (3.23 for `project_related`). `intra_document_reasoning` is right only
58% of the time with all its evidence shown, which is a reading limit, not a search limit.

**Success criteria** (all measured on the frozen test set, section 5):

1. The paired difference in combined score, Iterative RAG minus Hybrid + rerank, is at least +5
   points with a 95% interval that excludes zero.
2. The cost per answer stays under $0.01 on average and under $0.03 at the 95th percentile, against
   $0.0008 to $0.0016 for the single pass and $0.02 to $0.05 for the grep agent.
3. The comparison with the other three systems passes the fairness audit (section 11.2).

Failing the first one is an acceptable outcome if it is measured and written up: the gates in this
document exist so that a negative result is reported instead of tuned away.

**Non-goals.** No fine-tuning or training on benchmark data (it carries a canary and asks to stay out
of training corpora). No change to the index, chunking or embedding model (that is a re-embedding of
3 million chunks and a different project). No use of question types or gold documents at inference
time. No new rivals.

## 3. Working agreements

The assistant on the new machine has no memory of earlier sessions; these are the standing rules.

- **Language.** Talk to the user in Spanish. Write all files, documentation, comments and commit
  messages in English.
- **No attribution.** Never add `Co-Authored-By`, "Generated with" or any mention of an AI assistant as
  an author to commits, pull requests or files. Commits carry only the user's git identity
  (`git config user.name` / `user.email` must be set; ask if not).
- **Commit and push only when the user says so**, in that session. When asked: stage explicit paths
  (not blindly `git add -A`), scan the staged diff for secrets, never force-push. Push with
  `git push origin main` (or the credential helper the machine uses).
- **Secrets.** The Anthropic key lives only in `.env` as `ANTHROPIC_API_KEY`. Never print it, never
  `cat .env`, never put it in a command line, a log, a document or a commit. Containers receive it by
  name (`-e ANTHROPIC_API_KEY`). Use a key created for this machine, with a workspace spend limit.
- **Money.** Every model call goes through `--budget-usd` or the metering proxy. Each stage lists a
  cap (section 5.3). If a step would exceed it, stop and ask.
- **Compute.** The user dislikes sustained heavy CPU load. Cap containers (`--cpus`, `nice`), and ask
  before any job expected to run longer than an hour. Never leave long jobs attached to a tool call:
  start them with `nohup`/`setsid`, log to a file and poll with an `until` loop.
- **Honesty.** Report failures and negative results as they are. The gates are binding. Never tune on
  the test set. Say what was not measured.
- **Quality bar.** Match the surrounding code (comment density, naming, typing). `ruff check`,
  `ruff format --check`, strict `mypy` and `pytest` must pass; every new component gets unit tests with
  fakes (see `tests/fakes.py`), and nothing may break `python -m rag_engine.evaluation.cli gate`.
- **Do not edit a shell script while it runs.** Do not run development containers as root against the
  `hf_cache` volume.

## 4. Stage 0: prepare the machine

### 4.1 What only the user can provide

Ask for all of it at the start.

1. **A new Anthropic API key** for this machine (the key used earlier was pasted in a chat and should be
   rotated). The user writes it into `.env` themselves; the assistant must not see it.
2. **Kaggle access** (free GPU, for reranking): `kaggle auth login` or a token exported as
   `KAGGLE_API_TOKEN`; the Kaggle username (`uningenier0` was used so far).
3. **The files from the original machine** (4.2).
4. **The GPU model**: run `nvidia-smi` in WSL2. Many ASUS TUF A16 models have an AMD Radeon GPU, which
   WSL2 cannot use for PyTorch; an NVIDIA GPU can. Either way the plan works: reranking for the
   benchmark runs on Kaggle (section 6). A local NVIDIA GPU is an optional speed-up (appendix C).
5. **Approval of the budget** in section 5.3 (about $25 in total, cap per stage).

### 4.2 What to copy from the original machine

These are git-ignored, so a clone does not have them. On the original machine (the Mac):

```bash
cd ~/GitHub/hybrid-rag-production-engine
tar czf /tmp/eval-cache.tgz eval/cache                     # 61 MB: stored candidates and reranker scores
mkdir -p data/export
docker compose -f docker-compose.yml -f docker-compose.host-ports.yml up -d postgres
docker compose exec -T postgres sh -c 'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc' \
  > data/export/postgres.dump                              # about 1 GB: the cleaned documents
```

Copy `eval-cache.tgz` and `data/export/postgres.dump` to the new machine (USB drive, which WSL sees as
`/mnt/d` and so on, or `scp`). Then choose how the **vector index** reaches the new machine. Stage 1
needs only PostgreSQL (for the parent passages), so it can start while the index loads.

| Path | When | What to copy | Time on the new machine |
|---|---|---|---|
| **A. Qdrant snapshot** | The original machine has at least 35 GB free (check `df -h`) | `chunks_v1.snapshot`, 15.7 GB, made with the commands in [setup.md](setup.md#a-move-the-finished-index-from-another-machine-fastest) | tens of minutes. **Not yet tested at full size**; tested on a small collection |
| **B. Rebuild from the vectors** | Not enough disk for the snapshot | `data/vectors/`, 4.2 GB | stage 4 min, BM25 pass 20 to 45 min, loading the vectors 1 to 2 h, graph 10 to 17 min |

Do not fill the original machine's disk: if `df -h` shows under 35 GB free, use path B.

### 4.3 Install and verify

In the Ubuntu (WSL2) terminal. The repository must live in the Ubuntu filesystem (`~/`), never under
`/mnt/c`.

```bash
# Windows side, once, in PowerShell: wsl --install -d Ubuntu ; winget install Docker.DockerDesktop
#   Docker Desktop: Settings > Resources > WSL integration > enable Ubuntu.
#   %USERPROFILE%\.wslconfig  (16 GB machine):  [wsl2] memory=12GB  processors=8  swap=4GB
#   then in PowerShell: wsl --shutdown, and reopen Docker Desktop.

docker info --format '{{.MemTotal}} bytes, {{.NCPU}} CPUs'   # expect about 12 GB and 8 CPUs
curl -LsSf https://astral.sh/uv/install.sh | sh && exec $SHELL
git config --global core.autocrlf input                       # no CRLF in shell scripts
git clone https://github.com/UNIngenier0/hybrid-rag-production-engine.git ~/hybrid-rag-production-engine
cd ~/hybrid-rag-production-engine && cp .env.example .env
```

The user edits `.env`: set `POSTGRES_USER`, `POSTGRES_DB`, `POSTGRES_PASSWORD`, `ANTHROPIC_API_KEY`;
leave `API_KEYS` empty for local work. Then:

```bash
uv sync --locked
uv run ruff check . && uv run ruff format --check . && uv run mypy
uv run pytest -q                                              # expect about 560 passed, no services needed
uv run python -m rag_engine.evaluation.cli gate               # expect "No regression" and both checks matching
tar xzf /path/to/eval-cache.tgz                               # restores eval/cache
docker compose -f docker-compose.yml -f docker-compose.host-ports.yml up -d --build postgres qdrant redis
docker compose exec -T postgres sh -c 'pg_restore -U "$POSTGRES_USER" -d "$POSTGRES_DB" --no-owner --no-privileges' \
  < /path/to/postgres.dump
docker compose --profile ingest run --rm worker status        # PostgreSQL side: 511,962 documents staged
```

Then load the vector index by path A or B above (in the background, with `nohup`, logging to a file),
and the benchmark's own evaluator and documents, which the judge needs:

```bash
scripts/benchmark/setup.sh --volume       # about 4 GB download, several minutes to copy into a volume
uv run python -m rag_engine.ingestion.download   # corpus and the question file into data/raw (1.4 GB); always needed: the CLI reads data/raw/questions/test.parquet
```

### 4.4 Gate 0: the machine is ready

All must hold, and the assistant reports the outputs:

- `uv run pytest -q` passes; the gate reports no regression.
- `eval/cache/candidates.jsonl.gz` exists (470 questions) and `eval/cache/abstention/` holds the 30.
- `docker compose --profile ingest run --rm worker status` shows 511,962 documents; for Stage 2 on, also
  3,064,558 chunks and `points_without_dense` 0.
- **Speed measurement** (needed to plan Stages 2 and 3): with the full index, send 10 different
  `/v1/search` requests (`mode: hybrid`) and record the median and p95; the original machine measured
  10 s median. Put the numbers in the log.
- A smoke test of the whole path: `scripts/e2e.sh` (about 5 to 10 minutes, about 10 cents). It skips
  what the machine cannot run, and the assistant must say which steps were skipped and why.

## 5. The experimental protocol

This section is what keeps the result honest. Read it before writing any code.

### 5.1 Splits

Created once, in Stage 1, by a deterministic command and committed under `eval/splits/`:

| Split | Questions | Use |
|---|---|---|
| `test` | 126: the 96 questions of `eval/runs/rag-haiku/questions.txt` plus the 30 of `eval/runs/rag-haiku-unanswerable/questions.txt` | The only set on which the three existing rivals already have results. **Frozen: touched once per final configuration** |
| `dev` | 374: the other questions that have gold documents | Free retrieval experiments (recall needs no model) |
| `dev-small` | 150: a seeded stratified sample of `dev` (at least 8 per type, proportional otherwise) | Experiments that need answers and judging |

No decision (a threshold, a prompt, a number of passages, whether a stage stays) may use `test`.
Descriptive analysis in section 2 looked at all 470 questions; say so in the write-up.

### 5.2 Rules

- **Pre-register.** Before running each experiment, write in the log what is compared and the decision
  rule. After it, write the result and the decision.
- **Metrics.** Retrieval: gold-document recall of the evidence given to the model, with the average
  number of documents and an estimate of context tokens. Answers: the benchmark's `combined` score from
  the official evaluator with Claude Haiku 5.5 as judge, `--no-correction`, plus a 95% paired interval
  (`rag_engine.evaluation.metrics.paired_difference`). A difference below the run-to-run noise (about 4
  points on 96 questions) is noise.
- **Stop rules.** A component whose gate fails is removed from the design and the negative result is
  logged; do not stack more complexity on a failed premise.
- **Reproducibility.** Every run folder has `run.json` (with the new fields in 11.1), `trace.jsonl`,
  `answers.jsonl`, judge results and cost reports, like the existing ones. Commit them (they are small).

### 5.3 Budget (hard caps, in US dollars)

| Stage | Cap | What spends it |
|---|---|---|
| 1 | 3 | Answers and judging on `dev-small` for up to four selection policies |
| 2 | 3 | Query rewriting on `dev`, answers and judging on `dev-small` |
| 3 | 5 | Follow-up planning, extra answers, judging on `dev-small` |
| 4 | 4 | Stronger-model answers on the flagged share of `dev-small` |
| 6 | 10 | `test` (126) and the full 500 for the final configuration, two judges |
| Total | 25 | |

Pricing: Claude Haiku 5.5 is $0.10 per million input tokens and $0.50 per million output tokens (see
`src/rag_engine/generation/cost.py`). For any other model, look up its current price in Anthropic's
documentation before adding it to that table; never guess a price.

### 5.4 The log

Create `docs/phase-8-log.md` in Stage 1 and append an entry for every experiment: date, question
("does X raise Y?"), configuration, split, n, result with interval, cost, decision. It becomes the
source for the write-up.

## 6. Stage 1: deeper reranking and better evidence selection

**Goal.** Raise the recall of the evidence given to the model without any new search, using only the
candidates already stored. **Expected effort:** 4 to 6 hours of work, about 30 minutes of free Kaggle
GPU, under $3.

### 6.1 Splits and the log

Add `src/rag_engine/evaluation/splits.py` and a `make-splits` command in
`src/rag_engine/evaluation/cli.py` that writes `eval/splits/{test,dev,dev-small}.txt` (one question ID
per line) by the rules in 5.1; seed 0. Unit-test that the sets are disjoint, `test` has 126 IDs,
`dev` 374, `dev-small` 150 and that rerunning gives the same files. Create the log.

### 6.2 Score a deeper pool

Today the reranker scores the first 30 of a fused list built from 50 chunks per retriever. The stored
candidates hold 100 per retriever, so up to about 200 unique chunks exist per question.

1. Give `export-pool`, `rerank` and `collect`-adjacent code a `--fusion-depth` option (default 50, the
   present behaviour) that is passed to `FusionSetting(depth=...)`, and let `export-pool --pool` go to
   150. Fuse with equal weights (`dense_weight=1.0`, `sparse_weight=1.0`), as every reranker
   measurement did (`Settings.rerank_sparse_weight`). The compact pool file puts the already fused pool in
   the `dense` list, and `FusionSetting.depth` slices each list, so the `rerank` command that runs on
   Kaggle needs the same `--fusion-depth` or it silently scores only 50 chunks: pass the option through
   `scripts/kaggle/rerank.py` and its kernel template, and add a unit test that a pool of 150 chunks
   yields 150 scores (`pairs` in the score rows) per question.
2. `export-pool --fusion-depth 100 --pool 150` for the 470 (`eval/cache/candidates.jsonl.gz`) and, with
   `--types info_not_found high_level`, for the 30 (`eval/cache/abstention/candidates.jsonl.gz`).
3. Build the wheel and score it with the best model on the free GPU (about 75,000 pairs at 18.5 ms on a
   T4, about 25 minutes plus queue time):

   ```bash
   uv build --wheel
   uv run --group kaggle python scripts/kaggle/rerank.py run --username <kaggle-user> \
     --wheel dist/hybrid_rag_production_engine-0.1.0-py3-none-any.whl \
     --candidates eval/cache/deep/pool.jsonl.gz --models BAAI/bge-reranker-v2-m3 --pool 150 \
     --work data/kaggle/rerank-deep --out eval/cache/deep
   ```

   If the T4 queue is stuck for more than ten minutes, push the rendered kernel again with
   `kaggle kernels push -p data/kaggle/rerank-deep/kernel --accelerator NvidiaTeslaP100`.
   Score files are keyed by `point_id`; the output is `eval/cache/deep/rerank-BAAI_bge-reranker-v2-m3.jsonl.gz`.
   (The existing `eval/cache/rerank-*` files stay as they are.)

### 6.3 Selection policies

Add `src/rag_engine/retrieval/selection.py` with pure functions over a reranked list of `ChunkHit`
(sorted by score), each returning the chunks to show the model, and unit tests:

| Policy | Rule |
|---|---|
| `chunks(k)` | The first `k` chunks. `chunks(5)` is today's behaviour and must reproduce the baseline exactly |
| `docs(n)` | The best chunk of each of the first `n` distinct documents, in score order |
| `adaptive(n_min, n_max, alpha)` | Distinct documents in score order while the document's best score is at least `alpha` times the top score, between `n_min` and `n_max` documents. Uses the question's own scores only: no labels, no types |

Parent passages are attached afterwards with the existing `PassageStore` (one parent per document:
`build_context` already merges chunks that share a parent).

### 6.4 Replay

Extend `StoredRetriever` (`src/rag_engine/evaluation/stored.py`) with `rerank_pool` up to 150, the
`fusion` setting and a selection policy, and `AnswerGenerator` / the `answer` command with
`--policy chunks:5|docs:8|adaptive:4,12,0.5` and `--deep-rerank PATH`. The default must keep the old
behaviour so the gate and every existing run still reproduce.

### 6.5 Experiments and Gate 1

1. **Free, on `dev` (374):** document recall and mean documents shown for a grid: pool depth in
   {30, 50, 100, 150}, policy in {`chunks(5)`, `docs(5)`, `docs(8)`, `docs(12)`, `adaptive` with
   `n_min` in {4, 5}, `n_max` in {8, 12}, `alpha` in {0.3, 0.5, 0.7}}. Write
   `eval/reports/deep-retrieval.md` with the table by question type.
2. **With answers, on `dev-small` (150):** the baseline, the best fixed policy and the best adaptive
   policy by recall at the context budget below. Answer, judge, compare with paired intervals.
   Budget for context: at most twice the baseline's average input tokens.

**Gate 1.** Proceed to Stage 2 only if, on `dev-small`, the best policy raises the combined score by at
least 3 points with a paired 95% interval excluding zero, and the mean input tokens are at most twice
the baseline's. If the recall rises but the score does not, log it, keep the best policy off by default
and continue to Stage 2 only if the log shows the recall gain was on the multi-document types.

Fix the winning pool depth and policy in `eval/configs/phase8.json` (created now, extended by each
stage) and never change them again after Stage 3 starts.

## 7. Stage 2: query reformulation

**Goal.** Recover gold documents that the original question's words and embedding miss, mainly the
`semantic` type (one third of its gold documents is in none of the 200 candidates). **Expected effort:**
5 to 6 hours of work plus retrieval time, under $3.

### 7.1 The rewriter

`src/rag_engine/generation/rewrite.py`: `QueryRewriter(llm)` with `async rewrite(question) -> Rewrites`:
at most three extra queries, requested as strict JSON from Claude Haiku 5.5:

- a **keyword query**: the identifiers, product and feature names, numbers and rare terms of the
  question, with no function words (this is what BM25 needs);
- a **description query**: a one-sentence statement of what a passage that answers it would say (the
  idea of a hypothetical passage; this is what the dense model needs);
- **sub-questions**, only if the question asks for more than one thing (at most two).

Parse defensively: on any parse error, an empty list or a call failure, return no extra queries and
log it, so the system degrades to the original question. Do not send `temperature` (Haiku 5.5 rejects
it) and give the call `max_tokens` of at least 1500 (reasoning tokens count against it). The prompt
must forbid inventing facts and forbid adding terms that are not implied by the question. Unit-test
the parser with malformed outputs.

### 7.2 Multi-query candidates

`src/rag_engine/retrieval/multi.py`: given the candidate lists of the original question and of each
rewrite (each: dense 100, BM25 100), combine with reciprocal rank fusion across all lists (weights 1.0
for the original question's lists and 0.7 for the others; both values to be checked on `dev`), take the
union, and rerank it **against the original question** (the reranker scores pairs of the user's
question and a passage; a rewrite is only a way to find candidates).

### 7.3 Staged execution (the index is slow; keep it replayable)

Searches for rewritten queries are not in the stored candidates, so they are collected once and then
replayed, like everything else:

1. `cli rewrite --split dev --out eval/cache/iter/rewrites.jsonl`: one LLM call per question (about
   $0.0004 each), idempotent and resumable.
2. `cli collect --queries-file eval/cache/iter/rewrites.jsonl --out eval/cache/iter/candidates.jsonl.gz
   --depth 100 --concurrency 2`: extend `collect` to accept `(question_id, query_id, text)` rows and store
   candidates under `question_id#query_id`. Set `QDRANT_TIMEOUT_SECONDS=300` for batch collection.
   Plan with the speed measured at Gate 0: 374 questions with three rewrites is about 1,100 searches
   (3 to 5 hours at 10 s each, less on a machine that caches the index better). Run in the background.
3. `export-pool` a per-question union (original and rewrites) and score it on Kaggle as in 6.2.
4. Replay through `MultiQueryRetriever` offline.

### 7.4 Experiments and Gate 2

On `dev` (free, recall): original only, original plus keyword, plus description, plus sub-questions, all
three; with and without the 0.7 weight. On `dev-small` with answers: the best combination against the
Gate 1 configuration.

**Gate 2.** Keep the reformulation only if, on `dev`, it raises gold-document recall of the evidence by
at least 3 points overall (or 5 points on the questions of the `semantic` type) without lowering any
other type by more than 1 point, and if on `dev-small` the combined score rises or stays within
the noise at under $0.001 of added cost per question. Otherwise remove it and go to Stage 3 with the
Gate 1 configuration.

## 8. Stage 3: gap-driven iteration

**Goal.** Let the system notice that something is missing and look for exactly that. This is the
piece that should help `project_related`, `completeness` and `high_level`. **Expected effort:** 6 to 8
hours, under $5.

### 8.1 Components

1. **The signal.** `rag_engine.generation.coverage.coverage(raw_answer)` already returns a status
   (`answered`, `partial`, `not_found`) and the `gaps`: the sentences in which the answer says the
   documents lack something. Answers with gaps were right 15% of the time, the others 84%.
2. **`src/rag_engine/generation/followup.py`: `GapPlanner(llm)`.** Input: the question, the current
   answer, its gaps, and the titles of the documents already used. Output: at most two follow-up
   queries per round, each aimed at one gap, as strict JSON, deduplicated against all earlier queries
   (case- and whitespace-insensitive). Same rules as the rewriter for errors, `temperature` and
   `max_tokens`.
3. **`src/rag_engine/generation/iterative.py`: `IterativeAnswerer`**, an orchestrator written only
   against the existing ports (`Retriever`, `LLMClient`) so it can be tested with fakes:

   ```
   evidence = select(retrieve(question [+ Stage 2 rewrites]))
   answer   = generate(question, evidence)
   for round in 1..max_rounds:                         # default 2
       if answer.status == "answered": break           # nothing is said to be missing
       queries = planner.plan(question, answer, gaps)  # at most 2
       if not queries or searches_used + len(queries) > max_searches: break   # default 6
       new = select(retrieve(q) for q in queries)      # same reranking and selection as before
       evidence = merge(evidence, new, max_documents)  # dedupe by document, keep best scores
       if no new document was added: break
       answer = generate(question, evidence)           # one more answer step, same prompt
   ```

   Hard limits on rounds, searches, total tokens and wall-clock (defaults in settings, section 10).
   It returns the final `GeneratedAnswer` plus a `trace`: for each step the query, how many documents
   were new, the gaps found, tokens and cost. The answer prompt stays the same (`SYSTEM_PROMPT`).
4. **Document accounting.** `document_ids` for the benchmark are the documents cited in the final
   answer, as for the other systems.

### 8.2 Staged execution

The queries of round r depend on the answers of round r-1, so evaluation runs in rounds, each one
resumable and idempotent:

`cli iterate --run NAME --split dev-small --config eval/configs/phase8.json`

1. Run the answerer in *recording mode*: a `StagedRetriever` serves stored candidates keyed by the exact
   query text, and when a query has none it raises `MissingRetrieval(query)`. The command collects
   every missing query into `eval/cache/iter/pending-<round>.jsonl`.
2. `cli collect --queries-file ...` (live index), `export-pool`, Kaggle rerank, as in 7.3.
3. Rerun `iterate`. Repeat until it reports `0 missing`. Only about 44% of questions get past the first
   answer, so each round is smaller than the last.

Every LLM call made along the way is cached by a hash of its inputs in `eval/cache/iter/llm/` so a rerun
never pays twice, and its cost is counted once in the run's usage report.

### 8.3 Tests

Unit tests with fakes: stops when the status is `answered`; stops at the round limit, the search limit
and the token limit; stops when a round adds no new document; deduplicates queries; merges evidence
without duplicates; produces a trace; degrades to the first answer if the planner or a search fails;
never uses more than the configured limits.

### 8.4 Experiments and Gate 3

On `dev-small`: the Gate 1/2 configuration against the same with iteration, with `max_rounds` in {1, 2}.
Report the combined score overall and by type with paired intervals, and the cost, searches and latency
per question (mean and p95).

**Gate 3.** Keep iteration only if the combined score rises by at least 3 points with an interval
excluding zero **and** the mean cost is under $0.01 per question with at most 6 searches at the 95th
percentile. If it helps only one question type, keep it only if that type's gain is at least 10 points
and the loss elsewhere is below the noise.

## 9. Stage 4: cascade and escalation

**Goal.** Spend more only where the system's own signal says it is probably wrong. **Expected effort:**
about 2 hours, under $4.

- **Routing is already there**: only answers whose status is not `answered` continue (Stage 3). Check
  that it is not the verifier's `unsupported` count or anything that needs labels.
- **Escalation.** If, after the last round, the status is still `partial`, optionally write the final
  answer with a stronger model (`--escalation-model`, for example `claude-sonnet-5-5`, if the workspace
  has it; confirm the model exists with a one-token call before a run). Add the price to `cost.py` after
  looking it up. Same prompt, same evidence.
- **Experiment, on `dev-small`:** with and without escalation; report the gain, the share of questions
  escalated and the added cost. `intra_document_reasoning` (right only 58% with its evidence shown) is
  the expected beneficiary.

**Gate 4.** Keep escalation only if it adds at least 2 points overall (interval excluding zero) at under
$0.005 of added mean cost per question; otherwise drop it.

At the end of this stage write the final configuration into `eval/configs/phase8.json` and **freeze it**:
record its SHA-256 in the log. The only edits allowed afterwards are bug fixes that do not change
behaviour on the questions used so far.

## 10. Stage 5: the live service

**Goal.** The same pipeline available through the API, so that the benchmark numbers describe the
service and not only a replay. **Expected effort:** about 3 hours.

- Settings (`src/rag_engine/core/config.py`, `docker-compose.yml`, `.env.example`): `QUERY_STRATEGY`
  (`single` default, or `iterative`), `ITERATIVE_MAX_ROUNDS`, `ITERATIVE_MAX_SEARCHES`,
  `ITERATIVE_MAX_TOKENS`, `ITERATIVE_TIMEOUT_SECONDS`, `ESCALATION_MODEL`, and the frozen selection and
  rewrite settings. Every default equals the frozen configuration.
- `POST /v1/query` gets an optional `strategy` field; the response gains `steps` (the trace summary:
  queries, new documents, gaps, per-step tokens). The cache key must include the strategy and the
  configuration. The demo page shows the steps.
- Metrics: `rag_query_steps` (histogram of rounds), `rag_query_searches_total`, and the existing cost
  and token counters must include the extra calls. The per-client rate limit for `/v1/query` stays per
  request; document that one iterative request costs several model calls.
- Tests: unit (fakes), API tests like `tests/unit/test_query.py`, an integration test marked
  `integration`, and a **consistency check** like the earlier one: run 16 questions of `dev-small` through
  the live endpoint and through the replay and compare the evidence and the answers' document sets
  (differences must be explainable by ties and by the reranker the live service uses).
- Be explicit in the docs that the live service on a CPU machine uses the small MiniLM reranker unless
  `RERANKER_MODEL` names `BAAI/bge-reranker-v2-m3` and a GPU is available; the benchmark numbers use the
  larger one.

## 11. Stage 6: the final comparison

**Expected effort:** about 4 hours, mostly waiting for compute; under $10.

### 11.1 Runs

With the frozen configuration, from a clean state:

1. **Test (126 questions)**, once: run the staged `iterate` to completion on `test` (it needs candidates
   for the new queries on the 126; collect and rerank them), answer, judge with the official evaluator
   (Haiku, `--no-correction`). The run folder is `eval/runs/rag-iterative-test`. Its `run.json` records
   `"system": "rag-iterative"`, the frozen configuration hash, and `steps`, `searches`, `llm_calls`,
   `escalated` totals.
2. **All 500 questions**, for the headline: the same staged procedure on the rest (the 374 `dev`
   questions), folder `eval/runs/rag-iterative-all`. The report must say that the `dev` part was used to
   tune the configuration and the `test` part was not.
3. **Second judge**: score the answers of the single pass and of the iterative system again with a
   different judge model (Sonnet class) through the same evaluator (`LABEL=sonnet scripts/benchmark/evaluate.sh`
   with the judge model argument), on `test` and on the 500. Report both judges side by side; if they
   disagree on the ranking of the systems, say so.
4. If time allows and the user agrees, repeat the iterative run on `test` a second time to estimate its
   run-to-run noise (cost about $1).

### 11.2 The four-way comparison and the audit

- `compare` with four arms on the 126 `test` questions, in this order: `Hybrid + rerank (ours)`,
  `Iterative RAG (ours)`, `Iterative grep agent`, `Single-attempt grep`. The first is the baseline of the
  paired intervals. Output: `eval/reports/answer-comparison-iterative.{md,json}`.
- Extend `src/rag_engine/evaluation/audit.py`: the check "same evidence budget for single-pass systems"
  must treat `rag-iterative` like the grep agent (iterative by design, listed under "differences that
  remain by design"), not fail it; add a check that every iterative arm reports its cost, calls and
  searches; keep all existing checks. Add tests for both.
- `src/rag_engine/evaluation/gate.py`: register the new comparison, the new run folders and the new
  report so the regression gate recomputes them from the committed runs in every build, like the earlier
  ones (`check_answer_report` takes the arms and the report path). Reports not yet committed are skipped,
  so commit the runs and the reports together.
- `full-benchmark` for the iterative system:
  `--runs eval/runs/rag-iterative-all --out-md eval/reports/full-benchmark-iterative.md
  --out-json eval/reports/full-benchmark-iterative.json --submission-dir eval/submission-iterative`
  (extend the command to take a prefix for the paths and `DEFAULT_RUNS` accordingly), and a gate check
  for it.

### 11.3 Reporting rules

State, in `docs/results.md`: the paired difference with its interval on `test`; the cost, searches and
latency per question; the ablation table (each stage's contribution on `dev-small`, from the log); both
judges; what was tuned on what; and the categories where the system is still weak. If a success criterion
of section 2 was not met, say which and by how much.

## 12. Stage 7: write-up, site and commit

1. **Docs.** `docs/results.md` (new section and updated headline), `docs/evaluation.md` (four-way
   comparison, its audit and how to reproduce it), `docs/lessons-learned.md` (what the stages taught,
   including failed gates), `docs/reproduce.md` (new rows in the table), `docs/operations.md` (the new
   settings and metrics), `docs/leaderboard.md` (which submission to send), README (status table, quality
   targets, roadmap row 8). Keep every number traceable to a committed report; the gate enforces the ones
   it can.
2. **Site.** Re-record the examples against the live iterative service
   (`scripts/record_examples.py --ids ... --reranker <name>`; add the strategy and a steps display to
   `RENDER_JS` if useful), re-record the walkthrough's trace if the retrieval changed
   (`scripts/record_trace.py`; the walkthrough text in `site/template.html` states facts of that trace
   and a test checks them), rebuild with `scripts/build_site.py`, and update `NOTES` in
   `src/rag_engine/site_page.py` if the examples change. The Pages workflow redeploys on push.
3. **Checks.** `ruff`, `mypy`, `pytest`, the gate, `scripts/e2e.sh --rivals`, the link check used in
   earlier phases (every relative link and anchor in the README and `docs/` must resolve).
4. **Commit** only when the user says so: small, titled commits per stage or one per phase as the user
   prefers, no attribution, secrets scan before each. Then watch the CI and Pages runs to success.

## 13. Known traps

- **Claude Haiku 5.5:** never send `temperature` (HTTP 400). Reasoning tokens count against
  `max_tokens`; structured calls need at least 1500, answers 4000 (`LLM_MAX_TOKENS`).
- **Docker Compose passes unset variables as empty strings.** The settings validators already treat an
  empty `ANTHROPIC_API_KEY`, `VERIFICATION_MODEL`, `ABSTAIN_THRESHOLD`, `API_KEYS` and `METRICS_TOKEN`
  as unset; do the same for any new optional setting.
- **Metering proxies share names.** `evaluate.sh`, `run_agent.sh` and `run_grep_oneshot.sh` each start a
  proxy container; two scripts using the same `METER` name remove each other's proxy. Use
  `METER=rag-meter-judge` (or any distinct name) when running a judge beside another job.
- **Resumed runs split their cost report** in two. The audit reads `cost-*.json`; when a run is resumed,
  keep each session's report as `cost-*.partN.json` and write the sum into `cost-*.json`.
- **Background jobs started from a tool call can be killed when the call returns.** Use
  `nohup ... &`/`setsid`, redirect output to a file and poll the file with an `until` loop (a plain
  `sleep` followed by another command is rejected by some harnesses).
- **Kaggle.** The T4 queue sometimes stalls for an hour: push the kernel again with another accelerator
  (see 6.2). The wheel in `dist/` must be rebuilt after changing code that the kernel runs. Free quota is
  30 GPU-hours a week; the earlier work used under 3.
- **Qdrant.** The gRPC deadline is `QDRANT_TIMEOUT_SECONDS` (30 by default); a cold search can outlast it
  and the API now answers 503 instead of crashing. Set 300 for batch collection. The Qdrant version is
  pinned in `docker-compose.yml`; a snapshot only restores into the same version.
- **Memory.** Searches over cold parts of the index take seconds on a small page cache. Give Docker
  11 to 12 GB on a 16 GB machine and measure (Gate 0).
- **WSL2.** Keep the repository under `~/`. `host.docker.internal` is not defined inside WSL2 containers
  the way it is on macOS; the scripts that need the API from a container (`scripts/loadtest.py`) take
  `--url`, so give it the host's address or run the client in the same Compose network.
- **zsh vs bash.** An unmatched glob aborts a command chain in zsh; WSL's default shell is bash.
- **Leakage.** The 96 + 30 `test` questions already carry results for three systems. Using their scores
  to choose anything for the new system turns the comparison into a tuned one.
- **Do not commit** `.env`, `data/`, `dist/`, `_site/`, or `eval/cache/` (git-ignored).

## 14. Definition of done

- [ ] Gate 0 passed and its outputs are in the log.
- [ ] Splits committed; Gates 1 to 4 each have a logged decision, including the components that were
      dropped.
- [ ] `eval/configs/phase8.json` frozen and its hash in the log; the final run used exactly it.
- [ ] `test` evaluated once with the frozen configuration; the full 500 evaluated; two judges.
- [ ] Four-way comparison and its audit published and checked by the regression gate in CI.
- [ ] The service exposes the iterative strategy, behind a setting, with tests and metrics; the
      consistency check is logged.
- [ ] `docs/results.md`, `docs/evaluation.md`, `docs/lessons-learned.md`, `docs/reproduce.md`,
      `docs/operations.md`, the README and the site are updated; every link resolves.
- [ ] `ruff`, `mypy`, `pytest`, the gate and `scripts/e2e.sh --rivals` pass; CI and Pages are green.
- [ ] The log states, for each success criterion in section 2, whether it was met.

## 15. Appendix: command reference and file map

### A. Existing commands you will use

All of `python -m rag_engine.evaluation.cli ...` (prefix with `uv run` on the host, or run inside the
worker image with `docker compose --profile ingest run --rm --no-deps ... --entrypoint python worker -m
rag_engine.evaluation.cli ...`, mounting `./eval` and `./data` and passing `-e ANTHROPIC_API_KEY`):

| Command | Purpose |
|---|---|
| `collect [--types ...] [--depth 100] [--concurrency 2]` | Search the live index and store candidates |
| `export-pool [--types ...] [--pool N]` | Write the compact pool file to score on a GPU |
| `rerank --model M --out F` | Score pools with a cross-encoder (run on Kaggle by `scripts/kaggle/rerank.py`) |
| `answer --run NAME [--types ...] [--per-type N] --candidates F --rerank F [--budget-usd X]` | Write answers from stored candidates |
| `compare --arm "Name=eval/runs/X" ...` | Compare systems with the fairness audit |
| `abstention-scores`, `full-benchmark`, `build-gate-fixture`, `gate` | Reports and the regression gate |
| `scripts/benchmark/evaluate.sh RUN JUDGE BUDGET` | The benchmark's official evaluator (`LABEL=`, `STRIP_CITATIONS=1`, `METER=` options) |
| `scripts/e2e.sh [--rivals]` | End-to-end check |

### B. Files this phase adds or changes

| File | Change |
|---|---|
| `src/rag_engine/evaluation/splits.py`, `eval/splits/*.txt` | New: the three splits and their command |
| `src/rag_engine/retrieval/selection.py` | New: `chunks`, `docs`, `adaptive` policies |
| `src/rag_engine/retrieval/multi.py` | New: multi-query fusion and reranking against the original question |
| `src/rag_engine/generation/rewrite.py`, `followup.py`, `iterative.py` | New: rewriter, gap planner, orchestrator |
| `src/rag_engine/evaluation/stored.py`, `cli.py`, `answering.py` | Deeper pools, selection policy, `rewrite`, `iterate`, `--queries-file`, `--fusion-depth` |
| `src/rag_engine/generation/cost.py` | Prices of any new model (looked up, not guessed) |
| `src/rag_engine/generation/query.py`, `api/*`, `core/config.py` | Strategy, steps, settings, metrics |
| `src/rag_engine/evaluation/audit.py`, `gate.py`, `fullbench.py` | Four-way audit and gate checks |
| `eval/configs/phase8.json`, `eval/reports/*`, `eval/runs/rag-iterative-*` | The frozen configuration and the results |
| `docs/phase-8-log.md` and the docs listed in Stage 7 | The record |

### C. Optional: a local NVIDIA GPU

If `nvidia-smi` works in WSL2, the larger reranker can run locally and the offline Kaggle round trips can
be skipped for small jobs. The repository does not set this up: the API image installs the CPU build of
PyTorch, and Docker needs the NVIDIA container toolkit and a CUDA PyTorch. Treat it as a separate,
untested task, and keep Kaggle as the path for every number in the report so the results stay
reproducible by anyone. An AMD GPU cannot be used from WSL2 for this.
