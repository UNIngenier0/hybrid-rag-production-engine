# Reproducing the results

How to get from a clone of the repository to each number in [results.md](results.md), what each
step needs and costs, and what will not come out identical. The machine setup (Docker memory,
macOS, Windows, Linux) is in [setup.md](setup.md); this page is the map of what to run after it.

## What you need

| | |
|---|---|
| Machine | Docker with 6 GB or more for the containers and about 35 GB of free disk for the full index; 16 GB of RAM is comfortable |
| Tools | Docker Compose, [uv](https://docs.astral.sh/uv/) |
| A free Kaggle account | A GPU for the offline embedding and reranking jobs (2.16 of the 30 free weekly GPU-hours for the embeddings) |
| An Anthropic API key | Only for answers and judging. Scope it to a workspace with a spend limit |
| Time | The index from nothing: about 3.5 hours of wall clock, mostly waiting for the GPU job and the load. Everything after it: minutes |

## Three ways in, from cheapest

**1. Check the numbers without any service (about a minute).** The reports, the run folders and a
compact copy of what the retriever returned for every question are committed. The regression gate
recomputes the retrieval numbers, the answer comparisons, the abstention figures and the
whole-benchmark score from them and fails if any published number differs:

```bash
uv sync
uv run pytest                                      # about 550 unit tests
uv run python -m rag_engine.evaluation.cli gate    # what CI runs
```

**2. Run the stack on a machine that already has the index.** Move the PostgreSQL dump and the
Qdrant snapshot across ([setup.md](setup.md#a-move-the-finished-index-from-another-machine-fastest))
and the service works at once; `scripts/e2e.sh` then checks it end to end. Nothing about the
index is committed (it is 35 GB).

**3. Build everything.** The steps below. Each one can be interrupted and repeated.

## Building the index from nothing

```bash
cp .env.example .env                                  # set POSTGRES_PASSWORD, and a Kaggle token
docker compose -f docker-compose.yml -f docker-compose.host-ports.yml up -d qdrant postgres
python -m rag_engine.ingestion.download               # 1.4 GB, SHA-256 verified; under 1 minute
python -m rag_engine.workers.bulk_ingest profile      # sizes, no services needed; 5 minutes
python -m rag_engine.workers.bulk_ingest stage        # clean into PostgreSQL; 4 minutes
python -m rag_engine.workers.bulk_ingest index sparse # chunk + BM25 into Qdrant; about 42 minutes
```

The dense vectors come from a GPU job, because the same work projects to 80 to 125 hours on a
laptop CPU ([kaggle-embedding.md](kaggle-embedding.md) has the commands; the free GPU did it in
2.16 hours across four runs). Then:

```bash
python -m rag_engine.workers.bulk_ingest load-vectors # attach them; about 2 hours
python -m rag_engine.workers.bulk_ingest status       # points_without_dense must be 0
docker compose up --build --wait                      # the API on :8000
```

The measured timeline, with the problems met on the way, is in the
[ingestion report](ingestion-report.md). Expect 511,962 documents and 3,064,558 chunks.

## Re-running the measurements

All of these go through `python -m rag_engine.evaluation.cli` and write under `eval/`; the
sequence and the options are in [evaluation.md](evaluation.md#how-it-runs) and
[its reproduction section](evaluation.md#reproduce-it).

| To get | Run | Needs | Cost |
|---|---|---|---|
| Candidates for all questions | `collect` (and `collect --types info_not_found high_level`) | the index | free; 27 minutes for the 470 on the development laptop |
| Reranker scores | `export-pool`, then `scripts/kaggle/rerank.py` | the candidates, Kaggle | free; about 20 minutes for five models |
| The retrieval ablation | `report` | candidates and scores | free |
| Answers for all 500 | `answer --per-type 1000 …` and `answer --types info_not_found high_level …` | stored candidates, a key | $0.42 |
| Their scores | `scripts/benchmark/evaluate.sh <run>` | the benchmark checkout (`scripts/benchmark/setup.sh --volume`), a key | $0.44 |
| The whole-benchmark score and the submission file | `full-benchmark` | the two judged runs | free |
| The two rivals and the comparison | `scripts/benchmark/run_agent.sh`, `run_grep_oneshot.sh`, `compare` | the raw documents (about 4 GB), a key | about $2 for the agent on 96 questions, $1.4 on the 30 |
| The citation check | `verify-check` | the stored candidates, a key | $0.30 |
| The abstention report | `build-gate-fixture --extra …`, then `abstention-scores` | candidates and scores of the 30 | free |
| The load test | `scripts/loadtest.py` | the running stack | $0.03 for the answers |
| The fairness audit tables | `scripts/analysis/fairness_audit.py` | stored candidates and scores | free |

The benchmark's agent and evaluator call the Anthropic API on their own, with no cost report and no
cap. The scripts in `scripts/benchmark` send those calls through a small metering proxy that holds
the real key, adds up what each call costs and refuses new calls once the budget is spent; run them
with a budget argument, as the examples in [evaluation.md](evaluation.md#reproduce-it) do.

## Checking a run

```bash
scripts/e2e.sh --rivals     # lint, types, tests, the gate, the stack, search, answers, rivals
```

It reports every step as passed, failed or skipped with the reason, and takes about 10 minutes
with the rivals. A passing run on the final code: 16 passed, 0 failed.

## What will not come out identical

- **Answers and scores move between runs.** The language model is not deterministic. A second run
  of the same pipeline on the same 96 questions scored 46.3 against 50.6, and the judge, asked
  twice about the same answers, changed its verdict on up to 2 of 96. A difference of a few points
  between two runs is noise; the intervals in the reports say how much.
- **Model versions.** The runs used Claude Haiku 5.5 as answering model and judge. A later
  version of the model behind the same name can score differently, which is why every manifest in
  `eval/runs/*/run.json` records the model.
- **Approximate search and ties.** A live search compared with the stored candidates on 32
  questions matched exactly for dense (32) and fused (32) rankings, on 31 for BM25 (one exact score
  tie) and on 29 for the reranked order (half-precision scores that tie reorder). The gate replays
  the stored candidates for that reason, and the live comparison is in
  [retrieval.md](retrieval.md#checked-against-the-running-service).
- **Latency** depends entirely on the machine; the figures are from one laptop.
