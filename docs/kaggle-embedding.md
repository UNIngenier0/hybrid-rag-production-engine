# Embedding the corpus on a free GPU (Kaggle)

The benchmark corpus is about 670 million tokens once chunked. Embedding it is the one step of
ingestion that needs a GPU, so it runs as an offline job on Kaggle's free GPUs and hands the
vectors back as Parquet files. Everything else (cleaning, chunking, BM25, PostgreSQL, Qdrant)
stays local.

## Why not locally

Measured on the development machine (8 GB RAM, 8 cores) with real corpus text:

| Where | Throughput (small 384-dim model) | Whole corpus |
|---|---|---|
| CPU cores, in Docker | 1.5-2.5 k tokens/s | 80-125 h |
| Apple GPU, half precision | 11-23 k tokens/s | 8-16 h, and the machine is unusable meanwhile |

A free Kaggle GPU is the cheapest way to take that off the laptop. Free embedding APIs do not
cover the corpus: Gemini's free tier is about 30 k tokens/min, Voyage gives 200 M free tokens
(under a third of the corpus) and Jina's free key is 100 k tokens/min.

## The model

`Snowflake/snowflake-arctic-embed-m-v1.5`, pinned to an exact revision: 768 dimensions,
109 M parameters, Apache-2.0, standard BERT architecture (no remote code). Mean nDCG@10 over the
15 English MTEB retrieval datasets, computed from the results in each model card:

| Model | nDCG@10 |
|---|---|
| snowflake-arctic-embed-xs (384 dims) | 50.15 |
| **snowflake-arctic-embed-m-v1.5 (768 dims)** | **55.14** |
| snowflake-arctic-embed-m-v2.0 (768 dims) | 55.38 |

v2.0 is no better on English text and needs `trust_remote_code`, so v1.5 is used. Changing the
model changes `EMBEDDING_MODEL`, `EMBEDDING_REVISION` and `EMBEDDING_DIMENSION`, and requires a new
Qdrant collection.

## How it fits together

```text
Kaggle kernel (GPU)                               your machine
-------------------                               ------------
download corpus (SHA-256 verified)
chunk with the same code as local     vectors-*.parquet
embed on the GPU(s), half precision   ----------------->   load-vectors  ->  Qdrant
write one shard per 5,000 documents                        (checks, then attaches)
```

The kernel script embeds the repository's code as a wheel, so it chunks exactly like the local
pipeline. Each shard records its model, revision, dimension and a fingerprint of the cleaning and
chunking code. `load-vectors` refuses a shard whose fingerprint differs from the local one, whose
documents the BM25 pass has not covered, or whose chunk counts differ from the local chunker's.

## Run it

One-time setup:

1. Create a Kaggle account and verify your phone number (Kaggle normally asks for it before it
   enables GPU and internet in notebooks).
2. Authenticate the Kaggle CLI: `kaggle auth login` (browser sign-in), or create an API token in
   Kaggle's Settings and export it as `KAGGLE_API_TOKEN`.
3. `uv sync --group kaggle`

Then, from the repository root:

```bash
uv build --wheel                       # packs the code that will run on Kaggle

# 1. Smoke run: 2 shards (10,000 documents). Measures the real throughput.
uv run --group kaggle python scripts/kaggle/orchestrate.py run \
    --username <kaggle-user> --wheel dist/*.whl --shards 0-1 --label smoke

# 2. The rest, in blocks (there are 103 shards). A second kernel name lets two blocks run at once.
uv run --group kaggle python scripts/kaggle/orchestrate.py run \
    --username <kaggle-user> --wheel dist/*.whl --shards 2-35 --label part-1
uv run --group kaggle python scripts/kaggle/orchestrate.py run \
    --username <kaggle-user> --wheel dist/*.whl --shards 36-69 --label part-2 --slug hybrid-rag-embeddings-b
uv run --group kaggle python scripts/kaggle/orchestrate.py run \
    --username <kaggle-user> --wheel dist/*.whl --shards 70-102 --label part-3
```

What the full run took (see the [ingestion report](ingestion-report.md)): the smoke run embedded
131,581 chunks in 342 s on 2x Tesla T4, and the three blocks took 55, 37 and 31 minutes. Blocks hold
very different numbers of chunks (the first has 48% of the corpus), so size them by chunk count,
not shard count. In total the job used 2.16 GPU-hours of the free quota.

Each run pushes the kernel, waits for it, and moves the shards into `data/vectors/`. Shards that
already exist are skipped, so a run can be repeated.

Locally, after the BM25 pass (`bulk_ingest index sparse`):

```bash
docker compose -f docker-compose.yml -f docker-compose.host-ports.yml up -d qdrant postgres
python -m rag_engine.workers.bulk_ingest load-vectors      # attaches the vectors
python -m rag_engine.workers.bulk_ingest status            # points_without_dense should be 0
```

## Limits to know

- Kaggle's free quota is roughly 30 GPU hours per week and a session lasts up to 12 hours; check
  yours with `kaggle quota`. The T4 is requested explicitly because recent PyTorch builds do not
  support Kaggle's P100.
- The kernel is private and has internet access only to download the corpus and the model.
- The vectors are stored in half precision: 4.2 GB of shards for the whole corpus.
