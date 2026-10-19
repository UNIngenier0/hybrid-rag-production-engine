# Ingestion report: the full reference corpus

Phase 2 indexed the whole EnterpriseRAG-Bench corpus, with no subset. This report records what was
built, how long each step took on the development machine (a laptop with 8 GB of RAM and 8 cores,
Docker limited to 6 GB), what went wrong and what was learned. Every number below was measured in
the run itself; commands to reproduce it are in the README and in
[kaggle-embedding.md](kaggle-embedding.md).

## Result

| | |
|---|---|
| Documents | 511,962 (2.45 billion characters) |
| Chunks in the index | **3,064,558**, each with a dense vector, a BM25 sparse vector and its payload |
| Tokens embedded | about 685 million (estimate from a 30,645-chunk sample; 0.003% of chunks exceed the model's 512-token limit) |
| Dense model | `snowflake-arctic-embed-m-v1.5`, 768 dimensions, fixed to an exact revision |
| Vectors without an embedding | 0 |
| Disk | vector shards 4.2 GB, Qdrant volume 15.7 GB, PostgreSQL 2.5 GB |

## Timeline

| Step | Where | Duration | Throughput |
|---|---|---|---|
| Download and verify the corpus (SHA-256) | local | under 1 minute | about 29 MB/s |
| Profile: clean and chunk everything | 1 CPU core | 4.8 minutes | 1,796 documents/s |
| Stage the cleaned corpus in PostgreSQL | local | 4.0 minutes | about 2,000 documents/s |
| BM25 pass: chunk, encode, index | local, Qdrant capped at 2 cores | about 42 minutes | about 1,200 chunks/s |
| Embeddings on Kaggle, 4 runs | 2x Tesla T4 per run | 2.16 GPU-hours of the 30 free per week | about 380 to 430 chunks/s per run |
| Load the vectors into Qdrant | local | about 2.1 hours | about 350 to 470 vectors/s |
| Build the dense HNSW graph | local, 2 threads | 17 minutes | |

The Kaggle runs were: a 10,000-document trial (342 s of embedding) to measure real throughput, then
three blocks of the remaining shards (55, 31 and 37 minutes). Blocks 2 and 3 ran at the same time.
From the first push to the finished graph took about 3.5 hours of wall-clock time, and the laptop
only ever ran the cleaning, the BM25 pass and the loading, all with limited CPU.

## Why the embeddings are computed elsewhere

On this machine the same job was projected at 80 to 125 hours on the CPU cores (measured with small
models, a third of this one's size) and 8 to 16 hours on the Apple GPU while making the machine
unusable. A free Kaggle GPU
did it in 2.16 hours. The vectors come back as Parquet shards that describe themselves (model,
revision, a fingerprint of the cleaning and chunking code, document range), and the loader checks
all of that before attaching anything.

That check held up. The Kaggle runs used Python 3.13 on another machine, and for all 103 shards the
number of chunks per document matched what the local chunker produced.

## What went wrong

1. **A Kaggle push failed once** with an empty server response (`Expecting value: line 1 column 1`).
   The orchestration script did not stop, moved to the next block, and left block 2 unsubmitted. It
   was resubmitted under a second kernel name and ran in parallel with block 3. Pushes now retry, and
   the script takes a `--slug` to run two jobs at once.
2. **Qdrant was killed for lack of memory** during the BM25 pass, at document 497,400 of 511,962. No
   data was lost (3,014,147 points survived), the pass resumed in under a minute, and nothing similar
   happened in the following 3 hours. The fixes were to bound segment size (250 MB) and memory-map
   segments early, to cap Qdrant at two cores, and to make the vector loader wait for the index to
   settle between shards.
3. **Block sizes were uneven.** Blocks were split by shard count, and the first one held 48% of all
   chunks (Gmail threads and meeting transcripts produce many chunks), so it took 55 minutes
   instead of the 40 I had estimated.

## Findings that matter for the next phases

- **Dense search is slow in this setup.** On the full index a dense query takes about 4 to 7 seconds,
  and a BM25 query 1 to 7 seconds, against a target of under 1 second. With real benchmark questions
  and Qdrant capped at the two cores used during ingestion, the median was 7.0 s for dense (p95 12 s)
  and 1.3 s for BM25 (p95 3.8 s). It is not disk-bound (about
  16 MB read per batch of queries) and the quantized search path is the one being used. Qdrant's
  per-segment counters show about 417 ms per segment on average, across 19 HNSW segments of about
  160,000 points that are searched for every query; lowering `ef` from 128 to 16 only halves the
  time, so a large fixed cost per segment dominates. The 250 MB segment cap that prevented the memory
  crash is a likely contributor. Levers to test, in order of cost: fewer and larger segments
  (needs more memory while rebuilding), a smaller dense vector (the model supports Matryoshka
  truncation to 256 dimensions, which needs only a reload from the shards, not a new GPU run), more
  RAM for the Docker VM, or serving the index from a larger machine. This is a Phase 3 and 6 task and
  the retrieval design does not depend on it.
- **The two-write design costs time.** Writing a point once with BM25 and payload and then adding its
  dense vector makes Qdrant rewrite the point. Loading 3 million vectors that way took about 2 hours
  of local CPU, while the GPU part took 2. A single pass that combines the chunk, its BM25 vector and
  its dense vector from the shards would avoid the rewrite and is the first thing to change if the
  corpus has to be re-indexed.

## Preliminary retrieval check

A smoke test of the finished index, not the evaluation (that is Phase 4): every fifth question that has
gold documents, 94 of the 470, each searched with its full text. Dense and BM25 each return their top
50 chunks, the hybrid mode merges the two lists with Reciprocal Rank Fusion, and the score is the share
of the gold documents found among the first 10 distinct documents. No reranking, no tuning, and with 94
questions each figure carries an uncertainty of several points.

| Mode | Document recall@10 | At least one gold document |
|---|---|---|
| Dense | 0.617 | 0.681 |
| BM25 | 0.678 | 0.723 |
| Hybrid (RRF) | **0.696** | **0.745** |

By category, hybrid recall@10: basic 0.83 (35 questions), semantic 0.28 (25), intra-document reasoning
1.00 (8), project-related 0.65 (8), constrained 1.00 (6), conflicting information 1.00 (4),
completeness 0.56 (4), miscellaneous 1.00 (4). The semantic category, built to avoid shared keywords,
is the weakest and the one a reranker and better fusion should help most. The sample had no
`high_level` or `info_not_found` questions, since those have no gold documents.

## Reproduce

```bash
python -m rag_engine.ingestion.download
python -m rag_engine.workers.bulk_ingest stage
python -m rag_engine.workers.bulk_ingest index sparse
python -m rag_engine.workers.bulk_ingest dense-graph off
uv run --group kaggle python scripts/kaggle/orchestrate.py run --username <user> --wheel dist/*.whl --shards 0-102
python -m rag_engine.workers.bulk_ingest load-vectors
python -m rag_engine.workers.bulk_ingest dense-graph on
```

See [kaggle-embedding.md](kaggle-embedding.md) for the Kaggle setup and
[reports/corpus-profile.json](reports/corpus-profile.json) for the chunking profile.
