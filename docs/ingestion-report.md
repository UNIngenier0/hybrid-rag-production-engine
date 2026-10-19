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

- **Search latency depends on how much of the index is in memory.** A dense query takes about
  10 ms when the pages it reads are cached and seconds when they are not, and a 6 GB Docker VM
  cannot keep the whole working set cached next to anything else (the int8 vectors alone are
  2.35 GB and are pinned in memory). Diverse queries are mostly cold: medians from 0.4 to 2.6
  seconds in the runs made. Two earlier readings were wrong. The first measurements (4 to 7
  seconds) blamed a fixed cost per segment, which the later runs do not support, and a later
  "32 ms" reading used 140 questions of one category, which all hit the same region of the
  graph. The measurements, the cheap lever found (a lower HNSW search effort) and the remaining
  options are in [retrieval.md](retrieval.md). Truncating to 256 dimensions would shrink the
  working set to a third; it needs a reload from the shards (about 2 hours of CPU), not a new
  GPU run.
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

## Does the vector dimension matter?

An exact (brute-force) dense search straight from the shards, with no Qdrant involved, over all 470
questions that have gold documents. Each chunk vector is truncated to its first N dimensions and
re-normalized (the model is trained with Matryoshka representation learning, so this is its
intended way to shrink). The score is the same document recall@10 as above, dense only.

| Dimensions | Recall@10, 470 questions | Change vs 768 (paired, 95% CI) | Vectors in RAM as int8 |
|---|---|---|---|
| 768 | 0.584 | | 2.35 GB |
| 512 | 0.568 | -0.016 (0.001 to 0.031) | 1.57 GB |
| 384 | 0.559 | -0.025 (0.008 to 0.042) | 1.18 GB |
| 256 | 0.558 | -0.027 (0.007 to 0.047) | 0.78 GB |
| 128 | 0.434 | -0.150 (0.116 to 0.184) | 0.39 GB |

- **The index loses nothing.** On the 94-question sample used above, the exact search gives 0.622
  at 768 dimensions and Qdrant's quantized HNSW gave 0.617. The approximate index is not the reason
  dense recall is modest.
- **The 94-question sample was optimistic.** Over all 470 questions dense recall is 0.584, not 0.617,
  and the `semantic` category is 0.248 (125 questions), not 0.28. Treat the hybrid figures above the
  same way until the full evaluation runs.
- **256 dimensions cost about 2.7 points of recall** (about 5% relative) and cut the in-memory
  vectors to a third; 128 dimensions is too lossy. The loss is small but real, so truncation is a
  memory trade (what it buys is smaller memory, see the latency finding above), not a free change.
- **The dimension is not what limits quality.** The weak categories (`semantic`, `completeness`,
  `project_related`) lose almost nothing between 768 and 256 dimensions because they are already
  low at 768; they need reranking and better candidate selection, not more dimensions.

The script took 6 minutes on 2 cores (373 s to score 3.06 million chunks, 33 s to embed the
queries).

## What a reranker could add

A reranker only reorders the candidates it is given, so the candidate set sets its ceiling. This
measures that ceiling on all 470 questions with gold documents: the exact dense list from the
shards plus Qdrant's BM25 list (both using the whole question as the query), fused with
chunk-level Reciprocal Rank Fusion (k = 60). "Gold recall" is the share of a question's gold
documents among the distinct documents of the selected chunks; "cap@5" is the same if a perfect
reranker could pass only 5 chunks (one per document) to the LLM.

| Candidate set | Chunks | Gold recall | At least one gold | All gold | cap@5 |
|---|---|---|---|---|---|
| Dense, top 50 | 50 | 0.679 | 0.730 | 0.634 | 0.679 |
| BM25, top 50 | 50 | 0.799 | 0.828 | 0.757 | 0.793 |
| Hybrid, no reranker, top 5 | 5 | 0.618 | 0.700 | 0.564 | 0.618 |
| Hybrid, no reranker, top 10 | 10 | 0.675 | 0.745 | 0.621 | 0.675 |
| **Hybrid 50+50, keep 30 (the design)** | 30 | **0.770** | 0.811 | 0.723 | 0.768 |
| Hybrid 100+100, keep 50 | 50 | 0.803 | 0.836 | 0.755 | 0.799 |
| Hybrid 100+100, keep 100 | 100 | 0.841 | 0.866 | 0.802 | 0.835 |

By category (gold recall): hybrid top 10 chunks / design candidates / 100+100 keep 100.

| Category | n | Top 10 | Design | Deep |
|---|---|---|---|---|
| basic | 175 | 0.806 | 0.880 | 0.931 |
| semantic | 125 | 0.360 | 0.480 | 0.608 |
| intra_document_reasoning | 40 | 0.975 | 1.000 | 1.000 |
| project_related | 40 | 0.497 | 0.698 | 0.868 |
| constrained | 30 | 0.883 | 0.933 | 0.933 |
| conflicting_info | 20 | 0.875 | 0.975 | 1.000 |
| completeness | 20 | 0.416 | 0.627 | 0.688 |
| miscellaneous | 20 | 1.000 | 1.000 | 1.000 |

- **A perfect reranker over the design's 30 candidates would lift the 5 chunks that reach the LLM
  from 0.618 to 0.768 gold recall**, at most 15 points. A real reranker captures only part of it.
- **The context budget is not the limit.** Questions have 1.58 gold documents on average (median 1,
  maximum 10) and only 5.5% have more than 5, so passing 5 chunks costs almost nothing against
  passing all candidates.
- **BM25 is the stronger retriever on this corpus** (0.799 against 0.679 for dense at 50 chunks), so
  equal-weight fusion may be leaving recall behind; weighted fusion is worth testing.
- **`semantic` is a candidate problem, not an ordering problem.** Even a perfect order over the
  design's 30 candidates reaches 0.480, and 100+100 candidates 0.608. It needs better first-stage
  retrieval (rewriting the query, keywords for BM25), not only a reranker.
- **Depth helps most where several documents are needed**: `project_related` goes from 0.497 to 0.698
  to 0.868 and `completeness` from 0.416 to 0.627 to 0.688. The price is reranker work, which is
  proportional to the number of candidates.

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
