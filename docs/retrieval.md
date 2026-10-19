# Retrieval: how it works and what was measured

Phase 3 turns the index into a search service. A query is embedded and BM25-encoded, the two
searches run side by side, and their rankings are merged. Reranking arrives in Phase 4.

## How a query runs

```text
query ──┬─ embed (arctic-embed-m-v1.5, query prefix) ─ dense search, top 50 ─┐
        │                                                                     ├─ RRF ─ top N ─ (parent passages)
        └─ BM25 encode ──────────────────── BM25 search, top 50 ──────────────┘
```

- **Modes.** `dense`, `bm25` and `hybrid`. Every mode returns the same number of chunks, so modes
  can be compared on equal terms. Scores are only comparable inside one result: cosine
  similarity, BM25 score or fused RRF score.
- **Fusion.** Reciprocal Rank Fusion over chunks (`1 / (k + rank)` summed over the lists, `k = 60`),
  with an optional weight per retriever. Only ranks are used because the two score scales differ.
- **Filters.** `source_types` and `doc_ids`, applied inside Qdrant to both searches.
- **Parent passages.** `include_context` attaches the 3,200-character passage around each chunk,
  read from PostgreSQL by character offsets. The chunk text always lies inside its passage
  (checked on 100 real hits).
- **Trace.** Each result carries the rank every retriever gave the chunk and the time of each stage.

`POST /v1/search` exposes it:

```bash
curl -s localhost:8000/v1/search -H 'content-type: application/json' -d '{
  "query": "default wait time of the tool-call batching invoker",
  "mode": "hybrid", "limit": 5, "include_context": true,
  "filters": {"source_types": ["github"]}
}'
```

The response lists the hits (`rank`, `chunk_id`, `doc_id`, `title`, `section`, `text`, `context`,
`score`, `dense_rank`, `bm25_rank`), the distinct `document_ids` in rank order and `timings_ms`.
The embedding model loads at the first query (about 25 s on the development laptop), not at
startup, so the health endpoints answer whether or not the models are available; until the first
query succeeds, `/v1/search` answers `503` if they cannot load.

| Setting | Default | Meaning |
|---|---|---|
| `DENSE_TOP_K`, `SPARSE_TOP_K` | 50, 50 | Chunks each retriever searches |
| `FUSION_TOP_K` | 30 | Chunks a query returns when the caller does not say |
| `RRF_K`, `RRF_DENSE_WEIGHT`, `RRF_SPARSE_WEIGHT` | 60, 1.0, 1.0 | Fusion constant and weights |
| `QDRANT_HNSW_EF` | 64 | HNSW search effort for dense queries (Qdrant's default is 128) |

## Quality, through the API

Gold-document recall (share of a question's gold documents among the distinct documents of the
returned chunks) over every fifth question with gold documents, 94 of the 470, against the full
index:

| Mode | Chunks returned | Recall | `semantic` (25 questions) |
|---|---|---|---|
| Dense | 10 | 0.590 | 0.200 |
| BM25 | 10 | 0.666 | 0.320 |
| Hybrid | 10 | 0.658 | 0.320 |
| Hybrid | 30 | 0.765 | 0.440 |

This agrees with the offline measurement over all 470 questions in the
[ingestion report](ingestion-report.md#what-a-reranker-could-add) (hybrid, 30 candidates: 0.770), which
is the evidence that the service returns what the index holds. With 94 questions each figure
carries several points of uncertainty. At 10 chunks, equal-weight fusion is no better than BM25
alone on this sample, so the fusion weights are the first thing to tune once the evaluation
harness exists.

## Latency on the development laptop

This section corrects two earlier readings, so the reasoning is spelled out.

The numbers come from a 6 GB Docker VM that also hosts PostgreSQL and the API, with Qdrant capped
at two cores and the query embedded on the CPU. Stage timings for 94 hybrid queries with context:

| Stage | p50 | p95 |
|---|---|---|
| Embed the query (CPU) | 415 ms | 779 ms |
| Dense search | 2,648 ms | 3,934 ms |
| BM25 encode / search | 4 ms / 458 ms | 8 ms / 3,336 ms |
| Fuse | 1 ms | 2 ms |
| Parent passages (PostgreSQL) | 22 ms | 72 ms |
| **Total** | **3,116 ms** | **4,638 ms** |

The target is a p95 under one second for retrieve, fuse and rerank, and this setup does not meet
it. What the measurements say about why:

- **It depends on whether the pages Qdrant needs are in memory.** Repeating a query, or running
  queries that touch the same region of the graph, answers a dense search in about 10 ms. Queries
  that land in cold regions take seconds. The working set (the int8 vectors, 2.35 GB, are pinned in
  memory; the HNSW graphs, the half-precision originals and the payloads are read from disk on
  demand) does not fit in what a 6 GB VM leaves for the page cache, and the VM was swapping.
- **An earlier "32 ms" reading was biased.** It used 140 questions, all from the `basic` category,
  which cluster in one region of the graph. Diverse queries are mostly cold.
- **An earlier explanation was not supported.** A fixed cost per segment (19 segments) was blamed
  for the slowness. Nothing here shows that; the same segments answer in 10 ms when warm.
- **Rescoring with the on-disk originals is not the cost.** Turning it off gave the same recall
  (0.611 on 157 questions) and no speed-up.
- **Search effort is a cheap lever.** On queries drawn from random corpus vectors (cold regions):

  | `hnsw_ef` | Dense search p50 | p95 | Recall@10 (157 questions) |
  |---|---|---|---|
  | 128 (default) | 368 to 584 ms | 2.7 to 3.4 s | 0.611 |
  | 64 | 37 to 106 ms | 0.5 to 1.1 s | 0.608 |
  | 32 | 45 to 182 ms | 0.35 to 0.5 s | 0.591 |

  64 is the default for that reason. Ranges are two runs each; the laptop's cache state moves them.

The untested remedies, in order of cost: a machine with enough memory to keep the index cached
(the final numbers should be measured on one), or vectors truncated to 256 dimensions, which would
shrink the working set to a third at a cost of about 2.7 points of recall (see the ingestion
report) and takes a reload from the existing shards, about two hours of CPU, not a new GPU run.

## Not in this phase

Reranking and the evaluation harness (Phase 4), including the two grep baselines the project will be
measured against; query rewriting and keyword extraction for the BM25 side; generation and citation
verification (Phase 5).
