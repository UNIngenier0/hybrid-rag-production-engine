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
