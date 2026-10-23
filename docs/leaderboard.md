# Submitting to the EnterpriseRAG-Bench leaderboard

What the leaderboard asks for, what this repository already has ready, and what is still needed
before an entry could honestly be made. Nothing has been submitted: that is a step for the author
to take, by email, from the details below.

## What the benchmark asks

From the [benchmark's README](https://github.com/onyx-dot-app/EnterpriseRAG-Bench#leaderboard): to
submit, contact the maintainers (the address is in that README), and every submission has to be
reproducible: an open-source system provides a guide for reproducing its results, a closed one a
sandbox or endpoint where questions can be sent for verification. The maintainers' own product is
excluded from the ranking. The 100 extra metadata questions are not part of it.

This project is open source, so the route is the guide: [reproduce.md](reproduce.md).

## What is ready

| Item | Where |
|---|---|
| The answers to all 500 questions, in the evaluator's file format (`question_id`, `answer`, `document_ids`) | [`eval/submission/answers.jsonl`](../eval/submission/answers.jsonl) |
| What produced them: models, retrieval, index, checksum of the file | [`eval/submission/manifest.json`](../eval/submission/manifest.json) |
| The score they get here, per question type | [`eval/reports/full-benchmark.md`](../eval/reports/full-benchmark.md) |
| A reproduction guide, with the cost and time of each step | [reproduce.md](reproduce.md) |
| The caveats an entry must carry | [results.md](results.md#what-these-results-cannot-support) |

`python -m rag_engine.evaluation.cli full-benchmark` rebuilds the three files from the judged
runs, refuses to write them if any benchmark question is missing or answered twice, and the
regression gate fails the build if the committed copies stop matching the runs.

The answers are what the pipeline writes with its defaults: hybrid retrieval with
`bge-reranker-v2-m3`, five passages per question, Claude Haiku 5.5. `document_ids` are the
documents the answer cites, which is what the benchmark's document metrics score.

## What is missing before an entry

1. **The leaderboard's judge.** Scores on the public leaderboard come from the benchmark's evaluator
   with a GPT judge; the ones here come from the same evaluator with Claude Haiku 5.5, scored
   against the original gold answers (`--no-correction`). Run the evaluator the maintainers use on
   `eval/submission/answers.jsonl`, with their judge and settings, from a checkout of the benchmark:

   ```bash
   python -m src.scripts.answer_evaluation.metrics_based_eval \
       --answers-file /path/to/eval/submission/answers.jsonl
   ```

   Until that is run, the score of this system on the leaderboard's terms is not known, and the
   figures in [results.md](results.md) must not be quoted next to a leaderboard entry. The README
   targets that name other entries (above LangChain and LlamaIndex) have not been tested for that
   reason.
2. **A fresh end-to-end run, if the maintainers want one.** The answers were written from stored
   retrieval results (the candidates collected from the index, with the reranker scores computed on
   a GPU), through the same fusion and reranking code the service runs; the live service was
   checked against that replay on 32 questions ([retrieval.md](retrieval.md#checked-against-the-running-service)).
   To answer all 500 through the live endpoint instead, send each question to `POST /v1/query` with
   `"cache": false` and keep `answer` and `document_ids`; with a GPU for the reranker that takes
   minutes, on the development laptop about 33 seconds per question.
3. **Honest framing.** The system is a retrieval pipeline with one answering step. On the
   benchmark's categories it is strong on single-document questions and weak on `semantic`,
   `project_related` and `completeness` ones; the entry should say so. An agentic variant that
   retrieves again when the first pass is thin is the obvious next step and is not built.
