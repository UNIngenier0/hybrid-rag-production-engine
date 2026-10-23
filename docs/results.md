# Results

Everything this project measured, in one place, with where each number comes from and how far it
can be trusted. The numbers are quoted from the reports in `eval/reports` and the run folders in
`eval/runs`, and the regression gate recomputes them from those files in every build, so this page
cannot drift from them without CI failing.

**Read this first.** The corpus is EnterpriseRAG-Bench: 511,962 synthetic documents of an invented
company and 500 questions about them. Every system here, and the judge that scored it, is Claude
Haiku 5.5, chosen to keep a full run under a dollar. The benchmark's public leaderboard uses a GPT
judge, so none of these scores is comparable with a leaderboard entry. What they do support is the
comparison between systems inside this project, which used the same model, judge, questions and
evidence budget for each (the fairness checks are in [evaluation.md](evaluation.md#fairness-audit)).

## The system in numbers

| | |
|---|---|
| Corpus indexed | 511,962 documents, **3,064,558 chunks**, each with a dense vector and a BM25 vector |
| Dense model | `snowflake-arctic-embed-m-v1.5`, 768 dimensions; computed on a free Kaggle GPU in 2.16 GPU-hours |
| Retrieval | dense + BM25 searched in parallel, reciprocal rank fusion, 30 candidates reranked by `bge-reranker-v2-m3`, the best 5 chunks shown with their parent passages |
| Answer | Claude Haiku 5.5 writes cited prose; a second call checks every statement against the passage it cites |
| Cost of an answer | **$0.0008** to write it, about **$0.0016** with the citation check |
| Whole benchmark | answering all 500 questions cost $0.42, judging them $0.44 |

## The whole benchmark

All 500 questions, one run, scored by the benchmark's own evaluator ([full report](../eval/reports/full-benchmark.md);
the answers in the benchmark's file format are in [`eval/submission`](../eval/submission)):

| | Questions | Correct | Completeness | Combined | Document recall |
|---|---|---|---|---|---|
| **All questions** | 500 | 55.4% | 61.4% | **50.97** | 68.0% |
| basic | 175 | 73.7% | 73.5% | 69.29 | 80.6% |
| conflicting_info | 20 | 85.0% | 88.3% | 80.61 | 90.0% |
| miscellaneous | 20 | 90.0% | 84.0% | 79.00 | 100.0% |
| intra_document_reasoning | 40 | 55.0% | 73.8% | 52.92 | 95.0% |
| constrained | 30 | 56.7% | 79.4% | 51.23 | 86.7% |
| high_level | 10 | 60.0% | 64.8% | 48.33 | |
| info_not_found | 20 | 100.0% | 100.0% | 100.00 | |
| completeness | 20 | 30.0% | 40.8% | 27.50 | 35.5% |
| semantic | 125 | 29.6% | 32.1% | 25.65 | 42.4% |
| project_related | 40 | 12.5% | 39.1% | 6.86 | 41.4% |

*Combined* is the benchmark's score: completeness, counted as zero when the answer is not correct.
Over the questions it is 50.97; averaged over the ten question types, as the leaderboard describes
its overall score, it is 54.14.

**The shape of the result.** The system is strong where the answer sits in one document the query
can name: single facts with identifiers (`basic`), reasoning inside one document, conflicting
sources, the odd mixed cases, and recognising that the corpus does not hold an answer (20 of 20).
It is weak in three places, and the weakness is retrieval, not writing: `semantic` questions, which
describe what they want without its exact words (the answers cite 42% of the gold documents);
`project_related` and `completeness` questions, whose answers are spread over many documents that
five passages cannot hold (41% and 35%). That matches the grep comparison below, where the iterating agent wins on exactly those.

## Retrieval

Document recall (the share of the gold documents found among the first chunks) over the 470
questions that have gold documents ([ablation](../eval/reports/retrieval-ablation.md)):

| System | Recall at 5 chunks | Recall at 10 | Hit rate at 5 |
|---|---|---|---|
| Dense only | 0.491 | 0.540 | 0.564 |
| BM25 only | 0.601 | 0.674 | 0.677 |
| Hybrid (fused) | 0.622 | 0.694 | 0.704 |
| **Hybrid + rerank** | **0.684** | **0.720** | **0.755** |

BM25 beats the dense model on this corpus, which is full of identifiers (flag names, region codes,
pool IDs); fusing the two adds 2 points over BM25 alone, and the cross-encoder adds 6 more. The
reranker choice, its cost and the pools tested are in [evaluation.md](evaluation.md#reranker-cost):
`bge-reranker-v2-m3` was the most accurate, `bge-reranker-base` with a pool of 20 reached 0.651, and
all of them need a GPU for the latency to be reasonable.

## Against the two grep baselines

The benchmark's own iterative grep agent and a single-attempt grep, with the same model and the same
judge ([report](../eval/reports/answer-comparison.md)). On 96 questions, 12 of each type that has
gold documents:

| System | Combined | Cost of the answers |
|---|---|---|
| **Hybrid + rerank (ours)** | **50.6** | $0.08 |
| Iterative grep agent | 40.0 | $1.94 |
| Single-attempt grep | 4.8 | $0.09 |

- **Against the single attempt the difference is large and robust** (-45.8, interval -55.5 to -36.2).
- **Against the agent the pipeline is ahead in both runs, but by how much is not settled**: +10.7
  (interval +0.7 to +20.6) in the first run, +6.4 (-4.7 to +17.4) in a second run of the same
  pipeline on the same questions. At a twenty-fifth of the cost per question.
- **The agent wins where answers are spread over many documents** (`completeness`, `project_related`),
  which is the case for a retrieve-check-retrieve-again step.

On the 30 questions with no gold documents ([report](../eval/reports/answer-comparison-unanswerable.md)):
ours 82.8, agent 83.8 (no measurable difference, +1.1 with an interval of -10.5 to +12.6), single
grep 66.7. All three recognise that the information is missing in 20 of 20; on the 10 broad
questions the agent is nominally ahead (51.5 against 48.3) and the single grep gets none. The agent
spent $1.43 on these 30 questions and the pipeline $0.03.

**Why the headline needs the audit.** The 96-question sample flatters the reranker (+0.154 recall
there against +0.055 on the other 374 questions), the generation step alone moves a score by about
4 points between runs, and the judge, asked twice, moves it by less than 1. The 470-question score
of the pipeline is 48.9, not 50.6. Everything in this paragraph is measured and written up in the
[fairness audit](evaluation.md#fairness-audit).

## Grounding: citations and honesty

- **Every statement is checked against its passage** ([method and results](generation.md)). On 96
  real answers, 66% of the statements came out verified, 18% carried no citation, 6% were only
  partly supported and 1% unsupported. Planted errors are caught: a citation moved to another
  passage 99% of the time, a number changed to one the passage does not contain 98%.
- **The check is strict**: one answer in four passes completely, and the biggest group of problems
  is a missing citation, not a wrong one.
- **The answer says when the documents lack something** ([details](generation.md#when-the-documents-do-not-hold-the-answer)).
  Answers that state that something is missing were judged correct 15% of the time, the others 84%;
  the statement marks 177 of the 219 wrong answers among the 470. A threshold on the reranker score
  is a much weaker signal (it separates correct from wrong answers with probability 0.68), so it
  ships as an optional setting, off.

## Speed and load

All measured on the development laptop (Apple silicon, Docker limited to 6 GB, the index mostly on
disk, embeddings and a small reranker on the CPU), where the sub-second retrieval target is **not
met** ([operations.md](operations.md#load-test)):

- An answer takes 33.7 s on average, three quarters of it retrieval (query embedding 6.7 s, the two
  searches about 11 s in parallel, CPU reranking 8.3 s), 4.0 s writing, 3.7 s checking citations.
- Throughput stops growing near 0.1 searches per second: four requests at once each take four times
  as long as one alone. The machine is the limit; a GPU for the cross-encoder (under
  a second for 30 passages instead of seconds) and enough RAM for the index would change that, and neither is
  measured here.
- A repeated question is answered from Redis in milliseconds at no cost.

## What these results cannot support

- **A place on the public leaderboard.** The judge differs, and the leaderboard asks for a
  reproduction guide and a sandbox ([leaderboard.md](leaderboard.md)). The score of a GPT-judged
  run is not known.
- **A claim that the pipeline beats the iterating agent** in general. It is ahead in both runs, by
  an amount the 96 and 30 questions cannot pin down; the replication on a second disjoint sample,
  which would settle it, is not done.
- **Production latency.** Only the development laptop was measured.
- **Quality on real company data.** The corpus is synthetic, generated by language models, and the
  questions come from the same generator.
- **Generalising the weak categories.** `project_related` and `completeness` are 60 questions
  between them; the pattern is clear and the exact figures are not.
