# Answer comparison

30 questions answered by every system, scored by the benchmark's evaluator against the original gold answers and documents. *Combined* is the benchmark's score: the mean over questions of completeness, counted as zero when the answer is not correct. The judge is a Claude model, not the GPT model of the public leaderboard, so these numbers are not comparable with it. `*` marks a paired difference whose 95% interval excludes zero.

| System | Correct | Completeness | Combined | Doc recall | Invalid extra docs | Combined vs Hybrid + rerank (ours) | Cost of the answers |
|---|---|---|---|---|---|---|---|
| Hybrid + rerank (ours) | 86.7% | 88.3% | **82.8** | 0.0% | 0.00 |  | $0.03 ($0.0008 per question) |
| Iterative grep agent | 90.0% | 86.6% | **83.8** | 0.0% | 0.00 | +1.1 (-10.5, +12.6) | $1.43 ($0.0477 per question) |
| Single-attempt grep | 66.7% | 73.7% | **66.7** | 0.0% | 0.00 | -16.1 (-28.3, -3.9)* | $0.03 ($0.0010 per question) |

## Combined score by question type

| System | high_level (10) | info_not_found (20) |
|---|---|---|
| Hybrid + rerank (ours) | 48.3 | 100.0 |
| Iterative grep agent | 51.5 | 100.0 |
| Single-attempt grep | 0.0 | 100.0 |

## Fairness checks

| Check | Result | Detail |
|---|---|---|
| same questions | pass | 30 questions |
| every question answered | pass | all answered |
| every answer scored | pass | all scored |
| same gold answers for everyone | pass | judged against the original gold set (no correction) |
| same judge | pass | claude-haiku-5-5 for every system (claude-haiku-5-5) |
| same answering model | pass | claude-haiku-5-5 for every system |
| same evidence budget for single-pass systems | pass | 5 passages or files per question for Hybrid + rerank (ours), Single-attempt grep |
| same document-reporting rule | pass | documents reported = cited for Hybrid + rerank (ours), Single-attempt grep |
| cost measured for every system | pass | every system has a usage report or token counts |

Differences that remain by design:

- Iterative grep agent decides for itself how much to read and search (cost and time are reported next to its score).
- Iterative grep agent: answers end with a source list, so the evaluator removes citations before judging, as the benchmark does by default.
