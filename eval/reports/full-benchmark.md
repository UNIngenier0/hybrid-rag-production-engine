# The whole benchmark

All 500 questions, answered by hybrid retrieval with reranking and claude-haiku-5-5 (5 passages each, mode `hybrid+rerank`) and scored by the benchmark's own evaluator with claude-haiku-5-5 as the judge, against the original gold answers. The judge is not the GPT model of the public leaderboard, so these numbers are not comparable with it.

| | Questions | Correct | Completeness | Combined | Document recall | Invalid extra docs |
|---|---|---|---|---|---|---|
| **All questions** | 500 | 55.4% | 61.36% | **50.97** | 68.01% | 1.28 |
| basic | 175 | 73.71% | 73.5% | **69.29** | 80.57% | 1.08 |
| completeness | 20 | 30.0% | 40.79% | **27.5** | 35.49% | 1.75 |
| conflicting_info | 20 | 85.0% | 88.26% | **80.61** | 90.0% | 0.5 |
| constrained | 30 | 56.67% | 79.4% | **51.23** | 86.67% | 0.57 |
| high_level | 10 | 60.0% | 64.83% | **48.33** | - | - |
| info_not_found | 20 | 100.0% | 100.0% | **100.0** | - | - |
| intra_document_reasoning | 40 | 55.0% | 73.75% | **52.92** | 95.0% | 0.8 |
| miscellaneous | 20 | 90.0% | 84.0% | **79.0** | 100.0% | 0.75 |
| project_related | 40 | 12.5% | 39.06% | **6.86** | 41.38% | 1.45 |
| semantic | 125 | 29.6% | 32.1% | **25.65** | 42.4% | 1.96 |

The benchmark's headline is the combined score over the questions, **50.97**. Averaging the 10 question types instead, as the leaderboard describes, gives **54.14**. Answering all of it cost $0.42 and judging it $0.44.
