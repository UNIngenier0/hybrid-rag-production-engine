# Abstention by evidence strength

The best cross-encoder score among the 30 reranked passages of each question, by question type. A threshold on this score is the simplest rule for declining to answer.

| Question type | Questions | Lowest | 25% | Median | 75% | Highest |
|---|---|---|---|---|---|---|
| basic | 175 | 0.225 | 0.928 | 0.966 | 0.989 | 0.999 |
| completeness | 20 | 0.415 | 0.932 | 0.955 | 0.974 | 0.999 |
| conflicting_info | 20 | 0.950 | 0.979 | 0.993 | 0.997 | 1.000 |
| constrained | 30 | 0.778 | 0.964 | 0.983 | 0.995 | 1.000 |
| high_level | 10 | 0.031 | 0.892 | 0.961 | 0.986 | 0.998 |
| info_not_found | 20 | 0.260 | 0.519 | 0.690 | 0.832 | 0.989 |
| intra_document_reasoning | 40 | 0.811 | 0.961 | 0.973 | 0.992 | 0.999 |
| miscellaneous | 20 | 0.053 | 0.864 | 0.929 | 0.988 | 0.996 |
| project_related | 40 | 0.734 | 0.897 | 0.965 | 0.983 | 1.000 |
| semantic | 125 | 0.189 | 0.590 | 0.792 | 0.925 | 0.997 |

## Telling `info_not_found` questions from answerable ones

Probability that a random answerable question has a higher best score than a random `info_not_found` one: **0.82** (0.50 means the score carries no information, 1.00 a perfect separation).

Each row is a threshold, chosen so that it refuses the stated share of the answerable questions:

| Answerable refused | Threshold | `info_not_found` caught | High-level refused |
|---|---|---|---|
| 0% (0) | 0.0530 | 0 of 20 | 1 of 10 |
| 1% (4) | 0.2251 | 0 of 20 | 1 of 10 |
| 2% (9) | 0.3176 | 2 of 20 | 1 of 10 |
| 5% (23) | 0.4949 | 5 of 20 | 1 of 10 |
| 10% (47) | 0.6304 | 9 of 20 | 2 of 10 |
| 20% (94) | 0.8223 | 15 of 20 | 2 of 10 |

## Stopping wrong answers

Among the answerable questions, whether the answer was judged correct, against the best score: probability that a correct answer has the higher score **0.68**.

| Threshold | Wrong answers avoided | Right answers given up |
|---|---|---|
| 0.0530 | 0 of 219 | 0 of 251 |
| 0.2251 | 3 of 219 | 1 of 251 |
| 0.3176 | 8 of 219 | 1 of 251 |
| 0.4949 | 21 of 219 | 2 of 251 |
| 0.6304 | 36 of 219 | 11 of 251 |
| 0.8223 | 62 of 219 | 32 of 251 |

## What the answer says about itself

`answered`: nothing is said to be missing. `partial`: part is stated and the rest is said to be missing. `not_found`: only that nothing was found. Read from the answer text, with no model call.

| Questions | Status | Answers | Judged correct |
|---|---|---|---|
| answerable | answered | 261 | 219 (84%) |
| answerable | partial | 207 | 32 (15%) |
| answerable | not_found | 2 | 0 (0%) |
| info_not_found | partial | 20 | 20 (100%) |
| high_level | answered | 6 | 5 (83%) |
| high_level | partial | 4 | 1 (25%) |
