# Retrieval ablation

470 questions with gold documents, scored on the distinct documents among the first N chunks a system returns. Dense and BM25 search 100 chunks deep over the full index; fusion is chunk-level RRF (k = 60) and uses the first 50 of each list unless stated. Recall is the share of a question's gold documents that were found. `*` marks a paired difference whose 95% interval excludes zero.

## 5 chunks returned

| System | Recall | Any gold | All gold | MRR | nDCG@10 | Recall vs hybrid |
|---|---|---|---|---|---|---|
| dense | 0.491 | 0.564 | 0.445 | 0.470 | 0.443 | -0.118 (-0.151, -0.085)* |
| bm25 | 0.601 | 0.677 | 0.549 | 0.594 | 0.563 | -0.007 (-0.036, +0.022) |
| hybrid, equal weights | 0.608 | 0.691 | 0.557 | 0.579 | 0.551 |  |
| hybrid, BM25 weight 1.5 | 0.622 | 0.704 | 0.568 | 0.604 | 0.576 | +0.014 (+0.003, +0.024)* |
| hybrid, BM25 weight 2 | 0.624 | 0.706 | 0.568 | 0.604 | 0.577 | +0.015 (+0.001, +0.029)* |
| hybrid, BM25 weight 3 | 0.622 | 0.706 | 0.566 | 0.611 | 0.580 | +0.014 (-0.003, +0.031) |
| hybrid, depth 100 | 0.585 | 0.666 | 0.534 | 0.571 | 0.540 | -0.023 (-0.039, -0.007)* |
| rerank minilm-6, pool 20 | 0.624 | 0.702 | 0.570 | 0.595 | 0.568 | +0.015 (-0.013, +0.044) |
| rerank minilm-6, pool 30 | 0.620 | 0.694 | 0.568 | 0.581 | 0.557 | +0.011 (-0.019, +0.041) |
| rerank minilm-6, pool 50 | 0.619 | 0.696 | 0.568 | 0.577 | 0.553 | +0.011 (-0.022, +0.043) |
| rerank minilm-12, pool 20 | 0.646 | 0.723 | 0.594 | 0.616 | 0.589 | +0.038 (+0.012, +0.063)* |
| rerank minilm-12, pool 30 | 0.631 | 0.706 | 0.581 | 0.606 | 0.579 | +0.023 (-0.006, +0.052) |
| rerank minilm-12, pool 50 | 0.629 | 0.706 | 0.577 | 0.607 | 0.578 | +0.020 (-0.011, +0.051) |
| rerank bge-base, pool 20 | 0.651 | 0.728 | 0.598 | 0.634 | 0.604 | +0.043 (+0.016, +0.069)* |
| rerank bge-base, pool 30 | 0.641 | 0.717 | 0.587 | 0.624 | 0.594 | +0.032 (+0.004, +0.061)* |
| rerank bge-base, pool 50 | 0.641 | 0.719 | 0.585 | 0.625 | 0.594 | +0.032 (+0.000, +0.064)* |
| rerank mxbai-base, pool 20 | 0.646 | 0.726 | 0.596 | 0.629 | 0.596 | +0.038 (+0.011, +0.065)* |
| rerank mxbai-base, pool 30 | 0.655 | 0.732 | 0.602 | 0.626 | 0.597 | +0.047 (+0.017, +0.076)* |
| rerank mxbai-base, pool 50 | 0.645 | 0.719 | 0.594 | 0.620 | 0.591 | +0.036 (+0.004, +0.069)* |
| rerank bge-v2-m3, pool 20 | 0.673 | 0.745 | 0.619 | 0.662 | 0.633 | +0.065 (+0.037, +0.093)* |
| rerank bge-v2-m3, pool 30 | 0.684 | 0.755 | 0.630 | 0.665 | 0.638 | +0.076 (+0.046, +0.105)* |
| rerank bge-v2-m3, pool 50 | 0.683 | 0.753 | 0.630 | 0.667 | 0.638 | +0.075 (+0.043, +0.106)* |

## 10 chunks returned

| System | Recall | Any gold | All gold | MRR | nDCG@10 | Recall vs hybrid |
|---|---|---|---|---|---|---|
| dense | 0.540 | 0.613 | 0.494 | 0.477 | 0.463 | -0.129 (-0.162, -0.096)* |
| bm25 | 0.674 | 0.732 | 0.615 | 0.603 | 0.594 | +0.005 (-0.021, +0.030) |
| hybrid, equal weights | 0.669 | 0.740 | 0.615 | 0.586 | 0.577 |  |
| hybrid, BM25 weight 1.5 | 0.694 | 0.760 | 0.638 | 0.612 | 0.606 | +0.025 (+0.008, +0.041)* |
| hybrid, BM25 weight 2 | 0.696 | 0.764 | 0.640 | 0.613 | 0.607 | +0.027 (+0.010, +0.045)* |
| hybrid, BM25 weight 3 | 0.689 | 0.755 | 0.632 | 0.618 | 0.609 | +0.020 (+0.000, +0.040)* |
| hybrid, depth 100 | 0.663 | 0.734 | 0.611 | 0.581 | 0.571 | -0.006 (-0.019, +0.007) |
| rerank minilm-6, pool 20 | 0.693 | 0.762 | 0.640 | 0.605 | 0.596 | +0.024 (+0.000, +0.048)* |
| rerank minilm-6, pool 30 | 0.694 | 0.762 | 0.643 | 0.592 | 0.587 | +0.025 (-0.000, +0.051) |
| rerank minilm-6, pool 50 | 0.691 | 0.764 | 0.638 | 0.587 | 0.581 | +0.023 (-0.005, +0.050) |
| rerank minilm-12, pool 20 | 0.696 | 0.764 | 0.649 | 0.622 | 0.609 | +0.027 (+0.004, +0.051)* |
| rerank minilm-12, pool 30 | 0.693 | 0.764 | 0.640 | 0.615 | 0.603 | +0.024 (-0.000, +0.049) |
| rerank minilm-12, pool 50 | 0.688 | 0.760 | 0.636 | 0.615 | 0.600 | +0.019 (-0.009, +0.047) |
| rerank bge-base, pool 20 | 0.707 | 0.772 | 0.657 | 0.641 | 0.628 | +0.038 (+0.017, +0.059)* |
| rerank bge-base, pool 30 | 0.719 | 0.779 | 0.668 | 0.633 | 0.625 | +0.050 (+0.027, +0.074)* |
| rerank bge-base, pool 50 | 0.703 | 0.764 | 0.651 | 0.632 | 0.620 | +0.034 (+0.006, +0.062)* |
| rerank mxbai-base, pool 20 | 0.685 | 0.757 | 0.632 | 0.634 | 0.614 | +0.017 (-0.004, +0.038) |
| rerank mxbai-base, pool 30 | 0.692 | 0.764 | 0.636 | 0.631 | 0.615 | +0.024 (-0.001, +0.048) |
| rerank mxbai-base, pool 50 | 0.700 | 0.768 | 0.647 | 0.628 | 0.615 | +0.031 (+0.002, +0.060)* |
| rerank bge-v2-m3, pool 20 | 0.704 | 0.768 | 0.651 | 0.665 | 0.647 | +0.035 (+0.014, +0.056)* |
| rerank bge-v2-m3, pool 30 | 0.720 | 0.779 | 0.668 | 0.669 | 0.654 | +0.051 (+0.027, +0.075)* |
| rerank bge-v2-m3, pool 50 | 0.725 | 0.783 | 0.674 | 0.672 | 0.657 | +0.056 (+0.029, +0.083)* |

## Candidate pools (ceiling for a reranker)

| Candidate pool | Recall |
|---|---|
| dense, 50 chunks | 0.645 |
| BM25, 50 chunks | 0.799 |
| hybrid depth 50, 30 chunks | 0.768 |
| hybrid depth 50, 50 chunks | 0.804 |
| hybrid depth 100, 50 chunks | 0.803 |
| hybrid depth 100, 100 chunks | 0.845 |

## Recall by question type, 5 chunks

| System | basic (175) | completeness (20) | conflicting_info (20) | constrained (30) | intra_document_reasoning (40) | miscellaneous (20) | project_related (40) | semantic (125) |
|---|---|---|---|---|---|---|---|---|
| hybrid, equal weights | 0.749 | 0.267 | 0.725 | 0.833 | 0.950 | 0.950 | 0.378 | 0.304 |
| dense | 0.629 | 0.185 | 0.675 | 0.783 | 0.600 | 0.850 | 0.325 | 0.208 |
| bm25 | 0.731 | 0.331 | 0.825 | 0.800 | 0.925 | 0.900 | 0.388 | 0.296 |
| rerank bge-v2-m3, pool 30 | 0.806 | 0.370 | 0.950 | 0.883 | 0.950 | 1.000 | 0.414 | 0.424 |

## Reranker cost

| Reranker | Model | Scored on | ms per pair | Time to score 30 passages |
|---|---|---|---|---|
| minilm-6 | `cross-encoder/ms-marco-MiniLM-L-6-v2` | cuda | 1.5 | 46 ms |
| minilm-12 | `cross-encoder/ms-marco-MiniLM-L-12-v2` | cuda | 2.5 | 74 ms |
| bge-base | `BAAI/bge-reranker-base` | cuda | 6.0 | 181 ms |
| mxbai-base | `mixedbread-ai/mxbai-rerank-base-v1` | cuda | 12.8 | 385 ms |
| bge-v2-m3 | `BAAI/bge-reranker-v2-m3` | cuda | 18.5 | 554 ms |

Speed is whatever the named device delivered while scoring; it says nothing about
another machine.
