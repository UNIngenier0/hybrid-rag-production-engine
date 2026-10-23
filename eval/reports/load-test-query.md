# Load test

aarch64 Linux, client on the same host. Server settings that matter: limits and cache off, 6 GB Docker VM, MiniLM-L6 reranker, Claude Haiku 5.5, citation judge on.
Every request asks a different benchmark question.

| Endpoint | Options | Concurrency | Requests | OK | Other statuses | Requests/s | Mean | p50 | p95 | p99 | Max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| query | mode=hybrid+rerank, verify=True | 1 | 8 | 8 | - | 0.036 | 27.945 s | 23.972 s | 40.226 s | 40.226 s | 40.226 s |
| query | mode=hybrid+rerank, verify=True | 2 | 8 | 8 | - | 0.048 | 40.867 s | 38.443 s | 51.44 s | 51.44 s | 51.44 s |
