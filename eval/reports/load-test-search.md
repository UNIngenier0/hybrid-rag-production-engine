# Load test

aarch64 Linux, client on the same host. Server settings that matter: limits and cache off, 6 GB Docker VM, MiniLM-L6 reranker.
Every request asks a different benchmark question.

| Endpoint | Options | Concurrency | Requests | OK | Other statuses | Requests/s | Mean | p50 | p95 | p99 | Max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| search | mode=hybrid, limit=10 | 1 | 12 | 12 | - | 0.081 | 12.387 s | 10.275 s | 18.0 s | 18.0 s | 18.0 s |
| search | mode=hybrid, limit=10 | 2 | 12 | 12 | - | 0.107 | 17.594 s | 14.601 s | 25.483 s | 25.483 s | 25.483 s |
| search | mode=hybrid, limit=10 | 4 | 12 | 12 | - | 0.124 | 32.269 s | 29.302 s | 44.224 s | 44.224 s | 44.224 s |
