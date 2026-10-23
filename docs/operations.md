# Running the service: access control, limits, cache, metrics and load

This page covers what stands between the API and the people who call it: who may call, how much,
how repeated questions are served, what the service reports about itself, and how much load it
carried when it was measured. All the settings are environment variables (see `.env.example`); the
defaults suit a laptop, and the first thing to change anywhere else is `API_KEYS`.

| Setting | Default | Meaning |
|---|---|---|
| `API_KEYS` | none (open) | `name:secret` pairs, comma separated; the secret has at least 16 characters |
| `METRICS_TOKEN` | none | When set, `/metrics` needs `Authorization: Bearer <token>` |
| `RATE_LIMIT_SEARCH_PER_MINUTE` | 120 | Requests per client per minute to `/v1/search`; 0 turns it off |
| `RATE_LIMIT_QUERY_PER_MINUTE` | 20 | The same for `/v1/query`, lower because each answer costs model calls |
| `QUERY_CACHE_TTL_SECONDS` | 3600 | How long an identical question is answered from Redis; 0 turns the cache off |
| `QUERY_CACHE_NAMESPACE` | `v1` | Change it to drop every cached answer (after re-indexing) |

## Access control

`/v1/search` and `/v1/query` ask for a key when `API_KEYS` is set:

```bash
export API_KEYS="demo:$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
curl -s localhost:8000/v1/search -H "X-API-Key: <secret>" -H 'content-type: application/json' \
     -d '{"query": "upload limit"}'
```

The key goes in the `X-API-Key` header, or as `Authorization: Bearer <secret>`. A request without
one, or with a wrong one, gets `401` with `WWW-Authenticate: Bearer`.

- **The name is the identity.** It appears in logs, in the limit counters and as a metric label; the
  secret is only compared, in constant time, and is never logged or used as a label. A rejected key
  is not written to the logs.
- **Mistakes stop the service at startup.** A malformed list, a secret shorter than 16 characters or
  a repeated name or secret makes the application refuse to start, instead of running open by
  accident.
- **Health checks and metrics stay outside the keys.** `/health/live` and `/health/ready` have to
  answer an orchestrator that holds no key. `/metrics` is open unless `METRICS_TOKEN` is set; if
  you do not set a token, keep the port on a private network.
- **With no `API_KEYS` the API is open**, and the service says so in its log at startup. That is
  the development default and is not meant for anything reachable from outside.
- **Rotation** is adding the new `name2:secret2` next to the old key, moving the clients, and then
  removing the old one; the service reads the list at startup, so each step is a restart.

## Rate limits

Each client has two counters, one for search and one for answers, kept in Redis so that every API
process sees the same numbers. The counter is a fixed window: the first request opens a window of a
minute and the requests in it are counted.

- A request over the limit gets `429` with `Retry-After` (the seconds until the window ends).
  Every response that was counted carries `X-RateLimit-Limit` and `X-RateLimit-Remaining`.
- **A client is the key's name.** Without keys, the client is the address of the socket. The
  `X-Forwarded-For` header is deliberately not trusted: a caller must not be able to pick its own
  bucket. Behind a reverse proxy that means everyone shares the proxy's address; run keys there.
- **A fixed window allows a burst of up to twice the limit across the boundary of two windows.**
  For protecting a service that is an acceptable price for one counter per client.
- **If Redis cannot be reached, requests go through** and `rag_rate_limit_errors_total` counts the
  failure. Refusing every caller because a counter is down would turn a cache outage into an API
  outage; the cost is that the limit is not enforced while Redis is down, which is why that
  counter is worth an alert.

## Answer cache

An identical `/v1/query` is answered from Redis for `QUERY_CACHE_TTL_SECONDS`, with
`"cached": true`, no model call, zero `usage` and a `timings_ms` that only holds the lookup.

- **What counts as identical.** The same question (whitespace ignored), mode, number of passages,
  filters and verification, under the same models, reranker and abstention threshold. Changing the
  answering model, the judge, the reranker or `ABSTAIN_THRESHOLD` therefore changes every key.
- **What is not stored.** Answers whose citation check could not run (the judge failed) are not
  kept, so asking again gets them checked.
- **A client can ask for a fresh answer** with `"cache": false`, which neither reads nor writes the cache.
- **After re-indexing**, change `QUERY_CACHE_NAMESPACE`: an answer written from the old index
  would otherwise be served for up to an hour.
- **If Redis is down**, the question is answered by the pipeline as if there were no cache.
- The cache holds answers for anyone who asks the same thing: it is shared across clients on
  purpose (that is where the saving is), so do not enable it if answers must differ by caller.

## Metrics

`GET /metrics` serves the Prometheus text format. Labels are bounded on purpose: the route is the
template (`/v1/query`), never the raw path, so a scanner cannot grow the series, and the client is a
configured key name.

| Metric | What it tells you |
|---|---|
| `rag_http_requests_total{method,route,status}` | Traffic and error rate per route |
| `rag_http_request_duration_seconds{method,route}` | Latency histogram, 10 ms to 80 s |
| `rag_http_requests_in_flight` | Requests being handled now |
| `rag_query_stage_seconds{stage}` | Time in each stage of an answer: `retrieval_*`, `generation`, `verification`, `total` |
| `rag_queries_total{status}` | Answers by `answered`, `partial`, `not_found`, `abstained` |
| `rag_query_cost_usd_total`, `rag_query_tokens_total{direction}` | Model spend and tokens (cached answers add nothing) |
| `rag_query_cache_total{result}` | `hit`, `miss`, and `error` when Redis failed |
| `rag_rate_limited_total{group,client}` | Refusals by limit and client |
| `rag_rate_limit_errors_total` | Requests let through because Redis was unreachable |
| `rag_auth_failures_total{reason}` | `missing` or `invalid` keys |

A scrape configuration and three queries worth having on a dashboard:

```yaml
scrape_configs:
  - job_name: rag
    metrics_path: /metrics
    authorization: {credentials: "<METRICS_TOKEN>"}   # only if the token is set
    static_configs: [{targets: ["api:8000"]}]
```

```promql
histogram_quantile(0.95, sum by (le) (rate(rag_http_request_duration_seconds_bucket{route="/v1/query"}[5m])))
sum by (status) (rate(rag_http_requests_total{route="/v1/query"}[5m]))
sum(increase(rag_query_cost_usd_total[1d]))
```

Counters live in the process: they reset when it restarts, which Prometheus handles, and with
several processes each exposes its own and Prometheus adds them up. Request IDs (`X-Request-ID`)
and one JSON access-log line per request were already in place; there is no distributed tracing.

## Load test

`scripts/loadtest.py` drives the running API with a different benchmark question in every request,
so neither the answer cache nor warm disk pages flatter a run, and reports requests per second and
the latency percentiles of the successful ones. Turn the limits off on the server while measuring
(`RATE_LIMIT_SEARCH_PER_MINUTE=0 RATE_LIMIT_QUERY_PER_MINUTE=0 QUERY_CACHE_TTL_SECONDS=0`), or the
limiter does the refusing.

```bash
uv run python scripts/loadtest.py --endpoint search --mode hybrid --requests 12 --concurrency 1 2 4
uv run python scripts/loadtest.py --endpoint query --verify --requests 8 --concurrency 1 2
```

**Results on the development laptop** (Apple silicon with the 6 GB Docker VM, the full 3.06 million
chunk index mostly on disk, embeddings and the MiniLM-L6 reranker on the CPU, the client on the same
machine; [search](../eval/reports/load-test-search.md), [answers](../eval/reports/load-test-query.md)):

| Endpoint | Concurrency | Requests | OK | Requests/s | Mean | p50 | p95 |
|---|---|---|---|---|---|---|---|
| `/v1/search`, hybrid | 1 | 12 | 12 | 0.081 | 12.4 s | 10.3 s | 18.0 s |
| `/v1/search`, hybrid | 2 | 12 | 12 | 0.107 | 17.6 s | 14.6 s | 25.5 s |
| `/v1/search`, hybrid | 4 | 12 | 12 | 0.124 | 32.3 s | 29.3 s | 44.2 s |
| `/v1/query`, reranked, judge on | 1 | 8 | 8 | 0.036 | 27.9 s | 24.0 s | 40.2 s |
| `/v1/query`, reranked, judge on | 2 | 8 | 8 | 0.048 | 40.9 s | 38.4 s | 51.4 s |

What the numbers say:

- **The machine is the limit, not the request handling.** Throughput stops growing near 0.1 to 0.12
  searches per second and latency grows in step with concurrency (four requests at once each take
  about four times as long as one alone). The service is CPU- and disk-bound here, so it does not
  turn extra concurrency into extra work. This laptop cannot meet the sub-second retrieval target;
  that was already measured in [retrieval.md](retrieval.md#latency-on-the-development-laptop), and
  these figures say what it costs when requests overlap.
- **Where an answer's time goes** (the mean of the 18 answers, read from `rag_query_stage_seconds`):
  33.7 s in total, of which retrieval 25.9 s (query embedding 6.7 s, dense search 10.8 s and BM25
  search 6.8 s in parallel, CPU reranking 8.3 s), writing the answer 4.0 s and checking its
  citations 3.7 s. Retrieval is three quarters of it. On a GPU the cross-encoder takes tens of
  milliseconds for the 30 passages instead of 8 s ([evaluation.md](evaluation.md#reranker-cost)), and
  a machine with enough memory for the index would take most of the search time away; neither is
  measured here.
- **An answer costs $0.0016 on average** ($0.029 for the 18, from `rag_query_cost_usd_total`), and
  a repeated question costs nothing.
- **Found by this test: a slow Qdrant returned `500`.** In the first run of the search test, five
  of the twelve requests at concurrency 4 outlasted Qdrant's 30-second deadline, and the API
  answered `500` with a stack trace instead of saying retrieval was unavailable. The adapters now
  turn a Qdrant or PostgreSQL timeout or connection failure into `503 {"detail": "retrieval
  unavailable"}` (covered by tests), and the table above is the run made after that. The first
  run's twelve at concurrency 4 had finished in 99 s with 5 errors; the second, with warmer disk
  caches, finished all twelve: cold-versus-warm differences on this machine are large, so read
  these as the scale of the problem, not as a benchmark.

## What this does not do

- **No TLS.** Terminate it in front of the API (a reverse proxy or the platform's load balancer);
  keys sent over plain HTTP can be read.
- **One limit per group for every client.** There are no per-key limits or quotas, and no usage
  records per client beyond the metrics.
- **Keys are read at startup** from the environment; there is no endpoint to create or revoke them
  and no storage of hashed keys.
- **No tracing and no alert rules** are shipped; the metrics are there to build them from.
