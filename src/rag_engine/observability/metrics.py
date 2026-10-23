"""Prometheus metrics. One registry per application, so tests and processes never share counters.

Labels are kept to values that cannot grow without bound: the route template instead of the URL,
the client name from the configured keys instead of anything a caller sends.
"""

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# Request latency spans a millisecond health check to a cold query that retrieves, writes and
# verifies an answer (tens of seconds on a small machine).
_REQUEST_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 40, 80)
_STAGE_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64)

CONTENT_TYPE = CONTENT_TYPE_LATEST


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "rag_http_requests_total",
            "HTTP requests by method, route template and status code",
            ["method", "route", "status"],
            registry=self.registry,
        )
        self.request_seconds = Histogram(
            "rag_http_request_duration_seconds",
            "HTTP request latency by method and route template",
            ["method", "route"],
            buckets=_REQUEST_BUCKETS,
            registry=self.registry,
        )
        self.in_flight = Gauge(
            "rag_http_requests_in_flight",
            "Requests being handled right now",
            registry=self.registry,
        )
        self.query_stage_seconds = Histogram(
            "rag_query_stage_seconds",
            "Time spent in each stage of POST /v1/query",
            ["stage"],
            buckets=_STAGE_BUCKETS,
            registry=self.registry,
        )
        self.queries = Counter(
            "rag_queries_total",
            "Answers by what they say about themselves (answered, partial, not_found, abstained)",
            ["status"],
            registry=self.registry,
        )
        self.query_cost = Counter(
            "rag_query_cost_usd_total",
            "Model spend of answered queries (cached answers cost nothing)",
            registry=self.registry,
        )
        self.query_tokens = Counter(
            "rag_query_tokens_total",
            "Model tokens used by answered queries",
            ["direction"],
            registry=self.registry,
        )
        self.cache = Counter(
            "rag_query_cache_total",
            "Query cache lookups (hit, miss) and Redis failures (error)",
            ["result"],
            registry=self.registry,
        )
        self.rate_limited = Counter(
            "rag_rate_limited_total",
            "Requests refused for exceeding a client's limit",
            ["group", "client"],
            registry=self.registry,
        )
        self.rate_limit_errors = Counter(
            "rag_rate_limit_errors_total",
            "Times the limiter could not reach Redis and let the request through",
            registry=self.registry,
        )
        self.auth_failures = Counter(
            "rag_auth_failures_total",
            "Requests refused for a missing or wrong API key",
            ["reason"],
            registry=self.registry,
        )

    def render(self) -> bytes:
        return generate_latest(self.registry)
