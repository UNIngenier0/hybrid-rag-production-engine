"""Request correlation ID and one access-log line per request."""

import logging
import re
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from rag_engine.observability.logging import request_id_var
from rag_engine.observability.metrics import Metrics

REQUEST_ID_HEADER = "X-Request-ID"

# Only well-formed IDs are echoed back and logged; anything else is replaced, which also
# keeps client-controlled text out of the logs.
_VALID_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")

logger = logging.getLogger("rag_engine.access")


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        incoming = request.headers.get(REQUEST_ID_HEADER, "")
        request_id = incoming if _VALID_REQUEST_ID.fullmatch(incoming) else uuid.uuid4().hex
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        metrics: Metrics | None = getattr(request.app.state, "metrics", None)
        if metrics is not None:
            metrics.in_flight.inc()
        # Health checks run every few seconds; keep them out of the INFO stream.
        level = logging.DEBUG if request.url.path.startswith("/health") else logging.INFO
        try:
            try:
                response = await call_next(request)
            except Exception:
                self._record(metrics, request, 500, started)
                logger.exception(
                    "request failed",
                    extra={"http_method": request.method, "http_path": request.url.path},
                )
                raise
            self._record(metrics, request, response.status_code, started)
            response.headers[REQUEST_ID_HEADER] = request_id
            logger.log(
                level,
                "request completed",
                extra={
                    "http_method": request.method,
                    "http_path": request.url.path,
                    "http_status": response.status_code,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                },
            )
            return response
        finally:
            if metrics is not None:
                metrics.in_flight.dec()
            request_id_var.reset(token)

    @staticmethod
    def _record(
        metrics: Metrics | None, request: Request, status_code: int, started: float
    ) -> None:
        if metrics is None:
            return
        # The route template ("/v1/query"), never the raw path, so the labels stay few.
        route = getattr(request.scope.get("route"), "path", "unmatched")
        metrics.requests.labels(request.method, route, str(status_code)).inc()
        metrics.request_seconds.labels(request.method, route).observe(time.perf_counter() - started)
