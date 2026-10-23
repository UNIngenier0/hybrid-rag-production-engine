"""API keys: who is calling, and the limit that applies to them.

Keys are configured as `name:secret` pairs. The name identifies the client in logs, limits and
metrics; the secret is only compared, in constant time, and is never logged or put in a label.
"""

import hmac
import logging
import re
from dataclasses import dataclass
from typing import Annotated, Literal

from fastapi import Depends, HTTPException, Request, Response, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from rag_engine.core.config import Settings
from rag_engine.observability.metrics import Metrics

logger = logging.getLogger("rag_engine.security")

_NAME = re.compile(r"[A-Za-z0-9._-]{1,40}")
MIN_KEY_LENGTH = 16
ANONYMOUS = "anonymous"

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer = HTTPBearer(auto_error=False)


class ApiKeys:
    def __init__(self, keys: dict[str, str]) -> None:
        self._keys = keys

    @classmethod
    def parse(cls, text: str) -> "ApiKeys":
        """Read `name:secret,name:secret`. Raises ValueError for anything malformed or weak, so a
        mistake in the configuration stops the service at startup instead of leaving it open."""
        keys: dict[str, str] = {}
        for item in (part.strip() for part in text.split(",")):
            if not item:
                continue
            name, separator, secret = item.partition(":")
            if not separator or not _NAME.fullmatch(name):
                raise ValueError(
                    "API_KEYS entries must look like name:secret (name: letters, digits, . _ -)"
                )
            if len(secret) < MIN_KEY_LENGTH:
                raise ValueError(
                    f"the API key of {name!r} is shorter than {MIN_KEY_LENGTH} characters"
                )
            if name in keys or secret in keys.values():
                raise ValueError(f"API_KEYS repeats the name or the secret of {name!r}")
            keys[name] = secret
        if not keys:
            raise ValueError("API_KEYS has no key in it")
        return cls(keys)

    def authenticate(self, presented: str) -> str | None:
        """The name of the client that owns the key, or None. Compares against every key, so the
        time taken does not say which one was close."""
        found: str | None = None
        for name, secret in self._keys.items():
            if hmac.compare_digest(presented.encode(), secret.encode()):
                found = name
        return found

    def __len__(self) -> int:
        return len(self._keys)


def build_api_keys(settings: Settings) -> ApiKeys | None:
    if settings.api_keys is None:
        return None
    return ApiKeys.parse(settings.api_keys.get_secret_value())


@dataclass(frozen=True, slots=True)
class Client:
    name: str  # a configured key's name, or "anonymous" when the API is open
    limit_key: str  # what its requests are counted under


def _address(request: Request) -> str:
    # The socket peer, not X-Forwarded-For: a header the caller writes must not pick the bucket.
    return request.client.host if request.client else "unknown"


async def require_client(
    request: Request,
    header: Annotated[str | None, Depends(api_key_header)],
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
) -> Client:
    """The caller, or 401. With no keys configured everyone is anonymous."""
    keys: ApiKeys | None = request.app.state.api_keys
    if keys is None:
        return Client(ANONYMOUS, f"ip:{_address(request)}")
    metrics: Metrics = request.app.state.metrics
    presented = header or (credentials.credentials if credentials else None)
    if not presented:
        metrics.auth_failures.labels("missing").inc()
        raise _unauthorized("an API key is required: send it in the X-API-Key header")
    name = keys.authenticate(presented)
    if name is None:
        metrics.auth_failures.labels("invalid").inc()
        logger.warning("request with an invalid API key", extra={"http_path": request.url.path})
        raise _unauthorized("the API key is not valid")
    return Client(name, f"key:{name}")


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status.HTTP_401_UNAUTHORIZED, detail, headers={"WWW-Authenticate": "Bearer"}
    )


Group = Literal["search", "query"]


def rate_limited(group: Group):  # type: ignore[no-untyped-def]
    """A dependency that counts the request against the client's limit for `group`.

    If Redis cannot be reached the request goes through and the failure is counted: refusing every
    caller because the counter is down would turn a cache outage into an API outage.
    """

    async def dependency(
        request: Request, response: Response, client: Annotated[Client, Depends(require_client)]
    ) -> None:
        settings: Settings = request.app.state.settings
        limit = (
            settings.rate_limit_search_per_minute
            if group == "search"
            else settings.rate_limit_query_per_minute
        )
        limiter = request.app.state.rate_limiter
        if limit == 0 or limiter is None:
            return
        metrics: Metrics = request.app.state.metrics
        try:
            decision = await limiter.hit(f"{group}:{client.limit_key}", limit, 60)
        except Exception:
            metrics.rate_limit_errors.inc()
            logger.warning("rate limiter unavailable; letting the request through", exc_info=True)
            return
        headers = {
            "X-RateLimit-Limit": str(decision.limit),
            "X-RateLimit-Remaining": str(decision.remaining),
        }
        if not decision.allowed:
            metrics.rate_limited.labels(group, client.name).inc()
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS,
                f"rate limit of {limit} {group} requests per minute exceeded",
                headers={**headers, "Retry-After": str(decision.retry_after_seconds)},
            )
        response.headers.update(headers)

    return dependency
