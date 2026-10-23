"""Redis-backed rate limiting and response caching, shared by every API process."""

from redis.asyncio import Redis

from rag_engine.core.models import RateDecision

# A request that waits longer than this for Redis is not worth waiting for: the callers carry on
# without the limit or the cache instead of slowing every request down.
SOCKET_TIMEOUT_SECONDS = 1.0


def connect(url: str) -> Redis:
    """A client that connects when first used, so building one never fails or blocks."""
    return Redis.from_url(
        url,
        decode_responses=True,
        socket_timeout=SOCKET_TIMEOUT_SECONDS,
        socket_connect_timeout=SOCKET_TIMEOUT_SECONDS,
    )


class RedisRateLimiter:
    """A fixed window per key: the first request opens a window of `window_seconds`, and the
    requests in it are counted. A client can send up to twice the limit across the boundary of
    two windows; for protecting a service that is an acceptable price for one counter per client."""

    def __init__(self, redis: Redis, prefix: str = "rag:rl") -> None:
        self._redis = redis
        self._prefix = prefix

    async def hit(self, key: str, limit: int, window_seconds: int) -> RateDecision:
        name = f"{self._prefix}:{key}"
        # One transaction, so a crash between the count and the expiry cannot leave a counter that
        # never resets. NX sets the expiry only on the request that opened the window.
        async with self._redis.pipeline(transaction=True) as pipeline:
            pipeline.incr(name)
            pipeline.expire(name, window_seconds, nx=True)
            pipeline.ttl(name)
            count, _, ttl = await pipeline.execute()
        return RateDecision(
            allowed=count <= limit,
            limit=limit,
            remaining=max(0, limit - count),
            retry_after_seconds=ttl if ttl > 0 else window_seconds,
        )


class RedisResponseCache:
    def __init__(self, redis: Redis, prefix: str = "rag:cache") -> None:
        self._redis = redis
        self._prefix = prefix

    async def get(self, key: str) -> str | None:
        value = await self._redis.get(f"{self._prefix}:{key}")
        return None if value is None else str(value)

    async def set(self, key: str, value: str, ttl_seconds: int) -> None:
        await self._redis.set(f"{self._prefix}:{key}", value, ex=ttl_seconds)
