import pytest

from rag_engine.adapters.redis_store import RedisRateLimiter, RedisResponseCache


class FakePipeline:
    def __init__(self, redis: "FakeRedis") -> None:
        self._redis = redis
        self._commands: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    async def __aenter__(self) -> "FakePipeline":
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    def incr(self, name: str) -> None:
        self._commands.append(("incr", (name,), {}))

    def expire(self, name: str, seconds: int, nx: bool = False) -> None:
        self._commands.append(("expire", (name, seconds), {"nx": nx}))

    def ttl(self, name: str) -> None:
        self._commands.append(("ttl", (name,), {}))

    async def execute(self) -> list[int]:
        return [
            getattr(self._redis, name)(*args, **kwargs) for name, args, kwargs in self._commands
        ]


class FakeRedis:
    """The few Redis commands the adapter uses, with the semantics that matter to it."""

    def __init__(self) -> None:
        self.counters: dict[str, int] = {}
        self.expiry: dict[str, int] = {}
        self.strings: dict[str, tuple[str, int | None]] = {}

    def pipeline(self, transaction: bool = True) -> FakePipeline:
        assert transaction, "the count and the expiry must be one transaction"
        return FakePipeline(self)

    def incr(self, name: str) -> int:
        self.counters[name] = self.counters.get(name, 0) + 1
        return self.counters[name]

    def expire(self, name: str, seconds: int, nx: bool = False) -> int:
        if nx and name in self.expiry:
            return 0
        self.expiry[name] = seconds
        return 1

    def ttl(self, name: str) -> int:
        return self.expiry.get(name, -1)

    async def get(self, name: str) -> str | None:
        found = self.strings.get(name)
        return found[0] if found else None

    async def set(self, name: str, value: str, ex: int | None = None) -> None:
        self.strings[name] = (value, ex)


async def test_the_window_opens_with_the_first_request_and_counts_the_rest() -> None:
    redis = FakeRedis()
    limiter = RedisRateLimiter(redis)  # type: ignore[arg-type]

    decisions = [await limiter.hit("search:key:alpha", 2, 60) for _ in range(3)]

    assert [d.allowed for d in decisions] == [True, True, False]
    assert [d.remaining for d in decisions] == [1, 0, 0]
    assert decisions[2].retry_after_seconds == 60
    assert redis.expiry == {"rag:rl:search:key:alpha": 60}


async def test_later_requests_do_not_push_the_end_of_the_window_further_away() -> None:
    redis = FakeRedis()
    limiter = RedisRateLimiter(redis)  # type: ignore[arg-type]
    await limiter.hit("k", 5, 60)
    redis.expiry["rag:rl:k"] = 17  # 43 seconds later

    decision = await limiter.hit("k", 5, 60)

    assert decision.retry_after_seconds == 17
    assert redis.expiry["rag:rl:k"] == 17


async def test_keys_are_counted_separately() -> None:
    limiter = RedisRateLimiter(FakeRedis())  # type: ignore[arg-type]
    await limiter.hit("a", 1, 60)

    assert (await limiter.hit("a", 1, 60)).allowed is False
    assert (await limiter.hit("b", 1, 60)).allowed is True


async def test_a_counter_without_an_expiry_still_gets_a_retry_time() -> None:
    redis = FakeRedis()
    limiter = RedisRateLimiter(redis)  # type: ignore[arg-type]
    redis.counters["rag:rl:k"] = 5
    redis.expire = lambda *a, **k: 0  # type: ignore[method-assign]

    decision = await limiter.hit("k", 3, 60)

    assert decision.allowed is False and decision.retry_after_seconds == 60


async def test_the_cache_stores_a_value_with_a_lifetime_and_reads_it_back() -> None:
    redis = FakeRedis()
    cache = RedisResponseCache(redis)  # type: ignore[arg-type]

    assert await cache.get("k") is None
    await cache.set("k", '{"a": 1}', 600)

    assert await cache.get("k") == '{"a": 1}'
    assert redis.strings["rag:cache:k"] == ('{"a": 1}', 600)


@pytest.mark.parametrize("stored", [b"bytes", 7])
async def test_a_value_that_is_not_text_is_returned_as_text(stored: object) -> None:
    class Odd(FakeRedis):
        async def get(self, name: str) -> object:  # type: ignore[override]
            return stored

    cache = RedisResponseCache(Odd())  # type: ignore[arg-type]

    assert isinstance(await cache.get("k"), str)
