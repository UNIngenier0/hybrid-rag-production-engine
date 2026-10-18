"""Health probes for the data services (Qdrant, PostgreSQL, Redis)."""

import asyncpg
import httpx
from redis.asyncio import Redis


class QdrantProbe:
    name = "qdrant"

    def __init__(self, url: str, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._url = url
        self._transport = transport

    async def check(self) -> None:
        async with httpx.AsyncClient(base_url=self._url, transport=self._transport) as client:
            response = await client.get("/readyz")
            response.raise_for_status()


class PostgresProbe:
    name = "postgres"

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    async def check(self) -> None:
        connection = await asyncpg.connect(self._dsn)
        try:
            await connection.execute("SELECT 1")
        finally:
            await connection.close()


class RedisProbe:
    name = "redis"

    def __init__(self, url: str) -> None:
        self._url = url

    async def check(self) -> None:
        client = Redis.from_url(self._url)
        try:
            await client.ping()
        finally:
            await client.aclose()
