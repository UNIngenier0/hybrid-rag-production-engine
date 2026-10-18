"""Interfaces that the rest of the code depends on. Adapters implement them."""

from typing import Protocol


class HealthProbe(Protocol):
    """Checks whether one external dependency is ready to serve requests."""

    name: str

    async def check(self) -> None:
        """Return normally when the dependency is ready, raise any exception when it is not."""
        ...
