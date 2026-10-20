"""What model calls cost, and a cap on what a run may spend.

Only the standard library is needed to price a call, because the metering proxy imports this
module in a bare Python environment.
"""

from typing import TYPE_CHECKING

from rag_engine.core.errors import BudgetExceededError
from rag_engine.core.models import LLMReply, LLMUsage

if TYPE_CHECKING:
    from rag_engine.core.ports import LLMClient

# US dollars per million tokens (input, output), from Anthropic's published prices for prompts up
# to 100k tokens. Prices change: a model that is not listed is refused rather than guessed.
PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-fable-5-1": (10.00, 50.00),
}


def cost_usd(model: str, usage: LLMUsage) -> float:
    if model not in PRICES:
        raise KeyError(f"no price is known for {model}; add it to PRICES before using it")
    price_in, price_out = PRICES[model]
    return (usage.input_tokens * price_in + usage.output_tokens * price_out) / 1_000_000


class CostMeter:
    """Adds up what the calls cost. With a budget, a call that starts after it is spent is refused.

    Calls already running when the budget runs out still finish, so a run can overshoot by the
    cost of the calls that were in flight.
    """

    def __init__(self, budget_usd: float | None = None) -> None:
        self.budget_usd = budget_usd
        self.spent_usd = 0.0
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def add(self, model: str, usage: LLMUsage) -> None:
        self.spent_usd += cost_usd(model, usage)
        self.calls += 1
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens

    def check(self) -> None:
        if self.budget_usd is not None and self.spent_usd >= self.budget_usd:
            raise BudgetExceededError(
                f"the budget of ${self.budget_usd:.2f} is spent (${self.spent_usd:.2f} so far)"
            )


class MeteredLLM:
    """An `LLMClient` that counts what its calls cost and stops at the budget."""

    def __init__(self, inner: "LLMClient", meter: CostMeter) -> None:
        self.model = inner.model
        self._inner = inner
        self.meter = meter

    async def complete(
        self, system: str, user: str, *, max_tokens: int = 1024, temperature: float | None = None
    ) -> LLMReply:
        self.meter.check()
        reply = await self._inner.complete(
            system, user, max_tokens=max_tokens, temperature=temperature
        )
        self.meter.add(reply.model if reply.model in PRICES else self.model, reply.usage)
        return reply
