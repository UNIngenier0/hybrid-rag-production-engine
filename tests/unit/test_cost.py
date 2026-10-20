import pytest

from fakes import FakeLLM
from rag_engine.core.errors import BudgetExceededError
from rag_engine.core.models import LLMUsage
from rag_engine.generation.cost import CostMeter, MeteredLLM, cost_usd


def test_cost_follows_the_per_million_token_prices() -> None:
    usage = LLMUsage(input_tokens=2_000_000, output_tokens=1_000_000)

    assert cost_usd("claude-haiku-5-5", usage) == pytest.approx(0.2 + 0.5)
    assert cost_usd("claude-sonnet-5-5", usage) == pytest.approx(4.0 + 10.0)


def test_a_model_without_a_known_price_is_refused() -> None:
    with pytest.raises(KeyError, match="no price"):
        cost_usd("some-new-model", LLMUsage(1, 1))


def test_the_meter_adds_up_calls_and_tokens() -> None:
    meter = CostMeter()
    meter.add("claude-haiku-5-5", LLMUsage(1000, 100))
    meter.add("claude-haiku-5-5", LLMUsage(3000, 300))

    assert (meter.calls, meter.input_tokens, meter.output_tokens) == (2, 4000, 400)
    assert meter.spent_usd == pytest.approx(0.0004 + 0.0002)


def test_without_a_budget_nothing_is_ever_refused() -> None:
    meter = CostMeter()
    meter.add("claude-fable-5-1", LLMUsage(10_000_000, 1_000_000))

    meter.check()


def test_a_spent_budget_refuses_the_next_call() -> None:
    meter = CostMeter(budget_usd=0.001)
    meter.add("claude-haiku-5-5", LLMUsage(100_000, 10_000))  # $0.015

    with pytest.raises(BudgetExceededError, match="budget"):
        meter.check()


async def test_the_metered_model_counts_each_call_and_stops_at_the_budget() -> None:
    meter = CostMeter(budget_usd=0.0003)
    llm = MeteredLLM(FakeLLM("a", usage=LLMUsage(2000, 200)), meter)  # $0.0003 a call

    await llm.complete("s", "u")

    assert meter.calls == 1
    with pytest.raises(BudgetExceededError):
        await llm.complete("s", "u")
    assert meter.calls == 1
