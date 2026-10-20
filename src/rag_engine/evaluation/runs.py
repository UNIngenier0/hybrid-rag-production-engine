"""What a run folder holds, read back."""

import json
from pathlib import Path

from rag_engine.core.models import LLMUsage
from rag_engine.generation.cost import cost_usd


def answering_cost(run_dir: Path) -> float | None:
    """What producing the run's answers cost, judge not included.

    Systems that ran through the metering proxy left a usage report; the others left a trace of
    the tokens of each answer, which is priced here.
    """
    for report in ("cost-agent.json", "cost-grep.json"):
        path = run_dir / report
        if path.exists():
            return float(json.loads(path.read_text(encoding="utf-8"))["spent_usd"])
    trace = run_dir / "trace.jsonl"
    if not trace.exists():
        return None
    total = 0.0
    for line in trace.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if "input_tokens" in row:
            total += cost_usd(row["model"], LLMUsage(row["input_tokens"], row["output_tokens"]))
    return total
