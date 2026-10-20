"""Checks that a comparison between systems is like for like.

A comparison is only worth reading if the systems were asked the same questions, answered by the
same model, given a comparable amount of evidence, and judged the same way against the same gold
answers. Each of those is checked here from the files a run leaves behind, so a comparison that
breaks one of them cannot be published by accident.

A run folder holds `questions.txt`, `answers.jsonl`, `run.json` (how the answers were produced),
`results-<judge>.json` and `judge-<judge>.json` (how they were scored), plus usage reports.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Systems that answer from a fixed number of passages or files; the iterative agent decides for
# itself how much to read, which is what makes it a different kind of rival.
SINGLE_PASS = ("rag", "grep-oneshot")


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class Audit:
    checks: list[Check]
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def failures(self) -> list[Check]:
        return [check for check in self.checks if not check.passed]

    def markdown(self) -> str:
        lines = ["## Fairness checks", "", "| Check | Result | Detail |", "|---|---|---|"]
        lines += [
            f"| {check.name} | {'pass' if check.passed else '**FAIL**'} | {check.detail} |"
            for check in self.checks
        ]
        if self.notes:
            lines += ["", "Differences that remain by design:", ""]
            lines += [f"- {note}" for note in self.notes]
        return "\n".join(lines) + "\n"


@dataclass(frozen=True, slots=True)
class RunFacts:
    name: str
    questions: list[str]
    answered: list[str]
    scored: list[str]
    manifest: dict[str, Any]
    judge_manifest: dict[str, Any]
    results: dict[str, Any]
    judge_models: set[str]
    answer_models: set[str]
    cost_records: dict[str, str]  # what each run has to show for its cost, or why it has none


def audit_runs(arms: Sequence[tuple[str, Path]], judge: str) -> Audit:
    """Audit the run folders of the systems being compared; the first is the baseline."""
    runs = [_facts(name, folder, judge) for name, folder in arms]
    checks = [
        _same_questions(runs),
        _all_answered(runs),
        _all_scored(runs),
        _same_gold(runs),
        _same_judge(runs, judge),
        _same_model(runs),
        _same_evidence(runs),
        _same_documents(runs),
        _costs_measured(runs),
    ]
    return Audit(checks, _notes(runs))


def write_manifest(run_dir: Path, manifest: Mapping[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _facts(name: str, folder: Path, judge: str) -> RunFacts:
    return RunFacts(
        name=name,
        questions=_lines(folder / "questions.txt"),
        answered=_ids(folder / "answers.jsonl"),
        scored=_scored(folder / f"results-{judge}.json"),
        manifest=_json(folder / "run.json"),
        judge_manifest=_json(folder / f"judge-{judge}.json"),
        results=_json(folder / f"results-{judge}.json").get("aggregate_stats", {}),
        judge_models=_usage_models(folder / f"cost-judge-{judge}.json"),
        answer_models=_answer_models(folder),
        cost_records=_cost_records(folder),
    )


def _same_questions(runs: Sequence[RunFacts]) -> Check:
    sets = {run.name: set(run.questions) for run in runs}
    first = next(iter(sets.values()))
    same = all(ids == first for ids in sets.values()) and all(
        len(run.questions) == len(set(run.questions)) for run in runs
    )
    return Check("same questions", same, f"{len(first)} questions" if same else _sizes(sets))


def _all_answered(runs: Sequence[RunFacts]) -> Check:
    missing = {run.name: sorted(set(run.questions) - set(run.answered)) for run in runs}
    bad = {name: ids for name, ids in missing.items() if ids}
    return Check(
        "every question answered",
        not bad,
        "all answered" if not bad else "; ".join(f"{n}: {len(i)} missing" for n, i in bad.items()),
    )


def _all_scored(runs: Sequence[RunFacts]) -> Check:
    missing = {run.name: sorted(set(run.questions) - set(run.scored)) for run in runs}
    bad = {name: ids for name, ids in missing.items() if ids}
    skipped = {run.name: run.results.get("skipped_rows", 0) for run in runs}
    problems = [f"{n}: {len(i)} unscored" for n, i in bad.items()]
    problems += [f"{n}: {s} skipped rows" for n, s in skipped.items() if s]
    return Check(
        "every answer scored", not problems, "all scored" if not problems else "; ".join(problems)
    )


def _same_gold(runs: Sequence[RunFacts]) -> Check:
    corrected = {run.name: run.results.get("num_corrected_questions") for run in runs}
    bad = {name: count for name, count in corrected.items() if count != 0}
    return Check(
        "same gold answers for everyone",
        not bad,
        "judged against the original gold set (no correction)"
        if not bad
        else f"the evaluator rewrote the gold set: {bad}",
    )


def _same_judge(runs: Sequence[RunFacts], label: str) -> Check:
    models = {run.name: run.judge_manifest.get("model") for run in runs}
    problems = []
    if None in models.values():
        problems.append("a run has no record of its judge")
    elif len(set(models.values())) != 1:
        problems.append(f"judged by different models: {models}")
    for run in runs:
        declared = models[run.name]
        if declared is not None and run.judge_models and run.judge_models != {declared}:
            problems.append(
                f"{run.name} declared {declared} but its usage shows {sorted(run.judge_models)}"
            )
    detail = (
        f"{next(iter(models.values()))} for every system ({label})"
        if not problems
        else "; ".join(problems)
    )
    return Check("same judge", not problems, detail)


def _same_model(runs: Sequence[RunFacts]) -> Check:
    models = {run.name: run.manifest.get("model") for run in runs}
    used = {run.name: run.answer_models for run in runs}
    problems = []
    if None in models.values():
        problems.append("a run has no run.json")
    elif len(set(models.values())) != 1:
        problems.append(f"declared models differ: {models}")
    for name, seen in used.items():
        declared = models[name]
        if declared is not None and seen and seen != {declared}:
            problems.append(f"{name} declared {declared} but its usage shows {sorted(seen)}")
    detail = (
        f"{next(iter(models.values()))} for every system" if not problems else "; ".join(problems)
    )
    return Check("same answering model", not problems, detail)


def _same_evidence(runs: Sequence[RunFacts]) -> Check:
    single = [run for run in runs if run.manifest.get("system") in SINGLE_PASS]
    budgets = {run.name: run.manifest.get("passages") for run in single}
    same = len(set(budgets.values())) <= 1 and None not in budgets.values()
    detail = (
        f"{next(iter(budgets.values()))} passages or files per question for " + ", ".join(budgets)
        if same and budgets
        else f"passages differ: {budgets}"
    )
    return Check("same evidence budget for single-pass systems", same, detail)


def _same_documents(runs: Sequence[RunFacts]) -> Check:
    policies = {
        run.name: run.manifest.get("documents")
        for run in runs
        if run.manifest.get("system") in SINGLE_PASS
    }
    same = len(set(policies.values())) <= 1 and None not in policies.values()
    detail = (
        f"documents reported = {next(iter(policies.values()))} for " + ", ".join(policies)
        if same and policies
        else f"document policies differ: {policies}"
    )
    return Check("same document-reporting rule", same, detail)


def _costs_measured(runs: Sequence[RunFacts]) -> Check:
    """A cost that is not measured must not be published as zero: a metering report with no calls
    next to answers that exist means the proxy did not see the traffic."""
    problems = [
        f"{run.name}: {run.cost_records['problem']}"
        for run in runs
        if "problem" in run.cost_records
    ]
    return Check(
        "cost measured for every system",
        not problems,
        "every system has a usage report or token counts" if not problems else "; ".join(problems),
    )


def _notes(runs: Sequence[RunFacts]) -> list[str]:
    notes = []
    for run in runs:
        if run.manifest.get("system") not in SINGLE_PASS:
            notes.append(
                f"{run.name} decides for itself how much to read and search (cost and time are "
                "reported next to its score)."
            )
        if run.judge_manifest.get("strip_citations"):
            notes.append(
                f"{run.name}: answers end with a source list, so the evaluator removes citations "
                "before judging, as the benchmark does by default."
            )
    return notes


def _sizes(sets: Mapping[str, set[str]]) -> str:
    return "; ".join(f"{name}: {len(ids)}" for name, ids in sets.items())


def _lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _ids(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [
        json.loads(line)["question_id"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _scored(path: Path) -> list[str]:
    return [row["question_id"] for row in _json(path).get("questions", [])]


def _json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _cost_records(folder: Path) -> dict[str, str]:
    for report in ("cost-agent.json", "cost-grep.json"):
        data = _json(folder / report)
        if data:
            if not data.get("calls"):
                return {"problem": f"{report} shows no calls although the system answered"}
            return {"source": report}
    trace = folder / "trace.jsonl"
    if trace.exists() and '"input_tokens"' in trace.read_text(encoding="utf-8"):
        return {"source": "trace.jsonl"}
    return {"problem": "no usage report and no token counts"}


def _usage_models(path: Path) -> set[str]:
    return set(_json(path).get("by_model", {}))


def _answer_models(folder: Path) -> set[str]:
    """The models that produced a run's answers, from its usage reports or its trace."""
    models: set[str] = set()
    for report in ("cost-agent.json", "cost-grep.json"):
        models |= _usage_models(folder / report)
    trace = folder / "trace.jsonl"
    if trace.exists():
        for line in trace.read_text(encoding="utf-8").splitlines():
            row = json.loads(line) if line.strip() else {}
            if "model" in row:
                models.add(row["model"])
    return models
