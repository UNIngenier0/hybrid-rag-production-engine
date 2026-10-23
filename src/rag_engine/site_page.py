"""The project's static site, for GitHub Pages.

GitHub Pages serves files and runs no code on a server, and the real service needs the 35 GB index
and a model key, so a public copy of the demo cannot be live. The site therefore replays responses
that were recorded from the running service (`site/examples.json`) and the retrieval trace of one
of those questions (`site/trace.json`), and says so on the page. Nothing on it calls any server.

The page is written as ordinary front-end files in `site/` (an HTML template, one stylesheet and
one script) and assembled here with the published reports, so every number on it is the
repository's. The drawing code of the answer cards is shared with the live `/demo` page.

The content security policy is a `<meta>` tag, because Pages cannot send headers. It allows the
page's one script and one style by hash, so nothing else can run on it even if some text on the page
were hostile, and the hashes are computed from the exact text that is inlined.
"""

import base64
import hashlib
import html
import json
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

from rag_engine.api.demo_page import RENDER_JS

SITE_DIR = Path(__file__).resolve().parents[2] / "site"

CORPUS_DOCUMENTS = 511_962
# The corpus table of the README: documents per source of EnterpriseRAG-Bench.
CORPUS_SOURCES = [
    ("Slack", 285_605),
    ("Gmail", 121_390),
    ("Linear", 35_308),
    ("Google Drive", 25_108),
    ("HubSpot", 15_017),
    ("Fireflies", 10_173),
    ("GitHub", 8_052),
    ("Jira", 6_120),
    ("Confluence", 5_189),
]

# What to look at in each recorded answer. These describe the question, not the answer's
# correctness, which only the benchmark's judge can say.
NOTES = {
    "qst_0411": (
        "Two sources disagree. The answer gives the current value and says what it replaced."
    ),
    "qst_0223": "The question describes a symptom without naming the document that explains it.",
    "qst_0001": "A single-fact question: the kind this system does best on.",
    "qst_0483": (
        "The documents do not hold this information. Look at what the answer says they do state."
    ),
    "qst_0471": "A broad question that no single document answers.",
    "qst_0341": "A question that spans several documents: the kind this system does worst on.",
}

# The benchmark's question types, in words.
TYPE_LABELS = {
    "basic": "Basic",
    "semantic": "Semantic",
    "constrained": "Constrained",
    "conflicting_info": "Conflicting sources",
    "completeness": "Completeness",
    "high_level": "High level",
    "info_not_found": "Not in the documents",
    "intra_document_reasoning": "Reasoning in one document",
    "miscellaneous": "Miscellaneous",
    "project_related": "Project related",
}

# The rows of the retrieval chart: the ablation report's system name for each.
RECALL_SYSTEMS = (
    ("Dense only", "dense", "dense"),
    ("BM25 only", "bm25", "bm25"),
    ("Fused", "hybrid, BM25 weight 1.5", "hybrid"),
    ("Fused + rerank", "rerank bge-v2-m3, pool 30", "rerank"),
)
OURS = "Hybrid + rerank (ours)"

DESCRIPTION = (
    "Hybrid dense + BM25 retrieval, cross-encoder reranking and statement-by-statement verified "
    "citations over 511,962 documents, measured on EnterpriseRAG-Bench."
)


def _hash(text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return "'sha256-" + base64.b64encode(digest).decode("ascii") + "'"


def _json_for_script(data: Any) -> str:
    """JSON that is safe inside a `<script>` data block: no `<`, so no `</script>` or `<!--`."""
    return json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")


def _esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def _one_decimal(value: float) -> str:
    """Rounded half up, as a reader would: 25.65 is 25.7 (a float format would say 25.6)."""
    return str(Decimal(str(value)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def _bar(label: str, detail: str, value: float, shown: str, tone: str) -> str:
    """One horizontal bar; `value` is its length in percent of the track."""
    return (
        f'<div class="hbar" data-tone="{tone}"><span class="hbar-label">{_esc(label)}'
        f"<small>{_esc(detail)}</small></span>"
        f'<span class="hbar-track"><span class="hbar-fill" data-w="{value:.2f}"></span></span>'
        f'<span class="hbar-val">{_esc(shown)}</span></div>'
    )


def _tone(combined: float) -> str:
    return "good" if combined >= 60 else "mid" if combined >= 40 else "weak"


def _type_bars(report: dict[str, Any]) -> str:
    rows = sorted(report["by_type"].items(), key=lambda item: -item[1]["combined"])
    return "".join(
        _bar(
            TYPE_LABELS.get(name, name),
            f"{summary['questions']} questions",
            summary["combined"],
            _one_decimal(summary["combined"]),
            _tone(summary["combined"]),
        )
        for name, summary in rows
    )


def _results_table(report: dict[str, Any]) -> str:
    rows = sorted(report["by_type"].items(), key=lambda item: -item[1]["combined"])
    body = "".join(
        f'<tr><td>{_esc(name)}</td><td class="num">{summary["questions"]}</td>'
        f'<td class="num">{summary["correct_pct"]}%</td>'
        f'<td class="num">{summary["combined"]}</td></tr>'
        for name, summary in rows
    )
    overall = report["overall"]
    return (
        '<table><thead><tr><th>Question type</th><th class="num">Questions</th>'
        '<th class="num">Correct</th><th class="num">Combined</th></tr></thead>'
        f"<tbody>{body}</tbody><tfoot><tr><th>All questions</th>"
        f'<th class="num">{overall["questions"]}</th><th class="num">{overall["correct_pct"]}%</th>'
        f'<th class="num">{overall["combined"]}</th></tr></tfoot></table>'
    )


def _recall(ablation: dict[str, Any], system: str) -> float:
    value: float = ablation["systems"][system]["5"]["overall"]["recall"]
    return value


def _recall_bars(ablation: dict[str, Any]) -> str:
    return "".join(
        _bar(
            label,
            "recall at 5",
            100 * _recall(ablation, system),
            f"{_recall(ablation, system):.3f}",
            tone,
        )
        for label, system, tone in RECALL_SYSTEMS
    )


def _rival_bars(comparison: dict[str, Any]) -> str:
    arms = comparison["arms"]
    bars = []
    for name, arm in sorted(arms.items(), key=lambda item: -item[1]["combined"]):
        combined = 100 * arm["combined"]
        bars.append(
            _bar(
                name.replace(" (ours)", ""),
                f"answers cost ${arm['cost_usd']:.2f}",
                combined,
                _one_decimal(round(combined, 4)),
                "ours" if name == OURS else "other",
            )
        )
    return "".join(bars)


def _percent_caught(verification: dict[str, Any], kind: str) -> int:
    row = next(r for r in verification["planted"] if r["kind"] == kind)
    percent: int = round(100 * row["flagged"] / row["planted"])
    return percent


def _says_missing_vs_not(abstention: dict[str, Any]) -> tuple[int, int]:
    """Percent judged correct among the answers that say something is missing, and among the rest
    (the answerable questions of the abstention report)."""
    rows = [r for r in abstention["statuses"] if r["group"] == "answerable"]

    def share(keep: tuple[str, ...]) -> int:
        chosen = [r for r in rows if r["status"] in keep]
        percent: int = round(100 * sum(r["correct"] for r in chosen) / sum(r["n"] for r in chosen))
        return percent

    return share(("partial", "not_found")), share(("answered",))


def _with_markers(text: str) -> str:
    """The answer as escaped HTML, each `[n]` shown as a small marker."""
    return re.sub(r"\[(\d{1,3})\]", r'<sup class="mk">\1</sup>', _esc(text))


def _hero(example: dict[str, Any]) -> dict[str, str]:
    response = example["response"]
    verification = response["verification"]
    cost = response["usage"]["cost_usd"]
    return {
        "HERO_QUESTION": _esc(example["question"]),
        "HERO_ANSWER": _with_markers(response["answer_with_markers"]),
        "HERO_CHECK": (
            f"{verification['verified']} of {verification['statements']} statements verified"
        ),
        "HERO_SOURCES": f"{len(response['citations'])} sources cited",
        "HERO_COST": (f"${cost:.4f}" if cost is not None else "cost unknown")
        + f" · {response['timings_ms']['total'] / 1000:.0f} s on a laptop CPU",
    }


def _latency_range(recording: dict[str, Any]) -> str:
    totals = [e["response"]["timings_ms"]["total"] / 1000 for e in recording["examples"]]
    return f"{min(totals):.0f} to {max(totals):.0f}"


def _social(site_url: str | None, has_image: bool) -> str:
    if not site_url:
        return '<meta name="twitter:card" content="summary">'
    url = _esc(site_url)
    lines = [f'<meta property="og:url" content="{url}">']
    if has_image:
        lines += [
            f'<meta property="og:image" content="{url}og.png">',
            '<meta property="og:image:width" content="1200">',
            '<meta property="og:image:height" content="630">',
            '<meta property="og:image:alt" content="Hybrid RAG Production Engine: answers you can '
            'check, over half a million documents">',
            '<meta name="twitter:card" content="summary_large_image">',
        ]
    else:
        lines.append('<meta name="twitter:card" content="summary">')
    return "\n".join(lines)


def _fill(template: str, values: dict[str, str]) -> str:
    """Replace every `__NAME__` in one pass, so that inserted text is never read as a placeholder.
    A name without a value raises instead of reaching the page."""
    return re.sub(r"__([A-Z][A-Z0-9_]*[A-Z0-9])__", lambda match: values[match.group(1)], template)


def render_site(
    recording: dict[str, Any],
    trace: dict[str, Any],
    report: dict[str, Any],
    abstention: dict[str, Any],
    ablation: dict[str, Any],
    comparison: dict[str, Any],
    verification: dict[str, Any],
    repo_url: str,
    *,
    site_url: str | None = None,
    site_dir: Path = SITE_DIR,
) -> str:
    """The whole page as one HTML document."""
    missing_right, rest_right = _says_missing_vs_not(abstention)
    examples = [
        {**example, "note": NOTES.get(example["id"], "")} for example in recording["examples"]
    ]
    story = next((e for e in examples if e["id"] == trace["question_id"]), examples[0])
    data = _json_for_script(
        {
            "examples": examples,
            "trace": trace,
            "labels": TYPE_LABELS,
            "facts": {
                "sources": CORPUS_SOURCES,
                "missingRight": missing_right,
                "restRight": rest_right,
            },
        }
    )

    style = (site_dir / "page.css").read_text(encoding="utf-8")
    script = RENDER_JS + "\n" + (site_dir / "page.js").read_text(encoding="utf-8")
    policy = (
        "default-src 'none'; "
        f"style-src {_hash(style)}; script-src {_hash(script)}; "
        "img-src 'self' data:; base-uri 'none'; form-action 'none'"
    )

    repo = repo_url.rstrip("/")
    overall = report["overall"]
    by_type = report["by_type"]
    owner = repo.rsplit("/", 2)[-2]
    values = {
        "POLICY": _esc(policy),
        "DESCRIPTION": _esc(DESCRIPTION),
        "SOCIAL": _social(site_url, (site_dir / "og.png").exists()),
        "STYLE": style,
        "SCRIPT": script,
        "DATA": data,
        "REPO": _esc(repo),
        "DOCS": _esc(f"{repo}/blob/main/docs"),
        "OWNER": _esc(owner),
        "OWNER_URL": _esc(repo.rsplit("/", 1)[0]),
        **_hero(story),
        "QUESTIONS": _esc(overall["questions"]),
        "COMBINED": _esc(overall["combined"]),
        "CORRECT": _esc(overall["correct_pct"]),
        "DOC_RECALL": _esc(overall["document_recall_pct"]),
        "JUDGE": _esc(report["judge"]),
        "ANSWER_COST": f"{report['cost_usd']['answers']:.2f}",
        "ANSWER_COST_EACH": f"{report['cost_usd']['answers'] / overall['questions']:.4f}",
        "RECORDED_ON": _esc(recording["recorded_on"]),
        "HARDWARE": _esc(recording["server"]["hardware"]),
        "RERANKER": _esc(recording["server"]["reranker"]),
        "R_DENSE": f"{_recall(ablation, 'dense'):.3f}",
        "R_BM25": f"{_recall(ablation, 'bm25'):.3f}",
        "R_HYBRID": f"{_recall(ablation, 'hybrid, BM25 weight 1.5'):.3f}",
        "R_RERANK": f"{_recall(ablation, 'rerank bge-v2-m3, pool 30'):.3f}",
        "PLANTED_CITATION": str(_percent_caught(verification, "wrong citation")),
        "PLANTED_NUMBER": str(_percent_caught(verification, "wrong number")),
        "MISSING_RIGHT": str(missing_right),
        "REST_RIGHT": str(rest_right),
        "TYPE_BARS": _type_bars(report),
        "TYPE_TABLE": _results_table(report),
        "RECALL_BARS": _recall_bars(ablation),
        "RIVAL_BARS": _rival_bars(comparison),
        "W_SEMANTIC": _one_decimal(by_type["semantic"]["combined"]),
        "W_PROJECT": _one_decimal(by_type["project_related"]["combined"]),
        "W_COMPLETENESS": _one_decimal(by_type["completeness"]["combined"]),
        "LATENCY_RANGE": _latency_range(recording),
    }
    return _fill((site_dir / "template.html").read_text(encoding="utf-8"), values)
