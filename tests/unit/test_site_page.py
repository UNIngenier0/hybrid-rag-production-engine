import base64
import copy
import hashlib
import json
import re
import struct
from pathlib import Path
from typing import Any

import pytest

from rag_engine.api.demo_page import RENDER_JS
from rag_engine.api.schemas import QueryResponse
from rag_engine.site_page import (
    CORPUS_DOCUMENTS,
    CORPUS_SOURCES,
    NOTES,
    TYPE_LABELS,
    _fill,
    _one_decimal,
    render_site,
)

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "site"
REPO = "https://github.com/example/project"
SITE_URL = "https://example.github.io/project/"


def load(path: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((ROOT / path).read_text(encoding="utf-8"))
    return data


@pytest.fixture(scope="module")
def recording() -> dict[str, Any]:
    return load("site/examples.json")


@pytest.fixture(scope="module")
def trace() -> dict[str, Any]:
    return load("site/trace.json")


def build(recording: dict[str, Any], trace: dict[str, Any], site_url: str | None = SITE_URL) -> str:
    return render_site(
        recording,
        trace,
        load("eval/reports/full-benchmark.json"),
        load("eval/reports/abstention-scores.json"),
        load("eval/reports/retrieval-ablation.json"),
        load("eval/reports/answer-comparison.json"),
        load("eval/reports/citation-verification.json"),
        REPO,
        site_url=site_url,
    )


@pytest.fixture(scope="module")
def page(recording: dict[str, Any], trace: dict[str, Any]) -> str:
    return build(recording, trace)


def sha(text: str) -> str:
    return "'sha256-" + base64.b64encode(hashlib.sha256(text.encode()).digest()).decode() + "'"


def site_script() -> str:
    return RENDER_JS + "\n" + (SITE / "page.js").read_text(encoding="utf-8")


# ---------- the recorded data ----------


def test_the_recorded_responses_still_have_the_shape_the_api_returns(
    recording: dict[str, Any],
) -> None:
    assert recording["examples"], "nothing was recorded"
    for example in recording["examples"]:
        response = QueryResponse.model_validate(example["response"])
        assert response.cached is False, example["id"]  # recorded from the pipeline, not the cache
        assert response.answer_with_markers.strip() and response.usage.input_tokens > 0


def test_every_noted_question_was_recorded(recording: dict[str, Any]) -> None:
    ids = {example["id"] for example in recording["examples"]}

    assert ids <= set(NOTES), "an example has no note about what to look at"
    assert set(NOTES) <= ids, "a note names a question that was not recorded"


def test_every_question_type_of_the_recording_has_a_label(recording: dict[str, Any]) -> None:
    assert {e["type"] for e in recording["examples"]} <= set(TYPE_LABELS)


def test_the_corpus_table_adds_up_to_the_corpus() -> None:
    assert sum(count for _, count in CORPUS_SOURCES) == CORPUS_DOCUMENTS == 511_962


def test_the_trace_is_the_whole_candidate_pool_of_a_recorded_question(
    recording: dict[str, Any], trace: dict[str, Any]
) -> None:
    pool = trace["pool"]
    question = next(e for e in recording["examples"] if e["id"] == trace["question_id"])

    assert trace["query"] == question["question"]
    assert len(pool) == 30 and len({hit["chunk_id"] for hit in pool}) == 30
    assert sorted(hit["rank"] for hit in pool) == list(range(1, 31))
    assert sorted(hit["fusion_rank"] for hit in pool) == list(range(1, 31))
    assert all(hit["dense_rank"] or hit["bm25_rank"] for hit in pool)
    assert trace["reranker"] == recording["server"]["reranker"]


def test_the_walkthrough_tells_what_the_recorded_trace_shows(
    recording: dict[str, Any], trace: dict[str, Any], page: str
) -> None:
    """The text of the steps names facts of one recorded question. If it is recorded again and a
    fact changes, this fails until the text is rewritten."""
    question = next(e for e in recording["examples"] if e["id"] == trace["question_id"])
    cited = {c["doc_id"] for c in question["response"]["citations"]}
    by_rerank = sorted(trace["pool"], key=lambda hit: hit["rank"])
    first_five = by_rerank[:5]
    pick = next(
        hit
        for hit in by_rerank
        if hit["doc_id"] in cited and hit["dense_rank"] is None and hit["bm25_rank"]
    )

    assert {hit["doc_id"] for hit in first_five} == cited and len(cited) == 2
    assert "from 2 different documents" in page
    assert (pick["rank"], pick["bm25_rank"], pick["fusion_rank"]) == (2, 2, 11)
    assert "climbs from 11th to 2nd" in page and "BM25 puts it second" in page
    for phrase in re.findall(r'\["([^"]+)", "hl-(?:exact|meaning)"\]', site_script()):
        assert phrase in trace["query"], f"the highlighted phrase {phrase!r} is not in the question"


# ---------- the page's rules ----------


def test_the_policy_allows_exactly_the_pages_own_script_and_style_by_hash(page: str) -> None:
    policy = re.search(r'http-equiv="Content-Security-Policy" content="([^"]+)"', page)
    assert policy
    text = policy.group(1).replace("&#x27;", "'")
    style = re.findall(r"<style>(.*?)</style>", page, re.S)
    script = re.findall(r"<script>(.*?)</script>", page, re.S)
    assert len(style) == 1 and len(script) == 1

    assert f"style-src {sha(style[0])}" in text
    assert f"script-src {sha(script[0])}" in text
    assert (
        "default-src 'none'" in text and "unsafe-inline" not in text and "connect-src" not in text
    )


def test_the_page_loads_nothing_from_another_site_and_has_no_inline_styles(page: str) -> None:
    assert not re.search(r"""\bsrc=["']https?:""", page)
    assert not re.search(r"""<(?:img|iframe|object|embed|form|base)\b""", page)
    links = re.findall(r"<link\b[^>]*>", page)
    assert len(links) == 1 and 'rel="icon"' in links[0] and 'href="data:image/svg+xml' in links[0]
    assert ' style="' not in page and "<style " not in page


def test_the_script_never_builds_markup_from_data_or_talks_to_the_network() -> None:
    script = site_script()
    for forbidden in (
        "innerHTML",
        "outerHTML",
        "insertAdjacentHTML",
        "document.write",
        "eval(",
        "new Function",
        "fetch(",
        "XMLHttpRequest",
        "WebSocket",
        "sendBeacon",
        "import(",
        "cssText",
        'setAttribute("style"',
    ):
        assert forbidden not in script, forbidden


def test_every_id_the_script_looks_up_exists_in_the_template() -> None:
    template = (SITE / "template.html").read_text(encoding="utf-8")
    defined = set(re.findall(r'\bid="([^"]+)"', template))
    wanted = set(re.findall(r'\$\("([^"]+)"\)', (SITE / "page.js").read_text(encoding="utf-8")))

    assert wanted - defined == set()


def test_the_walkthrough_has_one_button_per_scene_and_every_scene_has_a_duration(
    page: str,
) -> None:
    scenes = re.findall(r'<article class="wt-scene" data-dur="(\d+)"', page)
    buttons = re.findall(r'class="wt-step" type="button" data-go="(\d+)"', page)

    assert len(scenes) == len(buttons) == 7
    assert buttons == [str(i) for i in range(7)] and all(int(d) >= 5000 for d in scenes)


def test_no_placeholder_is_left_in_the_page(page: str) -> None:
    assert not re.search(r"__[A-Z][A-Z0-9_]*[A-Z0-9]__", page)


def test_a_placeholder_without_a_value_raises_instead_of_reaching_the_page() -> None:
    with pytest.raises(KeyError):
        _fill("<p>__MISSING__</p>", {})

    # Inserted text is never read as a placeholder.
    assert _fill("<p>__AA__</p>", {"AA": "__BB__"}) == "<p>__BB__</p>"


def test_text_in_the_recorded_data_cannot_end_the_data_block_or_open_markup(
    recording: dict[str, Any], trace: dict[str, Any]
) -> None:
    hostile = "</script><script>alert(1)</script><!-- <img src=x onerror=alert(1)> -->"
    changed = copy.deepcopy(recording)
    changed["examples"][0]["question"] = hostile
    changed["examples"][0]["response"]["answer_with_markers"] = hostile + " [1]"
    changed["examples"][0]["response"]["citations"][0]["title"] = hostile
    changed_trace = copy.deepcopy(trace)
    changed_trace["query"] = hostile
    changed_trace["pool"][0]["title"] = hostile

    page = build(changed, changed_trace)

    assert page.count("</script>") == 2  # the data block and the page's script, nothing else
    assert "<script>alert(1)" not in page and "<img src=x" not in page
    data = page.split('id="site-data">')[1].split("</script>")[0]
    assert "<" not in data
    assert json.loads(data)["examples"][0]["question"] == hostile
    assert json.loads(data)["trace"]["pool"][0]["title"] == hostile


# ---------- the numbers on the page ----------


def test_one_decimal_rounds_half_up_like_a_reader_would() -> None:
    assert _one_decimal(25.65) == "25.7" and _one_decimal(6.86) == "6.9"
    assert _one_decimal(27.5) == "27.5" and _one_decimal(100.0) == "100.0"


def test_the_numbers_on_the_page_are_the_published_ones(page: str) -> None:
    report = load("eval/reports/full-benchmark.json")
    overall = report["overall"]

    assert f"<strong>{overall['combined']}</strong>" in page
    assert f'data-count="{overall["combined"]}"' in page
    assert f"${report['cost_usd']['answers']:.2f}" in page
    assert f"__{overall['combined']}" not in page
    for name, summary in report["by_type"].items():
        assert f'<td>{name}</td><td class="num">{summary["questions"]}</td>' in page
        assert TYPE_LABELS[name] in page
    semantic = _one_decimal(report["by_type"]["semantic"]["combined"])
    assert f"score {semantic} (semantic)" in page


def test_the_retrieval_and_comparison_charts_use_the_reports(page: str) -> None:
    ablation = load("eval/reports/retrieval-ablation.json")
    comparison = load("eval/reports/answer-comparison.json")

    for system in ("dense", "bm25", "hybrid, BM25 weight 1.5", "rerank bge-v2-m3, pool 30"):
        assert f"{ablation['systems'][system]['5']['overall']['recall']:.3f}" in page
    for arm in comparison["arms"].values():
        assert (
            f'<span class="hbar-val">{_one_decimal(round(100 * arm["combined"], 4))}</span>' in page
        )


def test_the_verification_and_status_figures_are_computed_from_their_reports(page: str) -> None:
    check = load("eval/reports/citation-verification.json")
    moved = next(r for r in check["planted"] if r["kind"] == "wrong citation")
    changed = next(r for r in check["planted"] if r["kind"] == "wrong number")
    abstention = load("eval/reports/abstention-scores.json")
    answered = next(
        r
        for r in abstention["statuses"]
        if r["group"] == "answerable" and r["status"] == "answered"
    )

    assert f"<b>{round(100 * moved['flagged'] / moved['planted'])}% / " in page
    assert f"{round(100 * changed['flagged'] / changed['planted'])}%</b>" in page
    assert f"<strong>{round(100 * answered['correct'] / answered['n'])}%</strong>" in page


def test_the_page_says_it_is_a_recording_and_gives_the_judge_caveat(
    page: str, recording: dict[str, Any]
) -> None:
    assert "replays" in page and "does not call any" in page
    assert recording["server"]["reranker"] in page
    assert "not comparable" in page


def test_the_links_point_at_the_repository_or_the_benchmark(page: str) -> None:
    links = set(re.findall(r'href="([^"]+)"', page))
    allowed = ("#", REPO, "https://github.com/example", "https://github.com/onyx-dot-app/")

    assert f"{REPO}/blob/main/docs/results.md" in links
    assert all(link.startswith(allowed) or link.startswith("data:image/svg+xml") for link in links)


def test_the_share_preview_is_named_only_when_the_site_address_is_known(
    recording: dict[str, Any], trace: dict[str, Any], page: str
) -> None:
    assert f'<meta property="og:image" content="{SITE_URL}og.png">' in page
    assert 'content="summary_large_image"' in page
    assert "og:image" not in build(recording, trace, site_url=None)


def test_the_preview_image_is_1200_by_630_and_shows_the_published_numbers() -> None:
    header = (SITE / "og.png").read_bytes()[:24]
    width, height = struct.unpack(">II", header[16:24])
    source = (SITE / "og.html").read_text(encoding="utf-8")
    report = load("eval/reports/full-benchmark.json")["overall"]

    assert header[:8] == b"\x89PNG\r\n\x1a\n" and (width, height) == (1200, 630)
    assert f"<b>{report['combined']}</b>" in source
    assert f"<b>{report['correct_pct']}%</b>" in source
    assert f"<b>{CORPUS_DOCUMENTS:,}</b>" in source
