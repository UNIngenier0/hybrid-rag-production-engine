import base64
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest

from rag_engine.api.schemas import QueryResponse
from rag_engine.site_page import NOTES, render_site

ROOT = Path(__file__).resolve().parents[2]
REPO = "https://github.com/example/project"


def load(path: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((ROOT / path).read_text(encoding="utf-8"))
    return data


@pytest.fixture(scope="module")
def recording() -> dict[str, Any]:
    return load("site/examples.json")


@pytest.fixture(scope="module")
def page(recording: dict[str, Any]) -> str:
    return render_site(
        recording,
        load("eval/reports/full-benchmark.json"),
        load("eval/reports/abstention-scores.json"),
        REPO,
    )


def sha(text: str) -> str:
    return "'sha256-" + base64.b64encode(hashlib.sha256(text.encode()).digest()).decode() + "'"


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


def test_the_policy_allows_exactly_the_pages_own_script_and_style_by_hash(page: str) -> None:
    policy = re.search(r'http-equiv="Content-Security-Policy" content="([^"]+)"', page)
    assert policy
    text = policy.group(1).replace("&#x27;", "'")
    style = re.search(r"<style>(.*?)</style>", page, re.S)
    script = re.search(r"<script>(.*?)</script>", page, re.S)
    assert style and script

    assert f"style-src {sha(style.group(1))}" in text
    assert f"script-src {sha(script.group(1))}" in text
    assert (
        "default-src 'none'" in text and "unsafe-inline" not in text and "connect-src" not in text
    )


def test_the_page_loads_nothing_from_another_site_and_never_builds_markup_from_answers(
    page: str,
) -> None:
    assert not re.search(r"""(?:src|href)=["']https?://(?!github\.com/example/)""", page)
    assert not re.search(r"""<(?:img|iframe|link|object|embed)\b""", page)
    for forbidden in ("innerHTML", "insertAdjacentHTML", "eval(", "document.write", "fetch("):
        assert forbidden not in page


def test_text_in_a_recorded_answer_cannot_end_the_data_block_or_open_markup() -> None:
    hostile = {
        "recorded_on": "2026-01-01",
        "server": {"reranker": "r", "hardware": "h"},
        "examples": [
            {
                "id": "x",
                "type": "basic",
                "question": "</script><script>alert(1)</script>",
                "response": {"answer": "<!-- <img src=x onerror=alert(1)> -->"},
            }
        ],
    }

    page = render_site(
        hostile,
        load("eval/reports/full-benchmark.json"),
        load("eval/reports/abstention-scores.json"),
        REPO,
    )

    assert page.count("</script>") == 2  # the data block and the page's script, nothing else
    assert "<!--" not in page.split('id="recording">')[1].split("</script>")[0]
    data = page.split('id="recording">')[1].split("</script>")[0]
    assert json.loads(data)["examples"][0]["question"] == "</script><script>alert(1)</script>"


def test_the_numbers_on_the_page_are_the_published_ones(page: str) -> None:
    report = load("eval/reports/full-benchmark.json")

    assert f"<strong>{report['overall']['combined']}</strong>" in page
    assert f"${report['cost_usd']['answers']:.2f}" in page
    for name, summary in report["by_type"].items():
        assert f'<td>{name}</td><td class="num">{summary["questions"]}</td>' in page
    abstention = load("eval/reports/abstention-scores.json")
    answered = next(
        r
        for r in abstention["statuses"]
        if r["group"] == "answerable" and r["status"] == "answered"
    )
    assert f"the others\n    {round(100 * answered['correct'] / answered['n'])}%" in page


def test_the_page_says_it_is_a_recording_and_gives_the_judge_caveat(
    page: str, recording: dict[str, Any]
) -> None:
    assert "replays" in page and "does not call any" in page
    assert recording["server"]["reranker"] in page
    assert "not comparable" in page


def test_the_links_point_at_the_repository(page: str) -> None:
    links = set(re.findall(r'href="([^"]+)"', page))

    assert f"{REPO}/blob/main/docs/results.md" in links
    assert all(link.startswith(REPO) for link in links)
