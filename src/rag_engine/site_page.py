"""The project's static site, for GitHub Pages: the demo page replaying recorded responses.

GitHub Pages serves files and runs no code on a server, and the real service needs the 35 GB index
and a model key, so a public copy of the demo cannot be live. This page draws the same cards as
the live `/demo` (the look and the drawing code are imported from there) from responses that were
recorded from the running service and are embedded in the page, and says so on the page. Nothing
here calls any server.

The content security policy is a `<meta>` tag, because Pages cannot send headers. It allows the
page's one script and one style by hash, so nothing else can run on it even if some text on the page
were hostile, and the hashes are computed from the exact text that is inlined.
"""

import base64
import hashlib
import html
import json
from typing import Any

from rag_engine.api.demo_page import RENDER_JS, STYLE

_SITE_STYLE = """
header.top { margin-bottom: 1.5rem; }
nav { display: flex; flex-wrap: wrap; gap: .4rem 1.1rem; margin: .75rem 0 0; font-size: .95rem; }
a { color: var(--accent); }
.notice { background: var(--warn-bg); color: var(--fg); border-radius: 8px; padding: .75rem 1rem;
  font-size: .92rem; margin: 1rem 0; }
.question { font-weight: 600; margin: 0 0 .35rem; }
table { border-collapse: collapse; width: 100%; font-size: .92rem; }
th, td { text-align: left; padding: .35rem .5rem; border-bottom: 1px solid var(--line); }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
pre { background: var(--card); border: 1px solid var(--line); border-radius: 8px;
  padding: .8rem 1rem; overflow-x: auto; font-size: .88rem; }
footer { margin-top: 3rem; color: var(--muted); font-size: .85rem; }
ol.steps { padding-left: 1.2rem; }
ol.steps li { margin: .5rem 0; }
"""

_SITE_JS = """
const recording = JSON.parse($("recording").textContent);
const picker = $("example");
recording.examples.forEach((example, index) => {
  const text = example.question;
  const label = "[" + example.type + "] " + (text.length > 100 ? text.slice(0, 97) + "..." : text);
  const option = el("option", label);
  option.value = String(index);
  picker.append(option);
});

function show(index) {
  const example = recording.examples[index];
  const body = example.response;
  $("question").textContent = example.question;
  $("note").textContent = example.note || "";
  $("result").replaceChildren(
    el("h2", "Answer"), renderAnswer(body),
    el("h2", "Sources"), renderSources(body),
    el("h2", "Statement by statement"), renderClaims(body),
    el("h2", "Cost and time"), renderCost(body));
}

picker.addEventListener("change", () => show(Number(picker.value)));
show(0);
"""

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


def _hash(text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return "'sha256-" + base64.b64encode(digest).decode("ascii") + "'"


def _json_for_script(data: Any) -> str:
    """JSON that is safe inside a `<script>` data block: no `<`, so no `</script>` or `<!--`."""
    return json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")


def _results_table(report: dict[str, Any]) -> str:
    rows = sorted(report["by_type"].items(), key=lambda item: -item[1]["combined"])
    body = "".join(
        f'<tr><td>{html.escape(name)}</td><td class="num">{summary["questions"]}</td>'
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


def _says_missing_vs_not(abstention: dict[str, Any]) -> tuple[int, int]:
    """Percent judged correct among the answers that say something is missing, and among the rest
    (the answerable questions of the abstention report)."""
    rows = [r for r in abstention["statuses"] if r["group"] == "answerable"]

    def share(keep: tuple[str, ...]) -> int:
        chosen = [r for r in rows if r["status"] in keep]
        percent: int = round(100 * sum(r["correct"] for r in chosen) / sum(r["n"] for r in chosen))
        return percent

    return share(("partial", "not_found")), share(("answered",))


def render_site(
    recording: dict[str, Any],
    report: dict[str, Any],
    abstention: dict[str, Any],
    repo_url: str,
) -> str:
    """The whole page as one HTML document."""
    missing_right, rest_right = _says_missing_vs_not(abstention)
    repo = html.escape(repo_url, quote=True)
    examples = [
        {**example, "note": NOTES.get(example["id"], "")} for example in recording["examples"]
    ]
    data = _json_for_script({"examples": examples})
    style = STYLE + _SITE_STYLE
    script = RENDER_JS + "\n" + _SITE_JS
    policy = (
        "default-src 'none'; "
        f"style-src {_hash(style)}; script-src {_hash(script)}; "
        "img-src 'self' data:; base-uri 'none'; form-action 'none'"
    )
    docs = f"{repo}/blob/main/docs"
    overall = report["overall"]
    server = recording["server"]
    judge = html.escape(report["judge"])
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="{html.escape(policy, quote=True)}">
<meta name="description" content="Hybrid dense + BM25 retrieval, cross-encoder reranking and
verified citations over half a million documents, measured on EnterpriseRAG-Bench.">
<title>Hybrid RAG Production Engine</title>
<style>{style}</style>
</head>
<body>
<main>
  <header class="top">
    <h1>Hybrid RAG Production Engine</h1>
    <p class="lead">Questions answered over 511,962 company documents with hybrid search
    (dense vectors and BM25), a cross-encoder reranker, citations checked statement by statement,
    and an honest score on a public benchmark.</p>
    <nav>
      <a href="{repo}">Repository</a>
      <a href="{docs}/results.md">Results</a>
      <a href="{docs}/reproduce.md">Reproduce it</a>
      <a href="{docs}/lessons-learned.md">Lessons learned</a>
      <a href="{docs}/operations.md">Running it</a>
    </nav>
  </header>

  <h2>Try it: recorded answers</h2>
  <div class="notice">This page <strong>replays</strong> real responses of the service, recorded on
  {html.escape(recording["recorded_on"])} from a development laptop with
  {html.escape(server["hardware"])},
  using the small <code>{html.escape(server["reranker"])}</code> reranker. It does not call any
  server: the live service needs a 35&nbsp;GB index and a model key, so it runs from the
  repository (see below). The times are that laptop's, and the benchmark scores further down used
  a larger reranker on a GPU.</div>

  <label for="example">Pick a question</label>
  <select id="example"></select>
  <div class="card">
    <p class="question" id="question"></p>
    <p class="muted" id="note"></p>
  </div>
  <section id="result" aria-live="polite"></section>

  <h2>How it did on the benchmark</h2>
  <p>All {overall["questions"]} questions of EnterpriseRAG-Bench, answered and scored by the
  benchmark's own evaluator with {judge} as the judge: <strong>{overall["combined"]}</strong> on its
  combined correctness-and-completeness measure, {overall["correct_pct"]}% of answers correct.
  The judge is not the one the public leaderboard uses, so this is not comparable with it. The
  system is strong on questions that name what they want and weak on the ones spread over many
  documents:</p>
  {_results_table(report)}
  <p class="muted">Answering all 500 questions cost ${report["cost_usd"]["answers"]:.2f}.
  <a href="{docs}/results.md">Every result, with its limits</a>.</p>

  <h2>How it works</h2>
  <ol class="steps">
    <li><strong>Search two ways at once.</strong> A dense embedding model finds passages by
    meaning and BM25 finds them by exact words (identifiers, flag names); their rankings are
    fused.</li>
    <li><strong>Rerank.</strong> A cross-encoder rescores the 30 best candidates and the top five
    reach the language model with the text around them.</li>
    <li><strong>Answer with citations.</strong> The model writes prose that cites passages by
    number.</li>
    <li><strong>Check every statement.</strong> The cited passage must exist, its numbers and names
    must appear in it, and a second call must quote the sentence that supports the claim and that
    quote must be found in the passage. Statements that fail are flagged, not hidden.</li>
    <li><strong>Say when the documents fall short.</strong> The answer reports what it could not
    find; answers that say so were right {missing_right}% of the time, the others
    {rest_right}%.</li>
  </ol>

  <h2>Run the live service</h2>
  <pre>git clone {repo}
cd hybrid-rag-production-engine
cp .env.example .env            # set POSTGRES_PASSWORD, ANTHROPIC_API_KEY
docker compose up --build --wait
open http://localhost:8000/demo</pre>
  <p>It needs the index loaded first; <a href="{docs}/setup.md">the setup guide</a> covers the
  machine and <a href="{docs}/reproduce.md">the reproduction guide</a> every step and its cost.</p>

  <footer>
    <p>The documents and questions are from EnterpriseRAG-Bench (MIT license), a synthetic
    company's data generated by language models. Its authors ask that benchmark data stay out of
    training corpora; please respect that for the examples above.</p>
    <p><a href="{repo}">Source code and documentation on GitHub</a></p>
  </footer>
</main>
<script type="application/json" id="recording">{data}</script>
<script>{script}</script>
</body>
</html>
"""
