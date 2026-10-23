"""A one-page demo of POST /v1/query, served by the API itself at /demo.

It is plain HTML, CSS and JavaScript in one string: no build step, no framework and nothing loaded
from another site. The page never puts what the API returns into the document as HTML (everything
goes through `textContent`), and the response carries a content security policy that allows only
this page's own script and style, so a hostile passage in the corpus cannot run code in it.
"""

import secrets

# The pieces are shared with the recorded copy of this page (site_page.py), which the project's
# GitHub Pages site serves: the look and the drawing code are written once.
STYLE = """:root {
  --bg: #fafaf7; --fg: #1c1d1f; --muted: #62666d; --line: #dcdcd4; --card: #ffffff;
  --accent: #1f5f8b; --ok: #1d7a46; --warn: #946200; --bad: #a3322b;
  --ok-bg: #e4f3ea; --warn-bg: #fbf0d4; --bad-bg: #f8e3e1; --neutral-bg: #ececE6;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #17181a; --fg: #e8e8e4; --muted: #9a9ea6; --line: #34363a; --card: #202225;
    --accent: #6cb4e4; --ok: #6fd09a; --warn: #e6b94d; --bad: #f0877f;
    --ok-bg: #1b3326; --warn-bg: #3a2f13; --bad-bg: #3c1f1d; --neutral-bg: #2b2d31;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg);
  font: 16px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 52rem; margin: 0 auto; padding: 1.5rem 1rem 4rem; }
h1 { font-size: 1.4rem; margin: 0 0 .25rem; }
h2 { font-size: 1rem; margin: 1.5rem 0 .5rem; text-transform: uppercase; letter-spacing: .04em;
  color: var(--muted); }
p.lead { margin: 0 0 1.25rem; color: var(--muted); }
label { display: block; font-size: .85rem; color: var(--muted); margin-bottom: .25rem; }
textarea, input, select { width: 100%; font: inherit; color: inherit; background: var(--card);
  border: 1px solid var(--line); border-radius: 6px; padding: .5rem .6rem; }
textarea { min-height: 5rem; resize: vertical; }
.row { display: flex; flex-wrap: wrap; gap: .75rem; margin-top: .75rem; }
.row > div { flex: 1 1 9rem; }
.row .check { flex: 0 0 auto; display: flex; align-items: center; gap: .4rem; padding-top: 1.4rem; }
.row .check input { width: auto; }
.row .check label { margin: 0; }
button { font: inherit; background: var(--accent); color: var(--bg); border: 0; border-radius: 6px;
  padding: .55rem 1.1rem; cursor: pointer; margin-top: 1rem; }
button:disabled { opacity: .5; cursor: progress; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 8px;
  padding: 1rem; margin-top: .5rem; }
.badge { display: inline-block; font-size: .75rem; padding: .1rem .5rem; border-radius: 99px;
  background: var(--neutral-bg); color: var(--fg); margin-right: .35rem; }
.badge.ok { background: var(--ok-bg); color: var(--ok); }
.badge.warn { background: var(--warn-bg); color: var(--warn); }
.badge.bad { background: var(--bad-bg); color: var(--bad); }
.answer { white-space: pre-wrap; }
.marker { color: var(--accent); cursor: pointer; font-size: .8em; vertical-align: super; }
.marker:focus-visible, button:focus-visible {
  outline: 2px solid var(--accent); outline-offset: 2px; }
ul { padding-left: 1.1rem; margin: .25rem 0; }
li { margin: .35rem 0; }
.claim { border-top: 1px solid var(--line); padding: .6rem 0; }
.claim:first-child { border-top: 0; }
.evidence { color: var(--muted); font-size: .9rem; margin-top: .2rem; }
.muted { color: var(--muted); font-size: .9rem; }
.bar { height: .5rem; background: var(--neutral-bg); border-radius: 4px; overflow: hidden; }
.bar > span { display: block; height: 100%; background: var(--accent); }
.stage { display: grid; grid-template-columns: 11rem 1fr 4.5rem; gap: .5rem; align-items: center;
  font-size: .85rem; margin: .2rem 0; }
.stage .num { text-align: right; color: var(--muted); }
.error { color: var(--bad); }
.hit { outline: 2px solid var(--accent); border-radius: 4px; }
"""

RENDER_JS = """"use strict";
const $ = (id) => document.getElementById(id);
const STATUS_CLASS = {
  verified: "ok", no_claim: "", partially_supported: "warn", uncited: "warn",
  unchecked: "warn", ungrounded: "warn", unsupported: "bad", unresolved: "bad",
};
const ANSWER_CLASS = { answered: "ok", partial: "warn", not_found: "warn", abstained: "bad" };
const ANSWER_TEXT = {
  answered: "answered", partial: "partial: the documents lack part of it",
  not_found: "not found in the documents", abstained: "declined: weak evidence",
};

function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined && text !== null) node.textContent = text;
  if (className) node.className = className;
  return node;
}
function badge(text, kind) { return el("span", text, "badge" + (kind ? " " + kind : "")); }

function renderAnswer(body) {
  const box = el("div", null, "card");
  const head = el("div");
  head.append(badge(ANSWER_TEXT[body.status] || body.status, ANSWER_CLASS[body.status]));
  if (body.cached) head.append(badge("from the cache"));
  head.append(badge(body.mode));
  box.append(head);

  const answer = el("p", null, "answer");
  // Split the text on its [n] markers; each marker becomes a button-like link to its source.
  for (const part of body.answer_with_markers.split(/(\\[\\d{1,3}\\])/)) {
    const marker = /^\\[(\\d{1,3})\\]$/.exec(part);
    if (marker) {
      const link = el("a", part, "marker");
      link.tabIndex = 0;
      link.setAttribute("role", "link");
      const go = () => highlight(Number(marker[1]));
      link.addEventListener("click", go);
      link.addEventListener("keydown", (e) => { if (e.key === "Enter") go(); });
      answer.append(link);
    } else {
      answer.append(document.createTextNode(part));
    }
  }
  box.append(answer);

  if (body.gaps.length) {
    box.append(el("h2", "What the answer says is missing"));
    const list = el("ul");
    for (const gap of body.gaps) list.append(el("li", gap));
    box.append(list);
  }
  if (body.abstention) {
    const a = body.abstention;
    box.append(el("p", "Reason: " + a.reason + (a.top_score === null ? "" :
      " (best score " + a.top_score.toFixed(3) + ", threshold " + a.threshold + ")"), "muted"));
  }
  return box;
}

function renderSources(body) {
  const box = el("div", null, "card");
  if (!body.citations.length) {
    box.append(el("span", "No passage is cited.", "muted"));
    return box;
  }
  const list = el("ul");
  for (const c of body.citations) {
    const item = el("li");
    item.id = "source-" + c.marker;
    item.append(badge("[" + c.marker + "]"));
    const verified = c.verified === true ? ["checked", "ok"] :
      c.verified === false ? ["not fully supported", "warn"] : ["not checked", ""];
    item.append(badge(verified[0], verified[1]), el("strong", c.title || "(untitled)"));
    item.append(el("div", c.source_type + " \u00b7 " + c.doc_id, "muted"));
    list.append(item);
  }
  box.append(list);
  return box;
}

function renderClaims(body) {
  const box = el("div", null, "card");
  if (!body.claims.length) {
    box.append(el("span", "No statements to check.", "muted"));
    return box;
  }
  const v = body.verification;
  box.append(el("p", v.method === "judge"
    ? v.verified + " of " + v.statements + " statements verified by " + v.model
    : "Citations were not checked by a judge (method: " + v.method + ")", "muted"));
  if (v.error) box.append(el("p", "The check could not run: " + v.error, "error"));
  for (const claim of body.claims) {
    const row = el("div", null, "claim");
    row.append(badge(claim.status.replace("_", " "), STATUS_CLASS[claim.status]));
    if (claim.markers.length) row.append(badge(claim.markers.map((n) => "[" + n + "]").join("")));
    row.append(document.createTextNode(claim.text));
    if (claim.reason) row.append(el("div", claim.reason, "evidence"));
    if (claim.evidence) row.append(el("div", "\u201c" + claim.evidence + "\u201d", "evidence"));
    box.append(row);
  }
  return box;
}

function renderCost(body) {
  const box = el("div", null, "card");
  const u = body.usage;
  box.append(el("p", u.input_tokens + " tokens in, " + u.output_tokens + " out \u00b7 " +
    (u.cost_usd === null ? "cost unknown" : "$" + u.cost_usd.toFixed(4)) +
    " \u00b7 " + u.model, "muted"));
  const stages = Object.entries(body.timings_ms).filter(([name]) => name !== "total");
  const longest = Math.max(1, ...stages.map(([, ms]) => ms));
  for (const [name, ms] of stages) {
    const row = el("div", null, "stage");
    const bar = el("div", null, "bar");
    const fill = el("span");
    fill.style.width = Math.max(1, (100 * ms) / longest) + "%";
    bar.append(fill);
    row.append(el("span", name.replace("retrieval_", "retrieval \u00b7 ").replace(/_/g, " ")), bar,
      el("span", (ms / 1000).toFixed(2) + " s", "num"));
    box.append(row);
  }
  box.append(el("p", "Total " + ((body.timings_ms.total || 0) / 1000).toFixed(2) + " s", "muted"));
  return box;
}

function highlight(marker) {
  document.querySelectorAll(".hit").forEach((n) => n.classList.remove("hit"));
  const target = document.getElementById("source-" + marker);
  if (target) { target.classList.add("hit"); target.scrollIntoView({ block: "center" }); }
}
"""

_LIVE_MAIN = """<main>
  <h1>Hybrid RAG demo</h1>
  <p class="lead">Ask the indexed company documents a question. Every statement of the answer is
  checked against the passage it cites.</p>

  <form id="form">
    <label for="question">Question</label>
    <textarea id="question" maxlength="2000" required
      placeholder="What is the default grace period of the tool-call batching invoker?"></textarea>
    <div class="row">
      <div><label for="key">API key (if the server asks for one)</label>
        <input id="key" type="password" autocomplete="off" spellcheck="false"></div>
      <div><label for="mode">Retrieval</label>
        <select id="mode">
          <option value="">server default</option>
          <option value="hybrid+rerank">hybrid + rerank</option>
          <option value="hybrid">hybrid</option>
          <option value="dense">dense</option>
          <option value="bm25">BM25</option>
        </select></div>
      <div><label for="chunks">Passages</label>
        <input id="chunks" type="number" min="1" max="10" placeholder="5"></div>
      <div class="check"><input id="verify" type="checkbox" checked>
        <label for="verify">Check citations</label></div>
      <div class="check"><input id="fresh" type="checkbox">
        <label for="fresh">Skip the cache</label></div>
    </div>
    <button id="ask" type="submit">Ask</button>
    <span id="progress" class="muted" role="status"></span>
  </form>

  <section id="result" hidden aria-live="polite"></section>
</main>
"""

_LIVE_JS = """const REASONS = {
  401: "The server asks for an API key. Enter one above.",
  429: "Too many requests. Try again in a moment.",
  502: "The language model failed. Try again.",
  503: "Retrieval or answer generation is not available right now.",
};

$("form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const result = $("result");
  const button = $("ask");
  const request = { query: $("question").value.trim(), verify: $("verify").checked,
    cache: !$("fresh").checked };
  if ($("mode").value) request.mode = $("mode").value;
  if ($("chunks").value) request.chunks = Number($("chunks").value);
  const headers = { "content-type": "application/json" };
  if ($("key").value) headers["X-API-Key"] = $("key").value;

  button.disabled = true;
  $("progress").textContent = " Working\u2026 a first question can take half a minute.";
  result.hidden = true;
  result.replaceChildren();
  try {
    const response = await fetch("/v1/query", {
      method: "POST", headers, body: JSON.stringify(request),
    });
    if (!response.ok) {
      let detail = "";
      try {
        const payload = await response.json();
        detail = typeof payload.detail === "string" ? payload.detail : "";
      } catch (_) {}
      result.append(el("p", (REASONS[response.status] || "Request failed.") +
        (detail ? " (" + detail + ")" : ""), "error"));
    } else {
      const body = await response.json();
      result.append(
        el("h2", "Answer"), renderAnswer(body),
        el("h2", "Sources"), renderSources(body),
        el("h2", "Statement by statement"), renderClaims(body),
        el("h2", "Cost and time"), renderCost(body));
    }
  } catch (error) {
    result.append(el("p", "Could not reach the server.", "error"));
  } finally {
    button.disabled = false;
    $("progress").textContent = "";
    result.hidden = false;
  }
});
"""

_SHELL = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hybrid RAG demo</title>
<style nonce="__NONCE__">
__STYLE__</style>
</head>
<body>
__MAIN__
<script nonce="__NONCE__">
__SCRIPT__</script>
</body>
</html>
"""


def render_demo() -> tuple[str, str]:
    """The page with a fresh nonce, and the content security policy that matches it."""
    nonce = secrets.token_urlsafe(16)
    policy = (
        "default-src 'none'; "
        f"script-src 'nonce-{nonce}'; "
        f"style-src 'nonce-{nonce}'; "
        "connect-src 'self'; "
        "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    )
    page = (
        _SHELL.replace("__STYLE__", STYLE)
        .replace("__MAIN__", _LIVE_MAIN)
        .replace("__SCRIPT__", RENDER_JS + "\n" + _LIVE_JS)
        .replace("__NONCE__", nonce)
    )
    return page, policy
