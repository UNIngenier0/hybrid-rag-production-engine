/* The project's site script. Inlined into the page by src/rag_engine/site_page.py after the
   drawing code of the demo page (RENDER_JS: `$`, `el`, `badge` and the render* functions).

   Rules this file keeps, because the content security policy and the tests depend on them:
   - Text from the recorded data only goes into the page through textContent or setAttribute,
     never as markup.
   - Nothing is loaded from the network.
   - Styles are set through custom properties (element.style.setProperty), never as attributes.

   All the timing of a scene is CSS (animation delays set here); the only JavaScript timer is the
   progress bar of the current step, whose end moves the walkthrough to the next step. Pausing
   therefore pauses everything at once. */

const root = document.documentElement;
root.classList.add("js");
const data = JSON.parse($("site-data").textContent);
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

/* ---------- small helpers ---------- */

function h(tag, props, ...kids) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key === "vars") {
      for (const [name, v] of Object.entries(value)) node.style.setProperty(name, String(v));
    } else node.setAttribute(key, value);
  }
  for (const kid of kids.flat()) {
    if (kid !== null && kid !== undefined && kid !== false) node.append(kid);
  }
  return node;
}

const SVG_NS = "http://www.w3.org/2000/svg";
function icon(name, className) {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("class", "ic" + (className ? " " + className : ""));
  svg.setAttribute("aria-hidden", "true");
  const use = document.createElementNS(SVG_NS, "use");
  use.setAttribute("href", "#i-" + name);
  svg.append(use);
  return svg;
}

const ms = (value) => Math.round(value) + "ms";
const fmt = (n) => Math.round(n).toLocaleString("en-US");
const seconds = (value) => (value / 1000).toFixed(1);
function shorten(text, limit) {
  return text.length > limit ? text.slice(0, limit - 1).trimEnd() + "…" : text;
}
function typeLabel(type) { return (data.labels && data.labels[type]) || type; }

/* ---------- theme ---------- */

(function theme() {
  let stored = null;
  try { stored = localStorage.getItem("theme"); } catch (_) { /* private mode */ }
  if (stored === "light" || stored === "dark") root.dataset.theme = stored;
  $("theme").addEventListener("click", () => {
    const light = window.matchMedia("(prefers-color-scheme: light)").matches;
    const shown = root.dataset.theme || (light ? "light" : "dark");
    const next = shown === "light" ? "dark" : "light";
    root.dataset.theme = next;
    try { localStorage.setItem("theme", next); } catch (_) { /* private mode */ }
  });
})();

/* ---------- scroll effects: reveal, counters, bars, the nav's current section ---------- */

function whenSeen(nodes, callback, threshold) {
  if (!("IntersectionObserver" in window)) { nodes.forEach(callback); return; }
  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (entry.isIntersecting) { observer.unobserve(entry.target); callback(entry.target); }
    }
  }, { threshold: threshold || 0.15 });
  nodes.forEach((node) => observer.observe(node));
}

function countUp(node) {
  const target = Number(node.dataset.count);
  const decimals = Number(node.dataset.decimals || 0);
  const suffix = node.dataset.suffix || "";
  const text = (value) =>
    (decimals ? value.toFixed(decimals) : Math.round(value).toLocaleString("en-US")) + suffix;
  if (reduceMotion.matches || !Number.isFinite(target)) return;
  const started = performance.now();
  const duration = 1600;
  node.textContent = text(0);
  (function frame(now) {
    const progress = Math.min(1, (now - started) / duration);
    node.textContent = text(target * (1 - Math.pow(1 - progress, 3)));
    if (progress < 1) requestAnimationFrame(frame);
  })(started);
}

whenSeen([...document.querySelectorAll(".reveal")], (node) => node.classList.add("in"), 0.12);
whenSeen([...document.querySelectorAll(".stats [data-count]")], countUp, 0.6);
document.querySelectorAll(".hbars").forEach((group) => {
  whenSeen([group], () => {
    group.querySelectorAll(".hbar-fill").forEach((fill, index) => {
      fill.style.setProperty("--d", ms(index * 70));
      fill.style.setProperty("--w", String(Number(fill.dataset.w) / 100));
    });
  }, 0.3);
});

(function scrollSpy() {
  if (!("IntersectionObserver" in window)) return;
  const links = new Map();
  document.querySelectorAll(".nav-links a").forEach((a) => links.set(a.getAttribute("href").slice(1), a));
  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      links.forEach((a, id) => a.classList.toggle("active", id === entry.target.id));
    }
  }, { rootMargin: "-35% 0px -60% 0px" });
  links.forEach((_, id) => { const section = document.getElementById(id); if (section) observer.observe(section); });
})();

/* ---------- the recorded retrieval of one question ---------- */

const trace = data.trace;
const story = data.examples.find((e) => e.id === trace.question_id) || data.examples[0];
const markerOf = new Map(story.response.citations.map((c) => [c.doc_id, c.marker]));
const pool = trace.pool.map((hit) => ({
  ...hit,
  marker: markerOf.get(hit.doc_id) || null,
  origin: hit.dense_rank && hit.bm25_rank ? "both" : hit.dense_rank ? "dense" : "bm25",
}));
const byRerank = [...pool].sort((a, b) => a.rank - b.rank);
const byFusion = [...pool].sort((a, b) => a.fusion_rank - b.fusion_rank);
// The passage the story follows: a cited one that only BM25 found, as high as the reranker put it.
const spot = byRerank.find((p) => p.marker && !p.dense_rank && p.bm25_rank) || null;

function shownTitle(hit) { return hit.source_type === "slack" ? "#" + hit.title : hit.title; }
function sourceTag(type) { return h("span", { class: "src", text: type.replace("google_drive", "drive") }); }
function markerChip(hit) { return hit.marker ? h("span", { class: "mk", text: String(hit.marker) }) : h("span"); }

/* ---------- scene 1: index ---------- */

(function buildIndex() {
  const list = $("prep-sources");
  data.facts.sources.forEach(([name, count], index) => {
    list.append(h("li", { class: "a-rise", vars: { "--d": ms(index * 110) } },
      h("span", { text: name }), h("b", { text: fmt(count) })));
  });
  const grid = $("chunk-grid");
  for (let i = 0; i < 36; i++) grid.append(h("i", { class: "a-pop", vars: { "--d": ms(900 + i * 38) } }));
})();

/* ---------- scene 2: ask ---------- */

(function buildAsk() {
  const question = trace.query;
  const wanted = [
    ["dp-132-usw", "hl-exact"], ["priority=high", "hl-exact"],
    ["interactive burst credits", "hl-meaning"], ["reserved exclusively", "hl-meaning"],
  ];
  const marks = [];
  for (const [phrase, kind] of wanted) {
    const at = question.indexOf(phrase);
    if (at >= 0) marks.push({ at, end: at + phrase.length, kind });
  }
  marks.sort((a, b) => a.at - b.at);
  const box = $("ask-text");
  let cursor = 0, count = 0, lit = 0;
  const word = (text, inner) => {
    const span = h("span", { class: "w", vars: { "--d": ms(250 + count * 85) } }, inner || text);
    count += 1;
    return span;
  };
  const plain = (text) => {
    for (const piece of text.split(/\s+/).filter(Boolean)) {
      box.append(word(piece), document.createTextNode(" "));
    }
  };
  for (const mark of marks) {
    if (mark.at < cursor) continue;
    plain(question.slice(cursor, mark.at));
    box.append(word("", h("span", { class: "hl " + mark.kind,
      vars: { "--d": ms(3000 + lit * 380) }, text: question.slice(mark.at, mark.end) })),
      document.createTextNode(" "));
    lit += 1;
    cursor = mark.end;
  }
  plain(question.slice(cursor));
  box.append(h("span", { class: "ask-caret" }));
  const send = document.querySelector(".ask-send");
  send.classList.add("a-pop");
  send.style.setProperty("--d", ms(2700));
})();

/* ---------- scene 3: search two ways ---------- */

// A list row: the position in that list, then the source, the cited source's number and the
// title. `oneLine` is the wide variant that also has room for a score.
function row(rank, hit, extra, delay, oneLine) {
  const title = h("span", { class: "ttl", text: shownTitle(hit), title: shownTitle(hit) });
  const found = oneLine ? null : h("span", { class: "found", "aria-hidden": "true" },
    hit.dense_rank ? h("i", { class: "f-dense" }) : null, hit.bm25_rank ? h("i", { class: "f-bm25" }) : null);
  const attrs = { class: "row a-rise" + (oneLine ? " one" : "") + (extra || ""), vars: { "--d": ms(delay) } };
  const position = h("span", { class: "rk", text: String(rank) });
  if (oneLine) return h("li", attrs, position, sourceTag(hit.source_type), title, markerChip(hit));
  return h("li", attrs, position, h("span", { class: "meta" }, sourceTag(hit.source_type), markerChip(hit), found), title);
}

function lane(kind, title, subtitle, delay) {
  return h("section", { class: "lane lane-" + kind + " a-rise", vars: { "--d": ms(delay) } },
    h("div", { class: "lane-head" }, h("h4", null, h("i"), title), h("small", { text: subtitle })));
}

(function buildSearch() {
  const dense = lane("dense", "Dense search", "by meaning: nearest vectors", 150);
  const bm25 = lane("bm25", "BM25", "by exact words", 450);
  const fused = lane("fused", "Fused list", "reciprocal rank fusion, k = 60; 30 go on", 4300);
  const listOf = (hits, key, start, spacing) => {
    const rows = h("ol", { class: "rows" });
    hits.forEach((hit, i) => {
      rows.append(row(hit[key], hit, spot && hit === spot ? " spot" : "", start + i * spacing));
    });
    return rows;
  };
  const top = (key) => pool.filter((p) => p[key]).sort((a, b) => a[key] - b[key]).slice(0, 6);
  dense.append(listOf(top("dense_rank"), "dense_rank", 700, 170));
  bm25.append(listOf(top("bm25_rank"), "bm25_rank", 1000, 170));

  const rows = h("ol", { class: "rows" });
  const head = byFusion.slice(0, 5);
  head.forEach((hit, i) => {
    rows.append(row(hit.fusion_rank, hit, "", 4800 + i * 150));
  });
  if (spot && spot.fusion_rank > head.length) {
    rows.append(h("li", { class: "row gap a-rise", "aria-hidden": "true", text: "⋯",
      vars: { "--d": ms(5700) } }));
    rows.append(row(spot.fusion_rank, spot, " spot", 6000));
  }
  fused.append(rows);

  if (spot && spot.bm25_rank <= 6 && !spot.dense_rank) {
    bm25.append(h("p", { class: "lane-note a-rise", vars: { "--d": ms(2600) },
      text: "BM25 ranks this ticket " + ordinal(spot.bm25_rank) + ". The dense search never lists it." }));
  }
  if (spot) {
    fused.append(h("p", { class: "lane-note a-rise", vars: { "--d": ms(6600) },
      text: "Fusion leaves it " + ordinal(spot.fusion_rank) + ", too low to count on." }));
  }
  $("v-search").append(h("div", { class: "lanes" }, dense, bm25, fused));
})();

function ordinal(n) {
  const rest = n % 100;
  if (rest >= 11 && rest <= 13) return n + "th";
  return n + (["th", "st", "nd", "rd"][n % 10] || "th");
}

/* ---------- scene 4: rerank ---------- */

(function buildRerank() {
  const COLUMNS = 10;
  const scores = pool.map((p) => p.score);
  const low = Math.min(...scores), high = Math.max(...scores);
  const grid = h("div", { class: "rr-grid", "aria-hidden": "true" });
  byFusion.forEach((hit, from) => {
    const to = byRerank.indexOf(hit);
    const tile = h("div", {
      class: "wt-tile" + (to < 5 ? " top" : "") + (hit === spot ? " spot" : ""),
      vars: {
        "--c0": from % COLUMNS, "--r0": Math.floor(from / COLUMNS),
        "--c1": to % COLUMNS, "--r1": Math.floor(to / COLUMNS), "--md": ms(4300 + to * 22),
      },
    }, h("div", { class: "tile-in t-" + hit.origin, vars: { "--d": ms(from * 26), "--rd": ms(5900) } },
      h("span", { text: String(hit.fusion_rank) }),
      h("i", { class: "tile-bar", vars: { "--s": ((hit.score - low) / (high - low || 1)).toFixed(3),
        "--sd": ms(1300 + from * 62) } })));
    if (hit === spot) {
      tile.append(h("span", { class: "spot-tag a-pop", vars: { "--d": ms(6000) },
        text: "BM25's pick: " + ordinal(hit.fusion_rank) + " → " + ordinal(to + 1) }));
    }
    grid.append(tile);
  });

  const phases = [
    ["30 candidates, in fusion order", 0, 1300],
    ["the cross-encoder scores each one", 1300, 3000],
    ["re-sorted by score: the best 5 go on", 4300, 0],
  ];
  const phase = h("span", { class: "rr-phase" }, phases.map(([text, at, length]) =>
    h("span", { class: "ph", text, vars: { "--d": ms(at), "--pd": ms(length) } })));
  const key = h("div", { class: "rr-key" },
    h("span", null, h("i", { class: "k-dense" }), "found by dense search"),
    h("span", null, h("i", { class: "k-bm25" }), "found by BM25"),
    h("span", null, h("i", { class: "k-both" }), "found by both"), phase);

  const best = h("ol", { class: "rows" });
  byRerank.slice(0, 5).forEach((hit, i) => {
    const r = row(i + 1, hit, hit === spot ? " spot" : "", 7200 + i * 150, true);
    r.append(h("span", { class: "sc", text: hit.score.toFixed(1) }));
    best.append(r);
  });
  const top = h("div", { class: "rr-top" },
    h("p", { class: "lbl a-rise", vars: { "--d": ms(7000) } },
      h("span", { text: "What the model will read" }), h("span", { text: "reranker score" })), best);
  $("v-rerank").append(h("div", { class: "rr" }, key, grid, top));
})();

/* ---------- scene 5: answer ---------- */

(function buildAnswer() {
  const reply = story.response;
  const head = byRerank.slice(0, 5);
  const firstSeen = new Map();
  const text = h("p", { class: "ans-text" });
  let t = 900;
  for (const part of reply.answer_with_markers.split(/(\[\d{1,3}\])/)) {
    const marker = /^\[(\d{1,3})\]$/.exec(part);
    if (marker) {
      if (!firstSeen.has(marker[1])) firstSeen.set(marker[1], t);
      text.append(h("span", { class: "mk w", text: marker[1], vars: { "--d": ms(t) } }));
      t += 70;
    } else {
      for (const piece of part.split(/\s+/).filter(Boolean)) {
        text.append(h("span", { class: "w", text: piece, vars: { "--d": ms(t) } }), document.createTextNode(" "));
        t += 52;
      }
    }
  }
  const cards = reply.citations.map((c, i) => {
    const passages = head.filter((p) => p.doc_id === c.doc_id).length;
    return h("div", { class: "src-card pulse", vars: { "--d": ms(150 + i * 300), "--pd": ms(firstSeen.get(String(c.marker)) || 1500) } },
      h("span", { class: "mk", text: String(c.marker) }), sourceTag(c.source_type),
      h("strong", { text: c.title || "(untitled)" }),
      h("span", { text: passages + (passages === 1 ? " passage" : " passages") + " of the five" }));
  });
  const foot = h("div", { class: "ans-foot a-rise", vars: { "--d": ms(t + 300) } },
    badge(reply.status, reply.status === "answered" ? "ok" : "warn"),
    h("span", { text: reply.citations.length + " sources cited" }));
  $("v-answer").append(h("div", { class: "ans" },
    h("div", null, h("h4", { text: "Evidence given to the model" }), cards),
    h("div", null, h("h4", { text: "Answer" }), h("div", { class: "ans-card" }, text, foot))));
})();

/* ---------- scene 6: verify ---------- */

(function buildVerify() {
  const reply = story.response;
  const checks = ["Passage exists", "Numbers and names found", "Quote found in passage"];
  const box = h("div", { class: "ver" }, h("h4", { text: "Statement by statement" }));
  reply.claims.slice(0, 3).forEach((claim, i) => {
    const start = 300 + i * 2700;
    const item = h("div", { class: "ver-item a-rise", vars: { "--d": ms(start) } },
      h("p", { class: "ver-claim", text: claim.text }));
    const pills = h("div", { class: "ver-checks" });
    checks.forEach((label, k) => {
      pills.append(h("span", { class: "chk", vars: { "--d": ms(start + 600 + k * 420) } },
        h("i", { vars: { "--d": ms(start + 600 + k * 420) } }, icon("check")), label));
    });
    pills.append(h("span", { class: "chk chk-final", vars: { "--d": ms(start + 2000) } },
      h("i", { vars: { "--d": ms(start + 2000) } }, icon("check")), claim.status.replace("_", " ")));
    item.append(pills);
    if (i === 0 && claim.evidence) {
      item.append(h("p", { class: "quote a-rise", vars: { "--d": ms(start + 1900) } },
        h("q", { text: shorten(claim.evidence, 170) })));
    }
    box.append(item);
  });

  let flagged = null;
  for (const example of data.examples) {
    const claim = example.response.claims.find((c) => c.status === "partially_supported" && c.reason);
    if (claim) { flagged = { claim, example }; break; }
  }
  if (flagged) {
    box.append(h("div", { class: "ver-fail a-rise", vars: { "--d": ms(8800) } },
      h("h4", null, icon("alert"), "When a check fails"),
      h("p", { class: "ver-claim" }, badge("partially supported", "warn"), shorten(flagged.claim.text, 150)),
      h("p", { class: "why", text: "Flagged, not hidden: " + flagged.claim.reason +
        ". From the answer to a “" + typeLabel(flagged.example.type).toLowerCase() + "” question." })));
  }
  $("v-verify").append(box);
})();

/* ---------- scene 7: report ---------- */

(function buildReport() {
  const reply = story.response;
  const verification = reply.verification;
  const shown = { answered: "answered", partial: "partial", not_found: "not found", abstained: "declined" };
  const states = h("div", { class: "status-row" }, ["answered", "partial", "not_found"].map((state) =>
    h("span", { class: "status-pill" + (reply.status === state ? " is" : ""), text: shown[state] })));
  const cost = reply.usage.cost_usd === null ? "cost unknown" : "$" + reply.usage.cost_usd.toFixed(4);
  const meta = h("p", { class: "rep-meta", text: reply.citations.length + " sources · " +
    verification.verified + " of " + verification.statements + " statements verified · " + cost +
    " · " + seconds(reply.timings_ms.total) + " s on a laptop CPU" });
  const gaps = reply.gaps.length
    ? h("ul", { class: "muted" }, reply.gaps.map((g) => h("li", { text: g })))
    : h("p", { class: "muted", text: "No gaps reported for this question." });

  const stages = [
    ["Embed the query", "retrieval_embed", "var(--blue)"],
    ["Dense search", "retrieval_dense_search", "var(--blue)"],
    ["BM25 search", "retrieval_bm25_search", "var(--amber)"],
    ["Rerank", "retrieval_rerank", "var(--violet)"],
    ["Write the answer", "generation", "var(--fg-2)"],
    ["Check citations", "verification", "var(--green)"],
  ].filter(([, key]) => typeof reply.timings_ms[key] === "number");
  const longest = Math.max(1, ...stages.map(([, key]) => reply.timings_ms[key]));
  const timing = h("div", null, stages.map(([label, key, colour], i) =>
    h("div", { class: "tm a-rise", vars: { "--d": ms(900 + i * 200) } },
      h("span", { text: label }),
      h("span", { class: "track" }, h("span", { class: "fill", vars: { "--w": (reply.timings_ms[key] / longest).toFixed(3), "--c": colour, "--d": ms(1000 + i * 200) } })),
      h("span", { class: "num", text: seconds(reply.timings_ms[key]) + " s" }))));

  const facts = data.facts;
  const bar = (label, value, colour, at) => h("div", { class: "big" },
    h("span", { text: label }),
    h("span", { class: "track" }, h("span", { class: "fill", vars: { "--w": (value / 100).toFixed(2), "--c": colour, "--d": ms(at) } })),
    h("span", { class: "num", text: value + "%" }));
  const honest = h("div", { class: "rep-card honest a-rise", vars: { "--d": ms(3000) } },
    h("h4", { text: "Is the status worth reading? Share of answers judged correct" }),
    bar("said something is missing", facts.missingRight, "var(--amber)", 3400),
    bar("said nothing is missing", facts.restRight, "var(--green)", 3800),
    h("p", { text: "470 answerable benchmark questions. An answer that admits a gap is usually the wrong one: the status is a trustworthy warning." }));

  $("v-report").append(h("div", { class: "rep" },
    h("div", { class: "rep-card a-rise", vars: { "--d": ms(150) } },
      h("h4", { text: "What comes back" }), states, meta, gaps, h("h4", { text: "Where the time went" }), timing),
    honest));
})();

/* ---------- the player ---------- */

(function player() {
  const wt = $("wt");
  const stage = $("wt-stage");
  const scenes = [...stage.querySelectorAll(".wt-scene")];
  const buttons = [...document.querySelectorAll(".wt-step")];
  const playButton = $("wt-play");
  const leaving = new Map();
  let current = -1, auto = false, frozen = false, userPaused = false, started = false, ended = false;

  document.querySelectorAll("#wt [data-d]").forEach((node) => {
    node.style.setProperty("--d", ms(Number(node.dataset.d)));
    node.classList.add("a-rise");
  });

  function fit() {
    const scene = scenes[Math.max(current, 0)];
    stage.style.height = scene.offsetHeight + "px";
  }

  function setFrozen(value) {
    frozen = value;
    wt.classList.toggle("frozen", value);
  }

  function show(index) {
    scenes.forEach((scene, k) => {
      scene.classList.toggle("on", k === index);
      scene.classList.toggle("past", k < index);
      scene.setAttribute("aria-hidden", k === index ? "false" : "true");
    });
    buttons.forEach((button, k) => {
      button.classList.toggle("done", k < index);
      button.classList.toggle("current", k === index);
      if (k === index) button.setAttribute("aria-current", "step");
      else button.removeAttribute("aria-current");
    });
  }

  function go(target) {
    const index = (target + scenes.length) % scenes.length;
    const previous = current;
    current = index;
    ended = false;
    show(index);

    const scene = scenes[index];
    clearTimeout(leaving.get(index));
    scene.classList.remove("run");
    void scene.offsetWidth;
    scene.classList.add("run");
    if (previous >= 0 && previous !== index) {
      clearTimeout(leaving.get(previous));
      leaving.set(previous, setTimeout(() => scenes[previous].classList.remove("run"), 1100));
    }

    buttons.forEach((button) => button.querySelector(".wt-fill").classList.remove("go"));
    const fill = buttons[index].querySelector(".wt-fill");
    void fill.offsetWidth;
    fill.style.setProperty("--dur", ms(Number(scene.dataset.dur) || 9000));
    fill.classList.add("go");

    scene.querySelectorAll("[data-count]").forEach(countUp);
    fit();
  }

  function userGo(target) {
    if (!started) started = true;
    setFrozen(false);
    if (userPaused) auto = false;
    go(target);
  }

  function togglePlay() {
    if (userPaused) {
      userPaused = false;
      auto = true;
      setFrozen(false);
      if (ended) go(current + 1);
    } else {
      userPaused = true;
      setFrozen(true);
    }
    playButton.classList.toggle("paused", userPaused);
    playButton.setAttribute("aria-label", userPaused ? "Play the animation" : "Pause the animation");
  }

  buttons.forEach((button, k) => {
    button.addEventListener("click", () => userGo(k));
    button.querySelector(".wt-fill").addEventListener("animationend", () => {
      if (k !== current) return;
      ended = true;
      if (auto) go(k + 1);
    });
  });
  $("wt-prev").addEventListener("click", () => userGo(current - 1));
  $("wt-next").addEventListener("click", () => userGo(current + 1));
  $("wt-replay").addEventListener("click", () => userGo(current));
  playButton.addEventListener("click", togglePlay);
  wt.addEventListener("keydown", (event) => {
    if (event.key === "ArrowRight") { event.preventDefault(); userGo(current + 1); }
    if (event.key === "ArrowLeft") { event.preventDefault(); userGo(current - 1); }
  });

  // Until the walkthrough is on screen, show the first step finished; it starts from the top
  // when it comes into view, so nobody misses the beginning.
  current = 0;
  show(0);
  fit();
  if (reduceMotion.matches) {
    playButton.hidden = true;
    $("wt-hint").textContent =
      "Autoplay is off because your system asks for less motion. Use the buttons or the arrow keys.";
    go(0);
    started = true;
  } else if ("IntersectionObserver" in window) {
    new IntersectionObserver((entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          if (!started) { started = true; auto = true; go(0); }
          if (!userPaused && !document.hidden) setFrozen(false);
        } else if (started) {
          setFrozen(true);
        }
      }
    }, { threshold: 0.3 }).observe(wt);
  } else {
    auto = true;
    started = true;
    go(0);
  }
  document.addEventListener("visibilitychange", () => {
    if (!started || reduceMotion.matches) return;
    if (document.hidden) setFrozen(true);
    else if (!userPaused) setFrozen(false);
  });

  window.addEventListener("resize", fit);
  if ("ResizeObserver" in window) {
    const watcher = new ResizeObserver(fit);
    scenes.forEach((scene) => watcher.observe(scene));
  }
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(fit);
})();

/* ---------- recorded answers ---------- */

(function explorer() {
  const list = $("ex-list");
  const tabs = data.examples.map((example, index) => {
    const answered = example.response.status === "answered";
    const tab = h("button", { class: "ex-btn", type: "button", role: "tab", "aria-selected": "false",
      "aria-controls": "ex-panel", id: "ex-tab-" + index },
      h("span", { class: "tp" }, h("i", { class: "dot " + (answered ? "dot-green" : "dot-amber") }),
        typeLabel(example.type)),
      h("span", { class: "tx", text: example.question }));
    tab.addEventListener("click", () => select(index));
    tab.addEventListener("keydown", (event) => {
      const step = { ArrowDown: 1, ArrowRight: 1, ArrowUp: -1, ArrowLeft: -1 }[event.key];
      if (!step) return;
      event.preventDefault();
      const next = (index + step + tabs.length) % tabs.length;
      tabs[next].focus();
      select(next);
    });
    list.append(tab);
    return tab;
  });

  function select(index) {
    const example = data.examples[index];
    const body = example.response;
    tabs.forEach((tab, k) => {
      tab.setAttribute("aria-selected", k === index ? "true" : "false");
      tab.tabIndex = k === index ? 0 : -1;
    });
    $("ex-panel").setAttribute("aria-labelledby", "ex-tab-" + index);
    $("ex-type").textContent = typeLabel(example.type);
    $("question").textContent = example.question;
    $("note").textContent = example.note || "";
    const result = $("result");
    result.replaceChildren(
      el("h2", "Answer"), renderAnswer(body),
      el("h2", "Sources"), renderSources(body),
      el("h2", "Statement by statement"), renderClaims(body),
      el("h2", "Cost and time"), renderCost(body));
    result.classList.remove("swap");
    void result.offsetWidth;
    result.classList.add("swap");
  }
  select(0);
})();

/* ---------- copy the commands ---------- */

$("copy").addEventListener("click", async () => {
  const label = $("copy-label");
  try {
    await navigator.clipboard.writeText($("cmds").textContent);
    label.textContent = "Copied";
  } catch (_) {
    const range = document.createRange();
    range.selectNodeContents($("cmds"));
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    label.textContent = "Press Ctrl+C";
  }
  setTimeout(() => { label.textContent = "Copy"; }, 2000);
});
