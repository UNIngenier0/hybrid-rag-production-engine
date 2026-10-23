"""Build the static site that GitHub Pages serves.

    uv run python scripts/build_site.py --out _site

The page is made from files that are committed: the front-end files in site/ (template.html,
page.css, page.js), the recorded responses (site/examples.json, from scripts/record_examples.py),
the recorded retrieval trace (site/trace.json, from scripts/record_trace.py) and the published
benchmark, ablation, comparison, citation-check and abstention reports. Building it again after
any of them changes keeps the site's numbers equal to the repository's.
"""

import argparse
import json
import shutil
from pathlib import Path

from rag_engine.site_page import render_site

REPOSITORY = "https://github.com/UNIngenier0/hybrid-rag-production-engine"
SITE_URL = "https://uningenier0.github.io/hybrid-rag-production-engine/"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, default=Path("site"))
    parser.add_argument("--reports", type=Path, default=Path("eval/reports"))
    parser.add_argument("--repo-url", default=REPOSITORY)
    parser.add_argument("--site-url", default=SITE_URL, help="where the page will be served")
    parser.add_argument("--out", type=Path, default=Path("_site"))
    args = parser.parse_args()

    def load(path: Path) -> dict:  # type: ignore[type-arg]
        return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]

    page = render_site(
        load(args.site / "examples.json"),
        load(args.site / "trace.json"),
        load(args.reports / "full-benchmark.json"),
        load(args.reports / "abstention-scores.json"),
        load(args.reports / "retrieval-ablation.json"),
        load(args.reports / "answer-comparison.json"),
        load(args.reports / "citation-verification.json"),
        args.repo_url,
        site_url=args.site_url,
        site_dir=args.site,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "index.html").write_text(page, encoding="utf-8")
    image = args.site / "og.png"
    if image.exists():
        shutil.copyfile(image, args.out / "og.png")
    # Pages would otherwise run the files through Jekyll, which this site does not need.
    (args.out / ".nojekyll").write_text("", encoding="utf-8")
    print(f"wrote {args.out / 'index.html'} ({len(page):,} characters)")


if __name__ == "__main__":
    main()
