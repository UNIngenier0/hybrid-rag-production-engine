"""Build the static site that GitHub Pages serves.

    uv run python scripts/build_site.py --out _site

The page is made from files that are committed: the recorded responses (site/examples.json, from
scripts/record_examples.py) and the published benchmark and abstention reports. Building it again
after any of them changes keeps the site's numbers equal to the repository's.
"""

import argparse
import json
from pathlib import Path

from rag_engine.site_page import render_site

REPOSITORY = "https://github.com/UNIngenier0/hybrid-rag-production-engine"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, default=Path("site/examples.json"))
    parser.add_argument("--report", type=Path, default=Path("eval/reports/full-benchmark.json"))
    parser.add_argument(
        "--abstention", type=Path, default=Path("eval/reports/abstention-scores.json")
    )
    parser.add_argument("--repo-url", default=REPOSITORY)
    parser.add_argument("--out", type=Path, default=Path("_site"))
    args = parser.parse_args()

    def load(path: Path) -> dict:  # type: ignore[type-arg]
        return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]

    page = render_site(
        load(args.recording), load(args.report), load(args.abstention), args.repo_url
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "index.html").write_text(page, encoding="utf-8")
    # Pages would otherwise run the files through Jekyll, which this site does not need.
    (args.out / ".nojekyll").write_text("", encoding="utf-8")
    print(f"wrote {args.out / 'index.html'} ({len(page):,} characters)")


if __name__ == "__main__":
    main()
