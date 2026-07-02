"""Rebuild sci_fi_corpus.txt from Project Gutenberg, using manifest.json (a list of Gutenberg
eBook IDs) as the book list. Each book's title is read from its own freshly-downloaded text
for progress logging — manifest.json intentionally carries no title field (see Task 3: the
original local corpus file titles were extracted from was mis-encoded and unusable).

Network-dependent; not covered by unit tests. gutenberg_utils.strip_gutenberg_boilerplate
is tested separately with fixture text in test_gutenberg_utils.py.
"""
import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

from gutenberg_utils import strip_gutenberg_boilerplate

HERE = Path(__file__).parent
GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
TITLE_RE = re.compile(r"Title:\s*(.+)")


def download_book(gutenberg_id: int) -> str:
    url = GUTENBERG_URL.format(id=gutenberg_id)
    with urllib.request.urlopen(url, timeout=30) as response:
        raw = response.read()
    return raw.decode("utf-8", errors="replace")


def extract_title(raw_text: str, fallback: str) -> str:
    match = TITLE_RE.search(raw_text)
    return match.group(1).strip() if match else fallback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=HERE / "manifest.json")
    parser.add_argument("--out", type=Path, default=HERE / "sci_fi_corpus.txt")
    parser.add_argument("--force", action="store_true", help="Re-download even if output already exists")
    args = parser.parse_args()

    if args.out.exists() and not args.force:
        print(f"{args.out} already exists, skipping download (use --force to rebuild)")
        return

    gutenberg_ids = json.loads(args.manifest.read_text(encoding="utf-8"))
    bodies = []
    for gid in gutenberg_ids:
        try:
            raw = download_book(gid)
        except Exception as exc:
            print(f"#{gid}: skipped ({exc})")
            continue
        title = extract_title(raw, fallback=f"eBook #{gid}")
        print(f"Downloaded #{gid}: {title}")
        bodies.append(strip_gutenberg_boilerplate(raw))
        time.sleep(0.5)  # be polite to gutenberg.org

    args.out.write_text("\n\n".join(bodies), encoding="utf-8")
    print(f"Wrote {len(bodies)} books ({args.out.stat().st_size:,} bytes) to {args.out}")


if __name__ == "__main__":
    main()
