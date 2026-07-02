"""Extract the list of Gutenberg eBook IDs referenced by an existing corpus file.

Writes manifest.json as a flat JSON list of integer eBook IDs (e.g. [84, 43, 35, ...]).
Titles are deliberately not extracted here: the source corpus text may be of
untrustworthy provenance (e.g. a mis-encoded/corrupted download), and only the
tightly bounded `[eBook #N]` ID pattern is safe to run against such text. Titles
for these IDs should instead come from freshly downloaded, verified Gutenberg
sources in a later step.
"""
import argparse
import json
from pathlib import Path

from gutenberg_utils import parse_gutenberg_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Path to the existing raw corpus text file")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "manifest.json")
    args = parser.parse_args()

    text = args.source.read_text(encoding="utf-8", errors="replace")
    ids = parse_gutenberg_ids(text)

    args.out.write_text(json.dumps(ids, indent=2), encoding="utf-8")
    print(f"Wrote {len(ids)} eBook IDs to {args.out}")


if __name__ == "__main__":
    main()
