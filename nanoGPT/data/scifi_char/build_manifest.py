"""Extract {gutenberg_id, title} pairs from an existing Gutenberg-derived corpus file."""
import argparse
import json
from pathlib import Path

from gutenberg_utils import parse_manifest_entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Path to the existing raw corpus text file")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "manifest.json")
    args = parser.parse_args()

    text = args.source.read_text(encoding="utf-8", errors="replace")
    entries = parse_manifest_entries(text)

    args.out.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    print(f"Wrote {len(entries)} entries to {args.out}")


if __name__ == "__main__":
    main()
