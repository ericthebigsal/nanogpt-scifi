"""Pure text-processing helpers for building the sci-fi corpus from Project Gutenberg sources."""
import re

ID_RE = re.compile(r"\[eBook #(\d+)\]")
START_RE = re.compile(r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE | re.DOTALL)
END_RE = re.compile(r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE | re.DOTALL)


def parse_gutenberg_ids(text: str) -> list[int]:
    """Extract the deduped list of `[eBook #N]` IDs from text, in order of first appearance.

    Deliberately does NOT attempt to extract titles: this function may be run against
    source text of untrustworthy provenance (e.g. mis-encoded/corrupted downloads), and
    ID_RE's tightly bounded pattern (`\\[eBook #`, digits, `\\]`) is safe to run on such
    text where an unbounded pattern like `Title:\\s*(.+)` would not be, since `.` only
    stops at real newline characters and corrupted sources may go megabytes without one.
    """
    seen_ids = set()
    ordered_ids = []
    for match in ID_RE.finditer(text):
        gid = int(match.group(1))
        if gid not in seen_ids:
            seen_ids.add(gid)
            ordered_ids.append(gid)
    return ordered_ids


def strip_gutenberg_boilerplate(text: str) -> str:
    """Return only the book body between the Gutenberg START/END markers."""
    start_match = START_RE.search(text)
    end_match = END_RE.search(text)
    if not start_match or not end_match:
        return text.strip()
    return text[start_match.end():end_match.start()].strip()
