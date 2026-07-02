"""Pure text-processing helpers for building the sci-fi corpus from Project Gutenberg sources."""
import re

TITLE_RE = re.compile(r"Title:\s*(.+)")
ID_RE = re.compile(r"\[eBook #(\d+)\]")
START_RE = re.compile(r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE | re.DOTALL)
END_RE = re.compile(r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE | re.DOTALL)


def parse_manifest_entries(text: str) -> list[dict]:
    """Pair each 'Title: ...' header line with the eBook ID that follows it, deduped by ID."""
    id_positions = [(m.start(), m.group(1)) for m in ID_RE.finditer(text)]
    entries = []
    seen_ids = set()
    for title_match in TITLE_RE.finditer(text):
        title = title_match.group(1).strip()
        title_pos = title_match.start()
        next_id = next((gid for pos, gid in id_positions if pos > title_pos), None)
        if next_id is None or next_id in seen_ids:
            continue
        seen_ids.add(next_id)
        entries.append({"gutenberg_id": int(next_id), "title": title})
    return entries


def strip_gutenberg_boilerplate(text: str) -> str:
    """Return only the book body between the Gutenberg START/END markers."""
    start_match = START_RE.search(text)
    end_match = END_RE.search(text)
    if not start_match or not end_match:
        return text.strip()
    return text[start_match.end():end_match.start()].strip()
