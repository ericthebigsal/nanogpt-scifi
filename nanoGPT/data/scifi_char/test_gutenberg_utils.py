from gutenberg_utils import parse_gutenberg_ids, strip_gutenberg_boilerplate

SAMPLE_HEADER = """The Project Gutenberg eBook of Sample Tale

Title: Sample Tale

Author: A. N. Author

Release date: January 1, 2000 [eBook #123]

Language: English

*** START OF THE PROJECT GUTENBERG EBOOK SAMPLE TALE ***

This is the body of the book. It has two lines.
Second line here.

*** END OF THE PROJECT GUTENBERG EBOOK SAMPLE TALE ***

Some trailing license text that should be discarded.
"""


def test_parse_gutenberg_ids_extracts_ids_in_order():
    combined = SAMPLE_HEADER + SAMPLE_HEADER.replace("123", "456")
    assert parse_gutenberg_ids(combined) == [123, 456]


def test_parse_gutenberg_ids_dedupes_by_id():
    combined = SAMPLE_HEADER + SAMPLE_HEADER
    assert parse_gutenberg_ids(combined) == [123]


def test_parse_gutenberg_ids_survives_missing_newlines_between_ids():
    # Regression test: a real Gutenberg corpus file was corrupted such that it was
    # the literal repr() of bytes objects concatenated together, so it contained
    # almost no real newline characters for megabytes at a stretch (only literal
    # backslash-r-backslash-n text sequences). An unbounded regex like
    # `Title:\s*(.+)` would run for megabytes before finding a real '\n' to stop
    # at, producing garbage multi-megabyte "titles". ID_RE's bounded pattern
    # (`\[eBook #`, digits, `\]`) should be unaffected by the missing newlines
    # since it doesn't rely on '.' stopping at line boundaries.
    filler = r"random filler text \r\n more literal backslash r backslash n text " * 500
    assert "\n" not in filler
    text = f"junk before [eBook #7]{filler}more junk [eBook #99] trailing junk"
    assert parse_gutenberg_ids(text) == [7, 99]


def test_strip_gutenberg_boilerplate_keeps_only_body():
    body = strip_gutenberg_boilerplate(SAMPLE_HEADER)
    assert body == "This is the body of the book. It has two lines.\nSecond line here."


def test_strip_gutenberg_boilerplate_falls_back_to_full_text_without_markers():
    text = "no markers here, just plain text"
    assert strip_gutenberg_boilerplate(text) == text
