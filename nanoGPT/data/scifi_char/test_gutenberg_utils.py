from gutenberg_utils import parse_manifest_entries, strip_gutenberg_boilerplate

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


def test_parse_manifest_entries_extracts_id_and_title():
    entries = parse_manifest_entries(SAMPLE_HEADER)
    assert entries == [{"gutenberg_id": 123, "title": "Sample Tale"}]


def test_strip_gutenberg_boilerplate_keeps_only_body():
    body = strip_gutenberg_boilerplate(SAMPLE_HEADER)
    assert body == "This is the body of the book. It has two lines.\nSecond line here."


def test_parse_manifest_entries_dedupes_by_id():
    combined = SAMPLE_HEADER + SAMPLE_HEADER
    entries = parse_manifest_entries(combined)
    assert entries == [{"gutenberg_id": 123, "title": "Sample Tale"}]


def test_strip_gutenberg_boilerplate_falls_back_to_full_text_without_markers():
    text = "no markers here, just plain text"
    assert strip_gutenberg_boilerplate(text) == text
