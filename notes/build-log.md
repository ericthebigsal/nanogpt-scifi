# Build Log

A running, chronological record of this project as it was actually built — decisions,
dead ends, and what each step taught, in the order it happened. This is deliberately
separate from [`pipeline-notes.md`](pipeline-notes.md) (not yet written — see Task 8 of
the [implementation plan](../docs/superpowers/plans/2026-07-02-nanogpt-scifi-implementation.md)),
which will be a polished, thesis-style technical report on the finished pipeline. This
document is the opposite of polished on purpose: the mistakes and course-corrections
below are part of the point of a from-scratch project like this one, not noise to edit
out.

Each entry follows the same shape: **what** was done, **why**, and **what it taught**.

---

## Task 1 — Repo scaffolding

**What:** Created `README.md`, `LICENSE` (MIT), `.gitignore`, and `requirements.txt`;
built a Python 3.14 virtual environment (`.venv/`) and installed `torch>=2.12`, `numpy`,
`tqdm`, and `pytest`.

**Why:** Every later task depends on a working, reproducible environment. Pinning to
Python 3.14 and `torch>=2.12` mattered specifically because this project runs on
Apple Silicon (M3) rather than the CUDA hardware nanoGPT assumes by default. A
[wheel](https://packaging.python.org/en/latest/specifications/binary-distribution-format/)
(`.whl` file) is Python's prebuilt binary package format — `pip install` prefers one
over building a package from source, but a wheel is only usable if someone has
published one matching both your Python version and your CPU architecture/OS (encoded
in its filename as a "platform tag," e.g. `cp314-cp314-macosx_14_0_arm64`: CPython
3.14, macOS 14+, Apple Silicon). PyTorch's C++/CUDA internals make building it from
source painful, so this project's viability depended on the PyTorch team having
already published a `cp314`-tagged, `arm64` wheel; without one, `pip install torch` would
either fail outright or silently fall back to a slow, from-source build. That wheel's
existence was confirmed before writing any model code (`torch 2.12.1
cp314-cp314-macosx_14_0_arm64`, per Task 1's installed-dependency list below).

**Learned:** Confirmed `torch.backends.mps.is_available()` returns `True` on this
machine before writing a single line of model code — cheap to check up front, expensive
to discover wrong after Task 6's training run.

_Commit: `24dfe29`_

---

## Task 2 — Vendor nanoGPT

**What:** Cloned [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) (MIT license)
into `nanoGPT/`, stripped its `.git` directory, and committed it as 26 plain tracked
files.

**Why:** The goal of this project is to understand the training pipeline end-to-end —
tokenization, attention, the training loop — not to reinvent a transformer
implementation. Vendoring a known-good, minimal reference implementation keeps the
learning focused on the *pipeline* (data → tokenizer → config → training → sampling)
while treating the model architecture itself as a well-understood building block.

**Learned:** Vendoring plain files (not a git submodule) was a deliberate choice —
submodules add a layer of git indirection that a reader of this portfolio repo would
have to understand just to clone it. A `git ls-tree` check confirmed every file landed
at mode `100644` (plain file), not `160000` (gitlink), before committing.

_Commit: `ed87214`_

---

## Task 3 — Gutenberg manifest extraction (and a real bug)

**What:** Wrote `gutenberg_utils.py` (regex-based helpers to parse a Gutenberg-derived
manifest and to strip Gutenberg's boilerplate header/footer from a book's raw text) and
`build_manifest.py`, following TDD — four failing tests written first, then the
implementation to turn them green. Ran `build_manifest.py` against a pre-existing local
file, `sci_fi_corpus.txt`, that was believed to already contain the text of ~97
public-domain sci-fi books, to produce `manifest.json`.

**Why:** Before fetching anything from the network, we needed a definitive list of
*which* ~97 books to fetch — the local file was assumed to be a usable source for both
the list of book IDs and their titles.

**What went wrong:** `manifest.json` came out at 36MB — for a metadata file that should
have been a few hundred bytes. The `Title:\s*(.+)` regex used to pull each book's title
was unbounded (`.+` stops only at a real newline), and the local source file turned out
to be corrupted: it was the literal Python `repr()` of `bytes` objects concatenated
together, containing almost no actual newline characters for megabytes at a stretch —
only literal two-character `\r`/`\n` *text*, not real line breaks. Each "title" match
ran for up to ~1.5MB before finding a real newline to stop at, producing garbage
multi-megabyte "titles" for all 97 entries.

**The fix:** Rather than patch the regex, we cut scope. The `[eBook #(\d+)]` ID
pattern is *tightly bounded* by fixed literal text on both sides, so — unlike the
title regex — it isn't vulnerable to the missing-newline problem and its output stayed
trustworthy even against the corrupted source. `parse_manifest_entries()` was deleted
outright (not patched) and replaced with `parse_gutenberg_ids(text) -> list[int]`, which
extracts *only* IDs. `manifest.json` became a flat list of 97 integers, with title data
punted entirely to Task 4 (sourced fresh from each book's own clean download). A
regression test (`test_parse_gutenberg_ids_survives_missing_newlines_between_ids`) was
added that explicitly asserts its fixture text contains zero real newline characters
before running the parser against it — proving the fix addresses the actual failure
mode, not just the symptom.

**Learned:** Two lessons worth keeping. First, don't trust a "wait, this output is
suspiciously huge" instinct less than an "it ran without error" signal — the original
implementation passed its own hand-written tests fine, because those tests used
clean fixture text; the corruption only showed up against the real file. Second, when
a regex fails on messy real-world input, ask whether the fix is "make the regex
smarter" or "reduce what the regex is trusted to do" — here, shrinking scope
(IDs only, no titles) was more robust than trying to out-clever a corrupted source.

_Commits: `cdc1fd0` (initial, buggy), `c87cf45` (fix)_

---

## Task 4 — Corpus fetch script

**What:** Wrote `fetch_corpus.py` to download all 97 books from Project Gutenberg
individually (by ID, from `manifest.json`), strip each one's boilerplate, extract its
title from its own clean download (for progress logging only — never from the corrupted
local file), and concatenate everything into `sci_fi_corpus.txt`.

**Why:** Task 3's fix meant `manifest.json` no longer carried titles, and — more
importantly — meant no committed code would ever read the corrupted local file again.
Fetching each book fresh, individually, sidesteps the corruption entirely: a single
concatenated 33MB local file failed in a way that 97 independently-downloaded clean
files don't.

**Learned:** Verified the download + title-extraction path at small scale (2 books:
Frankenstein and The Time Machine, sanity-checked for the word "Victor" and a minimum
byte count) *before* running the full 97-book fetch — cheap insurance against
discovering a bug 90 books into a multi-minute network-bound run. The full fetch
succeeded on all 97 books with zero skips, producing a clean 28MB corpus (`29,804,783`
bytes) in roughly 7–8 minutes. Titles surfaced in the run log were short and legitimate
(e.g. "The war of the worlds", "R.U.R. (Rossum's Universal Robots)", plus several
non-English titles like "Voyage au Centre de la Terre") — a visible, eyeballable
confirmation that Task 3's corruption problem was fully behind us.

_Commit: `938f6c4`_

---

## Task 5 — Char-level tokenizer prep

**What:** Wrote `prepare.py` (again via TDD) implementing `build_vocab`/`encode`/`decode`
following nanoGPT's own `shakespeare_char/prepare.py` convention, then ran it against
the real 28MB corpus to produce `train.bin`, `val.bin`, and `meta.pkl`.

**Why:** Character-level tokenization (not BPE/`tiktoken`) was a deliberate choice for
this project — it keeps the tokenizer itself trivial to fully understand (a sorted-set
vocabulary and two dict lookups) so that the learning effort goes into the transformer
and training loop, not into subword tokenization mechanics. That tradeoff is explicitly
revisited in Task 8/9's write-up.

**Learned:** The real corpus produced a **251-character vocabulary** — larger than
plain ASCII prose would need, because the corpus includes non-English titles/passages
(accented Latin characters, Greek letters, even runes) pulled in by the Gutenberg
source texts. That's a concrete, verifiable number to point to in a portfolio
write-up rather than an assumed "vocab size ~65" from a typical English-only corpus.
Final split: 28,934,015 characters total → 26,040,613 train tokens / 2,893,402 val
tokens (90/10).

_Commit: `edb0341`_

---

## Interlude — machine reboot mid-Task 6

**What happened:** Between Task 5 and Task 6, an OS update rebooted the machine
mid-session. On resume: `git log` and the `.superpowers/sdd/progress.md` ledger
confirmed Tasks 1–5 were fully committed and untouched; `nanoGPT/data/scifi_char/`
still had `train.bin`/`val.bin`/`meta.pkl` on disk (gitignored but never deleted);
`nanoGPT/config/train_scifi_char.py` existed on disk but was still untracked — Task 6
had been started (the config written) but the actual training run had not yet
happened (`nanoGPT/out-scifi-char/` was empty). `torch.backends.mps.is_available()`
was re-verified `True` post-reboot before resuming.

**Why this is worth logging:** A reboot mid-project is exactly the kind of ordinary
interruption a from-scratch training project runs into on laptop hardware (as opposed
to a managed cloud job) — and it's a real test of whether the project's own
bookkeeping (git commits + the plan's progress ledger) is trustworthy enough to resume
from cold, rather than needing to guess or re-derive what already happened.

**Learned:** The commit-per-task discipline paid for itself here — recovery was a
matter of reading `git log --oneline` and the progress ledger, not re-inspecting file
timestamps to reconstruct state.

---

## Task 6 — Training config and baseline run

_In progress — this section will be filled in with the actual run once training
completes (config: 6 layers, 6 heads, 384-dim embeddings, block size 256, batch size
64, `mps` backend, 3000 iterations)._

---

_This log is updated as work progresses; see `git log` for the authoritative,
timestamped record of every change described above._
