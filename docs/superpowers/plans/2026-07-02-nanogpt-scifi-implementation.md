# nanogpt-scifi Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and train a small character-level GPT on a public-domain sci-fi corpus, end-to-end on local Apple M3 hardware, packaged as a public GitHub portfolio repo.

**Architecture:** Vendor Karpathy's nanoGPT as the training engine. Add a dataset pipeline under `nanoGPT/data/scifi_char/` (Gutenberg manifest → corpus fetch → char-level tokenization) that mirrors nanoGPT's own per-dataset convention. Train with a laptop-scale config on the `mps` backend, then generate samples and write up results.

**Tech Stack:** Python 3.14, PyTorch 2.12.1 (mps backend), NumPy, pytest, git, GitHub CLI (`gh`).

## Global Constraints

- Python 3.14 venv; torch 2.12.1 has confirmed cp314 wheels for macOS arm64 — no version downgrade needed.
- Training device is `mps` (Apple M3 GPU); no cloud compute.
- Char-level tokenization only — no `tiktoken`/BPE, no `transformers`/`datasets`/`wandb` dependencies (unused in this path).
- Corpus (`sci_fi_corpus.txt`) and all generated data/checkpoints are gitignored, not committed — repo must be reproducible via `fetch_corpus.py` + `manifest.json` alone.
- nanoGPT vendored as plain files (its `.git` removed), not a submodule; credited in root README, its own `LICENSE` kept alongside.
- GitHub repo: public, under `ericthebigsal`, named `nanogpt-scifi`.
- git identity for this repo is already set locally: `Eric Salerno <esalerno86@gmail.com>`.
- **Amendment after Task 3:** `/Users/ericsalerno/Documents/llm-learning/sci_fi_corpus.txt` turned out to be mis-encoded (literal Python `bytes.repr()` text, ~0 real newline characters) — unusable as a source of body text or titles. It was used ONLY to extract the list of 97 Gutenberg eBook IDs (via the bounded `[eBook #N]` pattern, which survived the corruption). `manifest.json` is now `[int, ...]` — a plain list of Gutenberg IDs, no `title` field. The corpus body text and book titles are both sourced fresh from Project Gutenberg in Task 4 — the corrupted local file is not used for anything beyond Task 3 and is never referenced by committed code again.

---

### Task 1: Repo scaffolding

**Files:**
- Create: `README.md`
- Create: `LICENSE`
- Create: `.gitignore`
- Create: `requirements.txt`

**Interfaces:**
- Produces: a venv at `.venv/` with all dependencies installed, used by every later task.

- [ ] **Step 1: Write `.gitignore`**

```
.venv/
__pycache__/
*.pyc
.DS_Store
nanoGPT/data/scifi_char/sci_fi_corpus.txt
nanoGPT/data/scifi_char/train.bin
nanoGPT/data/scifi_char/val.bin
nanoGPT/data/scifi_char/meta.pkl
nanoGPT/out-scifi-char/
```

- [ ] **Step 2: Write `requirements.txt`**

```
torch>=2.12
numpy
tqdm
pytest
```

- [ ] **Step 3: Write `LICENSE`** (MIT, code only — the corpus is separately public-domain Gutenberg text, noted in README)

```
MIT License

Copyright (c) 2026 Eric Salerno

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 4: Write `README.md` skeleton** (Results/Samples sections filled in during Task 9)

```markdown
# nanogpt-scifi

A character-level GPT trained from scratch on a ~33MB corpus of public-domain
science fiction (Project Gutenberg), built to understand the transformer
training pipeline end-to-end — tokenization, self-attention, training loop,
loss curves — not just prompt around it.

Training engine: [nanoGPT](https://github.com/karpathy/nanoGPT) (Karpathy, MIT
license), vendored in `nanoGPT/`. Dataset: 98 public-domain works (Frankenstein,
The Time Machine, The War of the Worlds, Flatland, R.U.R., and others),
reproducible from source via the scripts below — the corpus itself is not
committed to this repo.

## Setup

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Rebuild the dataset

```bash
cd nanoGPT/data/scifi_char
python fetch_corpus.py      # downloads all 98 books from Project Gutenberg
python prepare.py           # tokenizes to train.bin / val.bin / meta.pkl
```

## Train

```bash
cd nanoGPT
python train.py config/train_scifi_char.py
```

## Sample

```bash
cd nanoGPT
python sample.py --out_dir=out-scifi-char
```

## Results

_(filled in after training — see Task 9)_

## What this demonstrates

_(filled in after training — see Task 9)_
```

- [ ] **Step 5: Create the venv and install dependencies**

Run: `cd /Users/ericsalerno/Documents/llm-learning/nanogpt-scifi && /opt/homebrew/bin/python3.14 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
Expected: torch, numpy, tqdm, pytest install without errors; `python -c "import torch; print(torch.backends.mps.is_available())"` prints `True`.

- [ ] **Step 6: Commit**

```bash
git add README.md LICENSE .gitignore requirements.txt
git commit -m "Scaffold repo: README, LICENSE, gitignore, requirements"
```

---

### Task 2: Vendor nanoGPT

**Files:**
- Create: `nanoGPT/` (cloned from `karpathy/nanoGPT`, `.git` removed)
- Modify: `README.md` (attribution already present from Task 1; verify link is correct)

**Interfaces:**
- Produces: `nanoGPT/train.py`, `nanoGPT/sample.py`, `nanoGPT/model.py`, `nanoGPT/configurator.py`, `nanoGPT/config/` — used by Tasks 6–7.

- [ ] **Step 1: Clone and de-git nanoGPT**

Run:
```bash
git clone --depth 1 https://github.com/karpathy/nanoGPT.git nanoGPT
rm -rf nanoGPT/.git
```
Expected: `nanoGPT/train.py`, `nanoGPT/model.py`, `nanoGPT/sample.py`, `nanoGPT/config/`, `nanoGPT/LICENSE` all present; no `.git` inside `nanoGPT/`.

- [ ] **Step 2: Verify it's tracked as plain files, not a submodule**

Run: `git status`
Expected: `nanoGPT/` shows as a large set of new untracked files (not a single "new commit" gitlink entry). If it shows as a gitlink (mode `160000`), the `.git` removal in Step 1 didn't take — redo it.

- [ ] **Step 3: Commit**

```bash
git add nanoGPT/
git commit -m "Vendor nanoGPT (karpathy/nanoGPT, MIT license)"
```

---

### Task 3: Gutenberg manifest extraction

**Files:**
- Create: `nanoGPT/data/scifi_char/gutenberg_utils.py`
- Create: `nanoGPT/data/scifi_char/build_manifest.py`
- Create: `nanoGPT/data/scifi_char/manifest.json` (committed — small metadata file, not the corpus)
- Test: `nanoGPT/data/scifi_char/test_gutenberg_utils.py`

**Interfaces:**
- Produces: `parse_manifest_entries(text: str) -> list[dict]` (each dict: `{"gutenberg_id": int, "title": str}`), `strip_gutenberg_boilerplate(text: str) -> str` — both consumed by Task 4's `fetch_corpus.py`.
- Produces: `manifest.json` — list of `{"gutenberg_id": int, "title": str}` — consumed by Task 4.

**ACTUAL result (post-fix, see commit `c87cf45` on top of `cdc1fd0`):** the local source file turned out to be mis-encoded (see Global Constraints amendment above). `parse_manifest_entries` was replaced with `parse_gutenberg_ids(text: str) -> list[int]` (title-pairing removed entirely, not patched — the source is untrustworthy for any unbounded pattern). `manifest.json` is `[int, ...]`, not `[{gutenberg_id, title}, ...]`. `strip_gutenberg_boilerplate` is unaffected and unchanged. Task 4 below is rewritten to match this actual interface.

- [ ] **Step 1: Write the failing tests**

```python
# nanoGPT/data/scifi_char/test_gutenberg_utils.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd nanoGPT/data/scifi_char && python -m pytest test_gutenberg_utils.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gutenberg_utils'`

- [ ] **Step 3: Write `gutenberg_utils.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest test_gutenberg_utils.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Write `build_manifest.py`**

```python
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
```

- [ ] **Step 6: Run it against the existing local corpus to generate `manifest.json`**

Run: `python build_manifest.py /Users/ericsalerno/Documents/llm-learning/sci_fi_corpus.txt`
Expected: prints `Wrote N entries to .../manifest.json` where N is around 97; `manifest.json` created in `nanoGPT/data/scifi_char/`.

- [ ] **Step 7: Commit**

```bash
git add nanoGPT/data/scifi_char/gutenberg_utils.py nanoGPT/data/scifi_char/build_manifest.py nanoGPT/data/scifi_char/manifest.json nanoGPT/data/scifi_char/test_gutenberg_utils.py
git commit -m "Add Gutenberg manifest extraction with tests"
```

---

### Task 4: Corpus fetch script

**REWRITTEN after Task 3's fix** (see Global Constraints amendment): `manifest.json` is now `[int, ...]` (Gutenberg IDs only, no titles — the local corrupted file is never touched by this task or any committed code again). Since each book is downloaded individually as its own clean, well-formed text, it's safe to extract that book's title directly from its own freshly-downloaded text for progress-logging purposes — the corruption problem was specific to the old concatenated local file, not to individual clean downloads. There is no "bootstrap from local file" step anymore — this task performs the real, full download of all ~97 books from Project Gutenberg.

**Files:**
- Create: `nanoGPT/data/scifi_char/fetch_corpus.py`

**Interfaces:**
- Consumes: `nanoGPT/data/scifi_char/manifest.json` (Task 3, now `[int, ...]`), `gutenberg_utils.strip_gutenberg_boilerplate` (Task 3, unchanged).
- Produces: `nanoGPT/data/scifi_char/sci_fi_corpus.txt` — consumed by Task 5's `prepare.py`.

- [ ] **Step 1: Write `fetch_corpus.py`**

```python
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
```

- [ ] **Step 2: Verify the download + title-extraction path works at small scale**

Run:
```bash
python -c "
import json, tempfile
from pathlib import Path
manifest = [84, 35]  # Frankenstein, The Time Machine
tmp = Path(tempfile.mkdtemp())
(tmp / 'manifest.json').write_text(json.dumps(manifest))
import subprocess
result = subprocess.run(['python', 'fetch_corpus.py', '--manifest', str(tmp/'manifest.json'), '--out', str(tmp/'out.txt')], check=True, capture_output=True, text=True)
print(result.stdout)
text = (tmp / 'out.txt').read_text()
assert 'Victor' in text, 'expected Frankenstein content not found'
assert len(text) > 100_000, f'unexpectedly small output: {len(text)} chars'
print('OK:', len(text), 'chars downloaded')
"
```
Expected: stdout shows lines like `Downloaded #84: Frankenstein; Or, The Modern Prometheus` (a short, correct title — not megabytes of garbage), then `OK: <large number> chars downloaded`. This confirms both the download/boilerplate-stripping path AND the new per-book title extraction work correctly on real Gutenberg text before running against all 97 books.

- [ ] **Step 3: Run the real, full fetch (all ~97 books, live network)**

Run: `cd nanoGPT/data/scifi_char && python fetch_corpus.py`
Expected: ~97 `Downloaded #N: <title>` lines with short, sane-looking titles (sanity-check a handful by eye — no multi-thousand-character titles), a few possible `skipped (...)` lines are acceptable (network hiccups), ending with `Wrote N books (X bytes) to .../sci_fi_corpus.txt` where X is in the tens-of-megabytes range. Takes a few minutes (network-bound, ~0.5s/book minimum).

- [ ] **Step 4: Confirm idempotency**

Run: `python fetch_corpus.py`
Expected: prints `... already exists, skipping download (use --force to rebuild)` — confirms re-running (e.g. after a partial failure) doesn't silently redo a multi-minute download.

- [ ] **Step 5: Commit**

```bash
git add nanoGPT/data/scifi_char/fetch_corpus.py
git commit -m "Add reproducible corpus fetch script, run full fetch"
```

(`sci_fi_corpus.txt` itself is gitignored per Task 1 — confirm with `git status` that it does not appear as untracked/staged.)

---

### Task 5: Char-level tokenizer prep

**Files:**
- Create: `nanoGPT/data/scifi_char/prepare.py`
- Test: `nanoGPT/data/scifi_char/test_prepare.py`

**Interfaces:**
- Consumes: `nanoGPT/data/scifi_char/sci_fi_corpus.txt` (Task 4).
- Produces: `build_vocab(text) -> (stoi, itos)`, `encode(text, stoi) -> list[int]`, `decode(ids, itos) -> str`; on disk: `train.bin`, `val.bin`, `meta.pkl` (dict with `vocab_size`, `stoi`, `itos`) — consumed by Task 6's `train.py` via nanoGPT's `dataset = 'scifi_char'` convention.

- [ ] **Step 1: Write the failing tests**

```python
# nanoGPT/data/scifi_char/test_prepare.py
from prepare import build_vocab, encode, decode


def test_build_vocab_and_roundtrip():
    text = "hello world"
    stoi, itos = build_vocab(text)
    ids = encode(text, stoi)
    assert decode(ids, itos) == text


def test_vocab_is_sorted_and_unique():
    text = "banana"
    stoi, _ = build_vocab(text)
    assert set(stoi.keys()) == {"b", "a", "n"}
    assert len(stoi) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest test_prepare.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'prepare'`

- [ ] **Step 3: Write `prepare.py`**

```python
"""Char-level tokenizer prep for the sci-fi corpus, following nanoGPT's shakespeare_char/prepare.py pattern."""
import pickle
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
INPUT_PATH = HERE / "sci_fi_corpus.txt"


def build_vocab(text: str):
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos


def encode(text: str, stoi: dict) -> list[int]:
    return [stoi[ch] for ch in text]


def decode(ids, itos: dict) -> str:
    return "".join(itos[i] for i in ids)


def main():
    text = INPUT_PATH.read_text(encoding="utf-8")
    print(f"length of dataset in characters: {len(text):,}")

    stoi, itos = build_vocab(text)
    vocab_size = len(stoi)
    print(f"vocab size: {vocab_size}")

    n = len(text)
    train_text = text[: int(n * 0.9)]
    val_text = text[int(n * 0.9):]

    train_ids = np.array(encode(train_text, stoi), dtype=np.uint16)
    val_ids = np.array(encode(val_text, stoi), dtype=np.uint16)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    train_ids.tofile(HERE / "train.bin")
    val_ids.tofile(HERE / "val.bin")

    meta = {"vocab_size": vocab_size, "stoi": stoi, "itos": itos}
    with open(HERE / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest test_prepare.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run against the real corpus**

Run: `python prepare.py`
Expected: prints dataset length, vocab size, train/val token counts; creates `train.bin`, `val.bin`, `meta.pkl` in `nanoGPT/data/scifi_char/` (all gitignored).

- [ ] **Step 6: Commit**

```bash
git add nanoGPT/data/scifi_char/prepare.py nanoGPT/data/scifi_char/test_prepare.py
git commit -m "Add char-level tokenizer prep with tests"
```

---

### Task 6: Training config and baseline run

**Files:**
- Create: `nanoGPT/config/train_scifi_char.py`

**Interfaces:**
- Consumes: `nanoGPT/data/scifi_char/{train.bin,val.bin,meta.pkl}` (Task 5), `nanoGPT/train.py` (Task 2).
- Produces: `nanoGPT/out-scifi-char/ckpt.pt` — consumed by Task 7's `sample.py`.

- [ ] **Step 1: Write the training config**

```python
# nanoGPT/config/train_scifi_char.py
# Config for training a small char-level GPT on the sci-fi corpus, tuned for a laptop-scale
# Apple Silicon GPU (mps) run instead of nanoGPT's default multi-GPU/CUDA assumptions.
out_dir = 'out-scifi-char'
eval_interval = 250
eval_iters = 100
log_interval = 10

always_save_checkpoint = True

wandb_log = False

dataset = 'scifi_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3
max_iters = 3000
lr_decay_iters = 3000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100

device = 'mps'
compile = False
```

- [ ] **Step 2: Run the baseline training**

Run: `cd nanoGPT && python train.py config/train_scifi_char.py`
Expected: prints an `iter 0: loss X.XXXX` line, then periodic `iter N: loss X.XXXX, time Yms` lines with the loss trending downward over the run (roughly 15–30 min on M3 at this config), ending with `nanoGPT/out-scifi-char/ckpt.pt` written to disk.

- [ ] **Step 3: Verify loss decreased**

Note the first and last printed loss values from Step 2's output. Confirm the last value is meaningfully lower than the first (character-level loss should drop from ~4.2 nats/char, roughly `ln(vocab_size)`, toward something noticeably lower). If loss is flat or increasing, stop and debug before proceeding (do not move to Task 7 with a broken run).

- [ ] **Step 4: Commit the config** (checkpoint itself stays gitignored — it's a large binary artifact, not source)

```bash
git add nanoGPT/config/train_scifi_char.py
git commit -m "Add M3/mps training config for scifi_char, run baseline"
```

---

### Task 7: Sample generation

**Files:**
- Create: `notes/sample_output.txt`

**Interfaces:**
- Consumes: `nanoGPT/out-scifi-char/ckpt.pt` (Task 6), `nanoGPT/sample.py` (Task 2).
- Produces: `notes/sample_output.txt` — consumed by Task 9's README.

- [ ] **Step 1: Generate samples**

Run: `cd nanoGPT && python sample.py --out_dir=out-scifi-char --num_samples=3 --max_new_tokens=500`
Expected: prints 3 generated text samples to stdout, separated by `---------------`.

- [ ] **Step 2: Save the output**

Run: `mkdir -p ../notes && python sample.py --out_dir=out-scifi-char --num_samples=3 --max_new_tokens=500 > ../notes/sample_output.txt`
Expected: `notes/sample_output.txt` created at the repo root, non-empty.

- [ ] **Step 3: Commit**

```bash
cd ..
git add notes/sample_output.txt
git commit -m "Add sample generations from the trained checkpoint"
```

---

### Task 8: Pipeline notes

**AMENDMENT (portfolio documentation standard):** this repo is a portfolio artifact and the user wants the write-up held to the rigor of a master's-thesis-style technical report, not a casual blog note. `notes/pipeline-notes.md` is the deep document; Task 9's README stays a concise, practical entry point that links out to it.

**Files:**
- Create: `notes/pipeline-notes.md`

**Interfaces:**
- Consumes: nothing programmatic — written by hand, informed by Tasks 2–7.
- Produces: `notes/pipeline-notes.md` — the Phase 1 "plain-language notes" deliverable, referenced from Task 9's README.

- [ ] **Step 1: Write the notes as a structured technical report**

Write `notes/pipeline-notes.md` with formal section structure (thesis-chapter style), grounded throughout in this specific project's real numbers and design choices — never generic textbook filler:

1. **Abstract** — 3-5 sentences: what was built, on what data, what it demonstrates.
2. **Motivation** — why go beneath the API layer; tie to the "going one layer deeper" framing from the source learning plan.
3. **Background** — brief, precise definitions of the transformer components this project actually exercises (tokenization, embedding, self-attention, the training loop, backprop/optimization), each with a one-line citation to source material (e.g. Vaswani et al., "Attention Is All You Need," 2017; Karpathy's nanoGPT/"Neural Networks: Zero to Hero").
4. **Methodology** — the actual pipeline built: char-level tokenization and why (vs. BPE), the `stoi`/`itos`/`meta.pkl` representation, how token IDs become embeddings in `model.py`, what `n_head`/`n_embd`/`block_size` control and why they were sized for M3/laptop scale rather than nanoGPT's defaults, the `mps` backend choice over `cpu`/`cuda`.
5. **Results** — this run's actual numbers: final train/val loss (cite the specific values from Task 6's run, not placeholders), training time, vocab size, dataset size, and what the loss trajectory showed at each `eval_interval`. Include 1-2 short sample generations as evidence (from Task 7's output).
6. **Discussion** — what the loss/samples do and don't demonstrate about the model at this scale (e.g. character-level coherence vs. semantic coherence), and the engineering tradeoffs made (why this wasn't a fine-tuning or BPE approach — that's explicitly Phase 2, not this project).
7. **Limitations** — be honest: laptop-scale model, small compute budget, character-level ceiling on sample quality.
8. **References** — a short reference list (Vaswani et al. 2017, Karpathy/nanoGPT repo, Karpathy's Zero to Hero series, Project Gutenberg as the data source).

Precise, technical language throughout — no hand-waving, no restating textbook facts without connecting them to what this project's code actually does.

- [ ] **Step 2: Commit**

```bash
git add notes/pipeline-notes.md
git commit -m "Add pipeline notes as a structured technical report"
```

---

### Task 9: Finalize README

**Files:**
- Modify: `README.md` (Results / What this demonstrates sections)

**Interfaces:**
- Consumes: `notes/sample_output.txt` (Task 7), `notes/pipeline-notes.md` (Task 8), loss numbers from Task 6.

- [ ] **Step 1: Replace the `## Results` placeholder**

Fill in with: final train/val loss from Task 6's run, total training time, vocab size and dataset size (characters) from Task 5's `prepare.py` output, and 1–2 short excerpts pulled from `notes/sample_output.txt`.

- [ ] **Step 2: Replace the `## What this demonstrates` placeholder**

Fill in with the case-study framing from the LLM Learning Plan doc: problem → what you did → skills demonstrated (tokenization/attention/training-loop understanding, reproducible ML repo engineering, laptop-scale training tradeoffs). Keep this section concise (README is the practical entry point) and link prominently to `notes/pipeline-notes.md` — described as "a full technical write-up of the training pipeline, methodology, and results" — since that's where the thesis-level depth from Task 8 lives.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "Finalize README with training results and case-study framing"
```

---

### Task 10: Publish to GitHub

**Files:** none (repository operations only)

**Interfaces:** none — terminal task.

- [ ] **Step 1: Create the GitHub repo**

Run: `gh repo create nanogpt-scifi --public --source=. --remote=origin --description "Character-level GPT trained from scratch on public-domain sci-fi (Project Gutenberg), built with nanoGPT"`
Expected: repo created at `https://github.com/ericthebigsal/nanogpt-scifi`, `origin` remote added.

- [ ] **Step 2: Confirm what's about to be pushed**

Run: `git log --oneline` and `git status`
Expected: all prior task commits present in order; working tree clean (no untracked corpus/binary files — those are gitignored).

- [ ] **Step 3: Push**

Run: `git push -u origin main`
Expected: push succeeds; `https://github.com/ericthebigsal/nanogpt-scifi` shows the repo publicly.
