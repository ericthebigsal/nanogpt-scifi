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

## Task 6 — Training config and baseline run (and a memory-pressure detour)

**What:** Wrote `nanoGPT/config/train_scifi_char.py` (6 layers, 6 heads, 384-dim
embeddings, block size 256, `mps` backend, 3,000 iterations) and ran the baseline
training. This task took two attempts because the first attempt uncovered a real
environmental problem, not a code bug — worth documenting in full because the
debugging process and the decision it led to are as instructive as the training
result itself.

### The problem

The first training attempt (`batch_size=64`, `gradient_accumulation_steps=1`,
16,384 tokens/iteration) never produced a usable run. It consumed CPU for 70+
minutes and produced zero checkpoint and almost no log output, well past the
15–30 minute window the task brief expected.

### Symptoms

- A short, verbose 20-iteration smoke test (added specifically to diagnose this —
  see Investigation below) showed the model initializing correctly (10.72M
  parameters, sane starting loss of ~5.49, matching the theoretical random-guess
  baseline of `ln(251) ≈ 5.5` for a 251-character vocabulary) — so the config and
  code path were not the problem.
- But per-iteration time was wildly erratic and *increasing*: `8.2s → 2.7s → 68.2s
  → 144.3s → 116.1s` for iterations 0–4. Real, correctly-configured training on
  this hardware should take well under a second per iteration at this model size —
  not tens of seconds, and not a climbing trend.
- `vm_stat` during the stalled run showed only ~4,000 free memory pages (~66MB)
  on an 8GB machine, alongside hundreds of millions of cumulative `Swapins`/
  `Swapouts` — i.e. the system was thrashing (swapping actively-used memory to
  disk and back), not idling or crashing.

### Investigation

An initial attempt to delegate this task to a subagent went in circles: it tried
importing `train.py` as a module to debug it (which broke nanoGPT's own
config-override mechanism and produced an unrelated, misleading `AssertionError:
Torch not compiled with CUDA enabled`), tried several output-buffering fixes, and
eventually reported BLOCKED without ever forming a testable hypothesis about *why*
iterations were slow — a good example of "random fixes waste time" from
[superpowers:systematic-debugging]: none of those attempts gathered evidence about
the actual runtime behavior of a working run.

The controller re-approached this by direct evidence-gathering: reproduce with a
short, cheap smoke test (20 iterations, unbuffered stdout, `log_interval=1`) rather
than immediately re-running the full 3,000-iteration job, then check system state
(`vm_stat`, `ps -eo pid,rss,comm -r`) *while it ran*. That surfaced the free-memory
and swap numbers above, plus the fact that Chrome (several helper processes) and
Claude Code itself were the two largest resident-memory consumers competing with
the training process for the same 8GB of unified memory that Apple's `mps` backend
shares between CPU and GPU.

**Root cause:** not a bug — a memory-constrained 8GB development machine, with
other running applications, left too little free memory for this training job's
working set, causing the OS to swap actively-used pages to disk mid-computation.
That swapping, not the model or the code, was responsible for the erratic,
escalating iteration times.

### Options considered

| Option | Pros | Cons |
|---|---|---|
| **Close memory-heavy apps (e.g. Chrome) before training** | Zero changes to config or code; matches the exact numbers already documented in `llm-primer.md`/the plan; simplest to reason about | Manual step required before every training run; doesn't fix the underlying fragility if something else consumes memory later |
| **Run the full job anyway, under memory pressure** | No changes needed at all | Could take hours instead of minutes (unbounded — the smoke test showed *worsening*, not stable, iteration times); wastes wall-clock proving nothing new |
| **Shrink the model itself** (`n_embd`, `n_layer`, or `block_size`) | Directly and reliably reduces memory footprint | Changes the actual model being trained — weaker capacity (`n_embd`/`n_layer`) or a smaller context window (`block_size`, which scales attention memory quadratically) would likely produce measurably worse loss/sample quality, and requires rewriting the numbers already committed to `llm-primer.md`, this build log, and the plan's Task 6 brief |
| **Reduce `batch_size`, compensate with `gradient_accumulation_steps`** (chosen) | Shrinks peak activation memory (the actual bottleneck) without changing the model or the effective training statistics — `batch_size=16` × `gradient_accumulation_steps=4` reproduces the exact same 16,384 tokens/iteration as the original `batch_size=64` single-step config | Each iteration now runs 4 smaller forward/backward passes instead of 1 large one, adding some fixed per-call overhead — slightly slower per iteration even once memory pressure is resolved |

### Decision

Eric chose **reduce `batch_size` + add `gradient_accumulation_steps`** — the
option that treats this as an engineering response to a memory-constrained
*environment*, not a reason to compromise the model being trained. It's also
arguably better portfolio material than either alternative: it demonstrates
understanding of what specifically consumes memory during training (activations,
not parameters) and how to trade a memory budget against wall-clock time without
touching model quality.

### Outcome

A second smoke test with `batch_size=16, gradient_accumulation_steps=4` (same
16,384 tokens/iteration) showed the fix worked immediately: iteration time
stabilized at a consistent ~900ms (vs. the previous run's climbing 8s→144s), with
loss dropping cleanly from 5.49 to 3.17 over the first 20 iterations. System-wide
free memory was still low, but the smaller per-step working set avoided triggering
the severe swapping the larger batch size caused.

**Learned:** when a training run "hangs" with no output, the instinct to add more
logging or fix output buffering can be a distraction from the real question —
*is the hardware actually keeping up?* A single `vm_stat` check settled in seconds
what 70+ minutes of blind waiting couldn't. It's also worth separating "is my
config/code correct" from "does my environment have the resources this job needs"
as two genuinely different failure modes with different fixes — conflating them
is what led the first debugging attempt in circles.

### A second interruption, and why checkpointing paid off

With the memory fix in place, the corrected run (`batch_size=16,
gradient_accumulation_steps=4`) started cleanly at 16:17 and trained smoothly —
loss dropping from `5.46` at iter 0 to `1.35` by iter 1500 (a checkpoint saved
automatically at that point, per `eval_interval=250` × `always_save_checkpoint`).
At 16:52 (iter 1740, ~35 minutes in) the background process was killed. This was
a *different* failure mode from the memory issue: macOS's unified log showed 15
sleep/wake cycles that day and no memory/jetsam kill anywhere near the process —
the training command simply hadn't been wrapped in `caffeinate`, so ordinary
system idle sleep took the process down. Root cause confirmed, not guessed: this
is exactly the kind of "two genuinely different failure modes" the memory bug's
lesson above calls out — thrashing and sleep produce superficially similar
symptoms (a training run that stops making progress) but need entirely different
fixes.

Because nanoGPT's `always_save_checkpoint=True` had been saving a full,
resumable checkpoint (model weights, optimizer state, and `iter_num`) every 250
iterations, none of that first 1500 iterations of work was lost. Training was
relaunched as `caffeinate -i python train.py config/train_scifi_char.py
--init_from=resume`, which picked back up at iter 1500 exactly (`step 1500: train
loss 1.3453, val loss 1.4053` — matching the pre-interruption checkpoint almost
exactly) and ran the remaining 1500 iterations to completion. One further,
smaller anomaly appeared mid-resume — a single iteration (2740) took ~15 minutes
against a ~1-second steady state, almost certainly a display-sleep/wake blip that
`caffeinate -i`'s idle-sleep prevention doesn't fully cover — but the run
self-recovered without intervention and reached `max_iters=3000`.

**Learned:** a `caffeinate`-wrapped, checkpoint-resumable training command is
worth setting up *before* the first real run on any long training job on
consumer hardware, not after the first interruption — the entire second
disruption cost zero lost training progress specifically because
`always_save_checkpoint` and nanoGPT's `init_from='resume'` path existed and were
already being used. Note also that `caffeinate` only prevents *idle* system
sleep; it cannot prevent sleep from a physically closed laptop lid, and does not
fully guarantee against display-sleep-related hiccups.

### Final result

| | Start (iter 0) | End (iter 3000) |
|---|---|---|
| Train loss | 5.4625 | **1.2332** |
| Val loss | 5.4589 | **1.3032** |

Val loss tracked train loss closely throughout (final gap: 0.07), showing no
meaningful overfitting at this scale. Full eval-interval trajectory:

```
step    0: train 5.4625, val 5.4589
step  250: train 2.1241, val 2.1001
step  500: train 1.7561, val 1.7495
step  750: train 1.5712, val 1.5959
step 1000: train 1.4663, val 1.4940
step 1250: train 1.4077, val 1.4349
step 1500: train 1.3500, val 1.3964   <- checkpoint; run interrupted by system sleep here
step 1500: train 1.3453, val 1.4053   <- resumed from checkpoint, confirms clean resume
step 1750: train 1.3125, val 1.3600
step 2000: train 1.2921, val 1.3437
step 2250: train 1.2602, val 1.3339
step 2500: train 1.2465, val 1.3182
step 2750: train 1.2475, val 1.3128
step 3000: train 1.2332, val 1.3032   <- final
```

Combined wall-clock across both attempts: ~35 minutes (first attempt, iters
0–1740) + ~69 minutes (resumed attempt, iters 1500–3000, inflated by the ~15-minute
mid-run anomaly noted above) — well above the plan's original 15–30 minute
estimate, almost entirely due to the two interruptions rather than steady-state
compute (which held at a consistent ~0.9–1.1s/iteration throughout).

_Commit: `2f0d61d`_

---

## Task 7 — Sample generation

**What:** Ran nanoGPT's `sample.py` against the Task 6 checkpoint to generate 3
samples of 500 characters each, saved to `notes/sample_output.txt`.

**Why:** Loss curves prove the model is learning *something*, but they don't show
what — actually reading generated text is the only way to see character-level
learning made concrete: real word boundaries, plausible spelling, punctuation and
dialogue conventions, sci-fi vocabulary bleeding through from training data.

**What went wrong (small, quick fix):** the first run failed immediately with
`ModuleNotFoundError: No module named 'tiktoken'`. Vendored nanoGPT's `sample.py`
imports `tiktoken` unconditionally at the top of the file, but only actually
*uses* it on a fallback path for datasets that have no `meta.pkl` (i.e. datasets
tokenized with GPT-2's BPE vocabulary) — a path this project's char-level
tokenizer (Task 5) never takes, since `meta.pkl` always exists here. Rather than
add `tiktoken` as a dependency this project doesn't otherwise need (violating the
plan's "no BPE/tiktoken" constraint just to satisfy an unused import), the fix was
to move `import tiktoken` from module load time into the specific `else` branch
that actually calls it — a one-line, behavior-preserving change to vendored code.

**Learned:** the samples themselves are a good illustration of what character-
level training does and doesn't buy you at this scale: spelling, punctuation, and
dialogue formatting (`"..." said the man.`) are consistently correct, and sci-fi
vocabulary from the training corpus surfaces clearly (*tentacles*, *projectiles*,
*monster*, *Empress*, *cavern*) — but sentences don't hold together semantically
("the carlot of addition," "Hercur"). That gap (fluent local structure, no
long-range meaning) is exactly what Module 8/9 of
[`llm-primer.md`](llm-primer.md) predicts for a 10.7M-parameter, character-level
model, and it's real evidence of it rather than a claim.

_Commit: `ba79299`_

---

_This log is updated as work progresses; see `git log` for the authoritative,
timestamped record of every change described above._
