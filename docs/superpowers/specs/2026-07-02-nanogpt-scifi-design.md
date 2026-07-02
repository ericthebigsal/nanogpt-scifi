# nanogpt-scifi — Design Spec

**Date:** 2026-07-02
**Context:** Phase 1 of the LLM Learning Plan (technical fundamentals — tokenization, self-attention, training loop, loss curves). This is also intended to be a public GitHub portfolio artifact.

## Goal

Train a small GPT from scratch, end-to-end, on a public-domain sci-fi text corpus, on local Apple M3 hardware. Produce a working baseline run, sample generations, and plain-language notes on the training pipeline (tokenize → embed → attention → loss → backprop → update). Ship it as a public, reproducible GitHub repo suitable for linking from a resume/LinkedIn.

## Data

`sci_fi_corpus.txt` — an existing local file, ~33MB, ~98 public-domain works from Project Gutenberg (e.g. *Frankenstein*, *The Time Machine*, *The War of the Worlds*, *Flatland*, R.U.R., several Lovecraft stories). 97 of 98 entries have a Gutenberg eBook ID embedded in their header (`[eBook #N]`), extractable into a manifest.

Public domain — no IP/confidentiality concerns for a public repo. 33MB is well beyond nanoGPT's standard ~1MB Shakespeare toy dataset, giving a from-scratch character-level model enough signal to produce semi-coherent sample output, not just noise.

**Not committed to git** — 33MB of raw text is unconventional to check into a portfolio repo. Instead:
- `data/scifi_char/manifest.json` — the ~97 `{gutenberg_id, title}` pairs extracted from the existing corpus
- `data/scifi_char/fetch_corpus.py` — rebuilds the corpus: reuses the local file if present, otherwise re-downloads each book by ID from Project Gutenberg and concatenates them, stripping the `*** START/END OF THE PROJECT GUTENBERG EBOOK ***` boilerplate and normalizing `\r\n` → `\n`
- Raw corpus and any downloaded artifacts are gitignored

This makes the dataset reproducible from source without bloating the repo.

## Environment

- Python 3.14 venv (torch 2.12.1 has cp314 wheels for macOS arm64 — confirmed via PyPI)
- torch, numpy, tiktoken, tqdm
- Training via PyTorch's `mps` backend (Apple M3 GPU) — no cloud compute needed at this scale
- No conda/pyenv present on the machine; using the system Homebrew Python 3.14 + venv is sufficient

## Training approach

nanoGPT (Karpathy, MIT-licensed), cloned into the repo and credited in the README — not a submodule, so it's simple to read/modify inline. Character-level tokenization, following the `shakespeare_char` example pattern but scaled to the larger corpus. Model size and training config tuned for laptop-scale (a few million params, training measured in tens of minutes), not datacenter scale.

## Repo structure

```
nanogpt-scifi/
├── README.md              # project overview, how to run, portfolio framing (problem/what-I-did/skills)
├── LICENSE                 # MIT (code only; corpus stays public-domain Gutenberg text)
├── .gitignore               # excludes data/scifi_char/sci_fi_corpus.txt and downloaded artifacts
├── requirements.txt
├── data/scifi_char/
│   ├── manifest.json
│   ├── fetch_corpus.py
│   └── prepare.py            # nanoGPT-style char-level prep → train.bin/val.bin/meta.pkl
├── nanoGPT/                  # Karpathy's nanoGPT source
├── notes/                    # plain-language pipeline notes (Phase 1 deliverable)
└── train_config.py           # small-model config tuned for M3/laptop scale
```

## GitHub

- Public repo under `ericthebigsal`, name `nanogpt-scifi`
- Created via `gh repo create`
- README written to double as the seed of the Phase 3 case-study write-up later (problem → what you did → impact → skills demonstrated), not just run instructions

## Deliverable (Phase 1 exit criteria)

- Working training run on the sci-fi corpus with visibly decreasing loss
- Sample text generated from the trained model
- `notes/` containing plain-language explanation of each pipeline stage
- Public GitHub repo, cloneable and reproducible from a fresh machine (via `fetch_corpus.py`)

## Out of scope (this spec)

- Fine-tuning (LoRA/QLoRA) — that's Phase 2
- Benchmark harness against a frontier-model API call — Phase 2
- Full Phase 3 case-study write-up — later, once Phase 1/2 numbers exist
