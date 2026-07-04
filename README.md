# nanogpt-scifi

A character-level GPT trained from scratch on a ~29MB corpus of public-domain
science fiction (Project Gutenberg), built to understand the transformer
training pipeline end-to-end — tokenization, self-attention, training loop,
loss curves — not just prompt around it.

Training engine: [nanoGPT](https://github.com/karpathy/nanoGPT) (Karpathy, MIT
license), vendored in `nanoGPT/`. Dataset: 97 public-domain works (Frankenstein,
The Time Machine, The War of the Worlds, Flatland, R.U.R., and others),
reproducible from source via the scripts below — the corpus itself is not
committed to this repo.

**New to how LLMs are trained?** Start with
[`notes/llm-primer.md`](notes/llm-primer.md) — a no-background-required
explainer covering tokenization, embeddings, attention, and the training loop,
grounded throughout in this project's actual numbers. For the engineering
process (including bugs and dead ends), see [`notes/build-log.md`](notes/build-log.md).

## Setup

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Rebuild the dataset

```bash
cd nanoGPT/data/scifi_char
python fetch_corpus.py      # downloads all 97 books from Project Gutenberg
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
python sample.py --out_dir=out-scifi-char --device=mps
```

(`--device=mps` is required on Apple Silicon — nanoGPT's default is `cuda`, which
asserts on this hardware.)

## Results

Training a 10.72M-parameter, 6-layer, character-level model for 3,000 iterations
on a single Apple M3 laptop GPU (`mps` backend, no cloud compute):

| | Start | End |
|---|---|---|
| Training loss | 5.4625 | **1.2332** |
| Validation loss | 5.4589 | **1.3032** |

Dataset: 28,934,015 characters, 251-character vocabulary, 26,040,613 training /
2,893,402 validation tokens. Val loss tracked train loss closely throughout — no
meaningful overfitting at this scale.

A generated sample (temperature 0.8):

> "Perhaps," said the control body. "I will go out alone of your own mother!"
>
> "Well?"
>
> "Come for me," he said, "that this activity-men is utterly in about the
> monster?"

Spelling, punctuation, and dialogue formatting are consistently correct;
sentence-level meaning isn't — the expected ceiling for a model this small at the
character level, not a bug. Full samples: [`notes/sample_output.txt`](notes/sample_output.txt).

## What this demonstrates

Most hands-on LLM experience today is prompting a model someone else built and
trained. This project builds the layer beneath that: a full pipeline from raw
public-domain text to a trained, sampling transformer, with every stage —
corpus construction, tokenization, model architecture, the training loop, and
generation — implemented or vendored and understood directly, on consumer
hardware rather than a managed training service.

Along the way, it also meant diagnosing and fixing two real engineering problems
that don't show up in a tutorial: a memory-pressure bug that silently made
training unusably slow (root-caused via system-level evidence, fixed with
gradient accumulation rather than shrinking the model), and a training run
interrupted by system sleep (resolved via checkpoint-resume with zero lost
progress). Skills demonstrated: transformer/attention/training-loop fluency,
reproducible ML pipeline engineering, and debugging under real hardware
constraints rather than assuming an idealized environment.

For the full, cited technical report — methodology, complete results, discussion,
and limitations — see [`notes/pipeline-notes.md`](notes/pipeline-notes.md). For
the engineering process as it actually happened, including both incidents above
in full, see [`notes/build-log.md`](notes/build-log.md).
