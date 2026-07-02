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
