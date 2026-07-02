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
