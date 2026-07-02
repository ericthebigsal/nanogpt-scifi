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
