"""BPE tokenizer tests — Python syntax preservation."""

from __future__ import annotations

from src.bpe_tokenizer import BPETokenizer, _pretokenize


def test_pretokenize_preserves_def_and_parens():
    pieces = _pretokenize("def add(a, b): return a + b")
    text = "".join(pieces)
    assert "def " in text
    assert "(" in text
    assert ")" in text
    assert "return" in text


def test_bpe_roundtrip_code():
    tok = BPETokenizer()
    sample = [
        "question write python code def add(a, b): return a + b answer def add(a, b): return a + b",
        "question reverse string answer def reverse(s): return s[::-1]",
    ]
    tok.build_vocab(sample, max_vocab=500)
    encoded = tok.encode("def add(a, b): return a + b", add_bos=False, add_eos=False)
    decoded = tok.decode(encoded, skip_special=True)
    assert "def " in decoded
    assert "add" in decoded
    assert "return" in decoded
