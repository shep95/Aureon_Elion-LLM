"""Byte-pair encoding tokenizer — preserves Python syntax for code generation."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from src.tokenizer import BOS, EOS, PAD, SPECIALS, THINK, THEREFORE, UNK

# Python-aware pretokenization — keeps operators, brackets, newlines, strings.
_CODE_PIECE_RE = re.compile(
    r"def |class |return |import |from |for |while |if |elif |else |"
    r"try |except |with |as |in |not |and |or |lambda |yield |"
    r"[a-zA-Z_][a-zA-Z0-9_]*|"
    r"\d+\.\d+|\d+|"
    r"\n|\t|"
    r"[():,\[\]{}:+\-*/%=<>@#&|^~\.]|"
    r"'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"|"
    r"  +| "
)


def _pretokenize(text: str) -> list[str]:
    pieces: list[str] = []
    for match in _CODE_PIECE_RE.finditer(text):
        piece = match.group(0)
        if piece.isspace() and "\n" not in piece and "\t" not in piece:
            continue
        pieces.append(piece)
    return pieces or [text]


def _pair_counts(symbols: list[str]) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for i in range(len(symbols) - 1):
        counts[(symbols[i], symbols[i + 1])] += 1
    return counts


def _merge_pair(symbols: list[str], pair: tuple[str, str], merged: str) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
            out.append(merged)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return out


@dataclass
class BPETokenizer:
    """BPE subword tokenizer with code-safe pretokenization."""

    word_to_id: dict[str, int] = field(default_factory=dict)
    id_to_word: list[str] = field(default_factory=list)
    merges: list[tuple[str, str]] = field(default_factory=list)

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_word)

    @property
    def pad_id(self) -> int:
        return self.word_to_id.get(PAD, 0)

    @property
    def unk_id(self) -> int:
        return self.word_to_id.get(UNK, 1)

    @property
    def bos_id(self) -> int:
        return self.word_to_id.get(BOS, 2)

    @property
    def eos_id(self) -> int:
        return self.word_to_id.get(EOS, 3)

    @property
    def think_id(self) -> int:
        return self.word_to_id.get(THINK, 4)

    def build_vocab(
        self,
        texts: Iterable[str],
        *,
        min_freq: int = 1,
        max_vocab: int = 32_000,
        num_merges: int | None = None,
    ) -> None:
        freq: Counter[str] = Counter()
        corpus_symbols: list[list[str]] = []
        for text in texts:
            symbols = _pretokenize(text)
            corpus_symbols.append(symbols)
            for sym in symbols:
                freq[sym] += 1

        base = sorted({s for s, c in freq.items() if c >= min_freq})
        vocab = list(SPECIALS) + base
        merges_target = num_merges
        if merges_target is None:
            merges_target = max(0, max_vocab - len(vocab) - 1)

        self.merges = []
        working = [list(s) for s in corpus_symbols]

        for _ in range(merges_target):
            pair_counter: Counter[tuple[str, str]] = Counter()
            for symbols in working:
                pair_counter.update(_pair_counts(symbols))
            if not pair_counter:
                break
            best_pair, _ = pair_counter.most_common(1)[0]
            merged = best_pair[0] + best_pair[1]
            if merged in vocab:
                continue
            vocab.append(merged)
            self.merges.append(best_pair)
            working = [_merge_pair(s, best_pair, merged) for s in working]
            if len(vocab) >= max_vocab:
                break

        self.id_to_word = vocab[:max_vocab]
        self.word_to_id = {word: idx for idx, word in enumerate(self.id_to_word)}

    def _encode_pieces(self, text: str) -> list[str]:
        symbols = _pretokenize(text)
        for pair in self.merges:
            merged = pair[0] + pair[1]
            symbols = _merge_pair(symbols, pair, merged)
        return symbols

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = True,
        add_eos: bool = False,
        max_tokens: int | None = None,
    ) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        for piece in self._encode_pieces(text):
            ids.append(self.word_to_id.get(piece, self.unk_id))
        if add_eos:
            ids.append(self.eos_id)
        if max_tokens is not None and len(ids) > max_tokens:
            ids = [ids[0], *ids[-(max_tokens - 1) :]]
        return ids

    def decode(self, ids: Iterable[int], *, skip_special: bool = True) -> str:
        parts: list[str] = []
        for token_id in ids:
            word = self.id_to_word[int(token_id)] if 0 <= int(token_id) < len(self.id_to_word) else UNK
            if skip_special and word in SPECIALS:
                continue
            parts.append(word)
        return "".join(parts)

    def to_dict(self) -> dict:
        return {
            "id_to_word": self.id_to_word,
            "merges": [list(p) for p in self.merges],
            "tokenizer_type": "bpe",
        }

    @classmethod
    def from_dict(cls, payload: dict) -> BPETokenizer:
        tok = cls()
        tok.id_to_word = list(payload["id_to_word"])
        tok.word_to_id = {word: idx for idx, word in enumerate(tok.id_to_word)}
        tok.merges = [tuple(p) for p in payload.get("merges", [])]
        return tok

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("tokenizer_type") == "bpe":
            return cls.from_dict(payload)
        # Fallback: word tokenizer file — wrap as BPE with no merges
        tok = cls()
        tok.id_to_word = list(payload["id_to_word"])
        tok.word_to_id = {word: idx for idx, word in enumerate(tok.id_to_word)}
        return tok
