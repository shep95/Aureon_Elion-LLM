"""Word-level tokenizer for the attention language model."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

PAD = "<pad>"
UNK = "<unk>"
BOS = "<bos>"
EOS = "<eos>"
CTX = "<ctx>"
THINK = "<think>"
THEREFORE = "<therefore>"

SPECIALS = (PAD, UNK, BOS, EOS, CTX, THINK, THEREFORE)


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s'-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class WordTokenizer:
    """Map words to integer ids and back."""

    word_to_id: dict[str, int] = field(default_factory=dict)
    id_to_word: list[str] = field(default_factory=list)

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

    def build_vocab(self, texts: Iterable[str], *, min_freq: int = 1, max_vocab: int = 1_000_000) -> None:
        freq: dict[str, int] = {}
        for text in texts:
            for word in _normalize(text).split():
                freq[word] = freq.get(word, 0) + 1

        ranked = sorted(
            (w for w, c in freq.items() if c >= min_freq),
            key=lambda w: (-freq[w], w),
        )[: max(0, max_vocab - len(SPECIALS))]

        self.id_to_word = list(SPECIALS) + ranked
        self.word_to_id = {word: idx for idx, word in enumerate(self.id_to_word)}

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
        for word in _normalize(text).split():
            ids.append(self.word_to_id.get(word, self.unk_id))
        if add_eos:
            ids.append(self.eos_id)
        if max_tokens is not None and len(ids) > max_tokens:
            ids = [ids[0], *ids[-(max_tokens - 1) :]]
        return ids

    def decode(self, ids: Iterable[int], *, skip_special: bool = True) -> str:
        words: list[str] = []
        for token_id in ids:
            word = self.id_to_word[int(token_id)] if 0 <= int(token_id) < len(self.id_to_word) else UNK
            if skip_special and word in SPECIALS:
                continue
            words.append(word)
        return " ".join(words)

    def to_dict(self) -> dict:
        return {"id_to_word": self.id_to_word}

    @classmethod
    def from_dict(cls, payload: dict) -> WordTokenizer:
        tok = cls()
        tok.id_to_word = list(payload["id_to_word"])
        tok.word_to_id = {word: idx for idx, word in enumerate(tok.id_to_word)}
        return tok

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> WordTokenizer:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)
