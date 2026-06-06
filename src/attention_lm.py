"""
Stacked self-attention language model (NumPy, from scratch).

Pipeline per user message:
  1. Tokenize words → ids
  2. Embed ids → vectors (d_model)
  3. Self-attention — each token weights every other token
  4. Stack layers (attention + feed-forward + residual)
  5. Softmax over vocabulary → next-token probabilities
  6. Autoregressive decode — append token, repeat
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.neural_network import relu, relu_derivative, softmax
from src.tokenizer import WordTokenizer


def _softmax_3d(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def _causal_mask(seq_len: int) -> np.ndarray:
    """True where attention must be blocked (future positions)."""
    return np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)


@dataclass
class AttentionLMConfig:
    d_model: int = 48
    n_layers: int = 4
    d_ff: int = 96
    max_seq_len: int = 64
    learning_rate: float = 0.05
    seed: int = 42


@dataclass
class FeedForwardBlock:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray

    @classmethod
    def create(cls, d_model: int, d_ff: int, rng: np.random.Generator) -> FeedForwardBlock:
        scale1 = np.sqrt(2.0 / d_model)
        scale2 = np.sqrt(2.0 / d_ff)
        return cls(
            w1=rng.normal(0, scale1, (d_model, d_ff)),
            b1=np.zeros((1, d_ff)),
            w2=rng.normal(0, scale2, (d_ff, d_model)),
            b2=np.zeros((1, d_model)),
        )

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        z1 = x @ self.w1 + self.b1
        h1 = relu(z1)
        out = h1 @ self.w2 + self.b2
        return out, z1, h1


@dataclass
class StackedAttentionLM:
    """Causal language model with multi-layer self-attention."""

    config: AttentionLMConfig
    tokenizer: WordTokenizer
    embeddings: np.ndarray
    w_out: np.ndarray
    b_out: np.ndarray
    layers: list[FeedForwardBlock] = field(default_factory=list)

    @classmethod
    def create(cls, tokenizer: WordTokenizer, config: AttentionLMConfig | None = None) -> StackedAttentionLM:
        cfg = config or AttentionLMConfig()
        rng = np.random.default_rng(cfg.seed)
        vocab = max(tokenizer.vocab_size, 4)
        emb_scale = 0.02
        model = cls(
            config=cfg,
            tokenizer=tokenizer,
            embeddings=rng.normal(0, emb_scale, (vocab, cfg.d_model)),
            layers=[FeedForwardBlock.create(cfg.d_model, cfg.d_ff, rng) for _ in range(cfg.n_layers)],
            w_out=rng.normal(0, emb_scale, (cfg.d_model, vocab)),
            b_out=np.zeros((1, vocab)),
        )
        return model

    def _embed(self, token_ids: np.ndarray) -> np.ndarray:
        return self.embeddings[token_ids]

    def _layer_norm(self, x: np.ndarray) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        return (x - mean) / (std + 1e-5)

    def _self_attention(
        self, x: np.ndarray, *, return_weights: bool = False
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Scaled dot-product self-attention (Q=K=V=x) with causal mask."""
        x = self._layer_norm(x)
        batch, seq, d_model = x.shape
        scores = x @ np.swapaxes(x, -1, -2) / np.sqrt(d_model)
        scores = np.clip(scores, -40.0, 40.0)
        mask = _causal_mask(seq)
        scores = np.where(mask[np.newaxis, :, :], -1e9, scores)
        weights = _softmax_3d(scores)
        out = weights @ x
        if return_weights:
            return out, weights
        return out, None

    def forward(
        self, token_ids: np.ndarray, *, return_attention: bool = False
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Return logits (batch, seq, vocab) and optional attention per layer."""
        if token_ids.ndim == 1:
            token_ids = token_ids[np.newaxis, :]

        seq = token_ids.shape[1]
        if seq > self.config.max_seq_len:
            token_ids = token_ids[:, -self.config.max_seq_len :]
            seq = token_ids.shape[1]

        x = self._embed(token_ids)
        attentions: list[np.ndarray] = []

        for block in self.layers:
            attn_out, weights = self._self_attention(x, return_weights=return_attention)
            if weights is not None:
                attentions.append(weights[0])
            x = x + attn_out
            ffn_out, _, _ = block.forward(x)
            x = x + ffn_out

        logits = x @ self.w_out + self.b_out
        return logits, attentions

    def next_token_distribution(
        self, token_ids: list[int] | np.ndarray, *, top_k: int = 5
    ) -> dict[str, Any]:
        arr = np.array(token_ids, dtype=int)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        logits, attentions = self.forward(arr, return_attention=True)
        probs = softmax(logits)[0, -1]
        top_idx = np.argsort(probs)[::-1][:top_k]
        distribution = [
            {
                "token": self.tokenizer.id_to_word[int(i)],
                "probability": round(float(probs[i]), 4),
            }
            for i in top_idx
        ]
        attention_pairs: list[dict[str, Any]] = []
        if attentions and arr.shape[1] > 1:
            last_layer = attentions[-1]
            src_tokens = [self.tokenizer.id_to_word[int(t)] for t in arr[0]]
            query_idx = len(src_tokens) - 1
            row = last_layer[query_idx]
            for j, score in enumerate(row):
                if j == query_idx or score < 0.05:
                    continue
                attention_pairs.append(
                    {
                        "query": src_tokens[query_idx],
                        "key": src_tokens[j],
                        "weight": round(float(score), 4),
                    }
                )
            attention_pairs.sort(key=lambda p: p["weight"], reverse=True)
            attention_pairs = attention_pairs[:6]

        return {
            "distribution": distribution,
            "top_token": distribution[0]["token"] if distribution else "",
            "top_probability": distribution[0]["probability"] if distribution else 0.0,
            "attention_pairs": attention_pairs,
            "layers": self.config.n_layers,
        }

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 12,
        stop_words: frozenset[str] | None = None,
    ) -> dict[str, Any]:
        stop_words = stop_words or frozenset({"<eos>", "<pad>"})
        token_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        generated_steps: list[dict[str, Any]] = []

        for _ in range(max_new_tokens):
            step_info = self.next_token_distribution(token_ids)
            next_id = int(
                np.argmax(
                    softmax(
                        self.forward(np.array([token_ids], dtype=int), return_attention=False)[0]
                    )[0, -1]
                )
            )
            word = self.tokenizer.id_to_word[next_id]
            token_ids.append(next_id)
            generated_steps.append(
                {
                    "token": word,
                    "distribution": step_info["distribution"],
                    "attention_pairs": step_info["attention_pairs"],
                }
            )
            if word in stop_words:
                break

        answer_tokens = [
            t
            for t in token_ids
            if self.tokenizer.id_to_word[t] not in {"<bos>", "<pad>", "<eos>", "<unk>"}
        ]
        return {
            "prompt": prompt,
            "token_ids": token_ids,
            "text": self.tokenizer.decode(answer_tokens),
            "steps": generated_steps,
        }

    def train(self, texts: list[str], *, epochs: int = 80, verbose: bool = False) -> list[dict[str, float]]:
        """Next-token cross-entropy on encoded sequences."""
        sequences: list[list[int]] = []
        for text in texts:
            ids = self.tokenizer.encode(text, add_bos=True, add_eos=True)
            if len(ids) >= 3:
                sequences.append(ids[: self.config.max_seq_len])

        if not sequences:
            return []

        history: list[dict[str, float]] = []
        lr = self.config.learning_rate
        vocab = self.embeddings.shape[0]
        rng = np.random.default_rng(self.config.seed)

        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            total_correct = 0
            total_tokens = 0

            order = rng.permutation(len(sequences))
            for idx in order:
                ids = sequences[int(idx)]
                x_ids = np.array(ids[:-1], dtype=int)[np.newaxis, :]
                targets = np.array(ids[1:], dtype=int)
                seq_len = x_ids.shape[1]

                x = self._embed(x_ids)
                layer_caches: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
                attentions: list[np.ndarray] = []

                for block in self.layers:
                    attn_out, attn_w = self._self_attention(x, return_weights=True)
                    attentions.append(attn_w)
                    x_attn = x + attn_out
                    ffn_out, z1, h1 = block.forward(x_attn)
                    layer_caches.append((x, attn_w, x_attn, z1, h1))
                    x = x_attn + ffn_out

                logits = x @ self.w_out + self.b_out
                probs = softmax(logits)[0]
                eps = 1e-9
                loss = -np.mean(np.log(probs[np.arange(seq_len), targets] + eps))
                total_loss += float(loss)
                total_correct += int(np.sum(np.argmax(probs, axis=1) == targets))
                total_tokens += seq_len

                d_logits = probs.copy()
                d_logits[np.arange(seq_len), targets] -= 1.0
                d_logits /= seq_len

                d_hidden = d_logits @ self.w_out.T
                self.w_out -= lr * (x[0].T @ d_logits)
                self.b_out -= lr * np.sum(d_logits, axis=0, keepdims=True)

                for t_idx, token_id in enumerate(x_ids[0]):
                    self.embeddings[int(token_id)] -= lr * d_hidden[t_idx]
                    norm = np.linalg.norm(self.embeddings[int(token_id)])
                    if norm > 3.0:
                        self.embeddings[int(token_id)] *= 3.0 / norm

            acc = total_correct / max(total_tokens, 1)
            metrics = {"epoch": float(epoch), "loss": total_loss / len(sequences), "accuracy": acc}
            history.append(metrics)
            if verbose and epoch % 20 == 0:
                print(f"  lm epoch {epoch:3d}  loss={metrics['loss']:.4f}  token_acc={acc:.1%}")

        return history

    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(path / "tokenizer.json")
        payload = {
            "config": self.config.__dict__,
            "embeddings": self.embeddings.tolist(),
            "w_out": self.w_out.tolist(),
            "b_out": self.b_out.tolist(),
            "layers": [
                {
                    "w1": layer.w1.tolist(),
                    "b1": layer.b1.tolist(),
                    "w2": layer.w2.tolist(),
                    "b2": layer.b2.tolist(),
                }
                for layer in self.layers
            ],
        }
        (path / "model.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> StackedAttentionLM:
        path = Path(directory)
        tokenizer = WordTokenizer.load(path / "tokenizer.json")
        payload = json.loads((path / "model.json").read_text(encoding="utf-8"))
        config = AttentionLMConfig(**payload["config"])
        model = cls(
            config=config,
            tokenizer=tokenizer,
            embeddings=np.array(payload["embeddings"], dtype=float),
            w_out=np.array(payload["w_out"], dtype=float),
            b_out=np.array(payload["b_out"], dtype=float),
        )
        model.layers = [
            FeedForwardBlock(
                w1=np.array(layer["w1"], dtype=float),
                b1=np.array(layer["b1"], dtype=float),
                w2=np.array(layer["w2"], dtype=float),
                b2=np.array(layer["b2"], dtype=float),
            )
            for layer in payload["layers"]
        ]
        return model
