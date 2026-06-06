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
import os
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
    d_model: int = 128
    n_layers: int = 6
    d_ff: int = 512
    max_seq_len: int = 1_000_000
    learning_rate: float = 0.05
    seed: int = 42
    model_version: int = 4


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
        x = self.embeddings[token_ids]
        seq = x.shape[1]
        x = x + self._positional_encoding(seq)[np.newaxis, :, :]
        return x

    def _positional_encoding(self, seq_len: int) -> np.ndarray:
        d = self.config.d_model
        pe = np.zeros((seq_len, d))
        position = np.arange(seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d, 2) * (-np.log(10000.0) / d))
        pe[:, 0::2] = np.sin(position * div_term)
        if d > 1:
            pe[:, 1::2] = np.cos(position * div_term[: d // 2])
        return pe

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
        max_prompt_tokens: int | None = None,
    ) -> dict[str, Any]:
        stop_words = stop_words or frozenset({"<eos>", "<pad>"})
        prompt_limit = max_prompt_tokens or max(8, self.config.max_seq_len - max_new_tokens)
        token_ids = self.tokenizer.encode(
            prompt, add_bos=True, add_eos=False, max_tokens=prompt_limit
        )
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
        """Next-token cross-entropy — backprop through output head, embeddings, and FFN layers."""
        sequences: list[list[int]] = []
        max_len = self.config.max_seq_len
        chunk_cap = int(os.environ.get("AUREON_PREDICT_TRAIN_CHUNK", "256"))
        train_max = min(max_len, max(32, chunk_cap))
        stride = max(train_max // 2, 32)
        for text in texts:
            ids = self.tokenizer.encode(text, add_bos=True, add_eos=True)
            if len(ids) <= train_max:
                if len(ids) >= 3:
                    sequences.append(ids)
                continue
            for start in range(0, len(ids) - 2, stride):
                chunk = ids[start : start + train_max]
                if len(chunk) >= 3:
                    sequences.append(chunk)

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

                ffn_lr = lr * 0.25
                d_layer = d_hidden[np.newaxis, :, :]
                for li in range(len(self.layers) - 1, -1, -1):
                    block = self.layers[li]
                    _, _, x_attn, z1, h1 = layer_caches[li]
                    d_ffn = d_layer[0]
                    d_h1 = d_ffn @ block.w2.T
                    grad_w2 = h1[0].T @ d_ffn
                    grad_w1 = x_attn[0].T @ (d_h1 * relu_derivative(z1[0]))
                    for grad, param in (
                        (grad_w2, block.w2),
                        (grad_w1, block.w1),
                    ):
                        gnorm = np.linalg.norm(grad)
                        if gnorm > 1.0:
                            grad = grad * (1.0 / gnorm)
                        param -= ffn_lr * grad
                    block.b2 -= ffn_lr * np.sum(d_ffn, axis=0, keepdims=True)
                    block.b1 -= ffn_lr * np.sum(d_h1 * relu_derivative(z1[0]), axis=0, keepdims=True)
                    d_layer = ((d_h1 * relu_derivative(z1[0])) @ block.w1.T)[np.newaxis, :, :]

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
        cfg_raw = dict(payload["config"])
        cfg_raw.setdefault("model_version", 1)
        allowed = {f.name for f in AttentionLMConfig.__dataclass_fields__.values()}
        config = AttentionLMConfig(**{k: v for k, v in cfg_raw.items() if k in allowed})
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
