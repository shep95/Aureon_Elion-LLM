"""
Stacked self-attention language model (NumPy, from scratch).

Pipeline per user message:
  1. Tokenize words → ids
  2. Embed ids → vectors (d_model)
  3. Self-attention — each token weights every other token (KV cache + sliding window)
  4. Stack layers (attention + feed-forward + residual)
  5. Softmax over vocabulary → next-token probabilities
  6. Autoregressive decode — append token, repeat (speculative draft + verify)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.efficient_inference import (
    attention_window,
    inference_profile,
    sliding_window_attention,
    speculative_draft_tokens,
    truncate_tokens_for_inference,
)
from src.multi_head_attention import MultiHeadAttention
from src.neural_network import relu, relu_derivative, softmax
from src.tokenizer import WordTokenizer


def _softmax_3d(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def _causal_mask(seq_len: int) -> np.ndarray:
    """True where attention must be blocked (future positions)."""
    return np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)


def _use_speculative_decode() -> bool:
    return os.environ.get("AUREON_SPECULATIVE_DECODE", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


@dataclass
class KVCache:
    """Per-layer hidden states for incremental decode — never recompute past tokens."""

    seq_len: int = 0
    embeddings: np.ndarray | None = None
    hidden: list[np.ndarray] = field(default_factory=list)
    last_logits: np.ndarray | None = None

    def clear(self) -> None:
        self.seq_len = 0
        self.embeddings = None
        self.hidden = []
        self.last_logits = None


@dataclass
class AttentionLMConfig:
    d_model: int = 128
    n_layers: int = 6
    n_heads: int = 4
    d_ff: int = 512
    max_seq_len: int = 1_000_000
    learning_rate: float = 0.05
    seed: int = 42
    model_version: int = 7


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

    def forward(self, x: np.ndarray, *, compute_dtype: type = np.float32) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x32 = x.astype(compute_dtype)
        z1 = x32 @ self.w1.astype(compute_dtype) + self.b1.astype(compute_dtype)
        h1 = relu(z1)
        out = h1 @ self.w2.astype(compute_dtype) + self.b2.astype(compute_dtype)
        return out.astype(x.dtype, copy=False), z1, h1


@dataclass
class StackedAttentionLM:
    """Causal language model with multi-layer self-attention."""

    config: AttentionLMConfig
    tokenizer: WordTokenizer
    embeddings: np.ndarray
    w_out: np.ndarray
    b_out: np.ndarray
    layers: list[FeedForwardBlock] = field(default_factory=list)
    attn_layers: list[MultiHeadAttention] = field(default_factory=list)
    _kv_cache: KVCache = field(default_factory=KVCache, repr=False)
    _quantized: bool = field(default=False, repr=False)

    @classmethod
    def create(cls, tokenizer: WordTokenizer, config: AttentionLMConfig | None = None) -> StackedAttentionLM:
        cfg = config or AttentionLMConfig()
        rng = np.random.default_rng(cfg.seed)
        vocab = max(tokenizer.vocab_size, 4)
        emb_scale = 0.02
        n_heads = max(1, cfg.n_heads)
        if cfg.d_model % n_heads != 0:
            n_heads = 1
        model = cls(
            config=cfg,
            tokenizer=tokenizer,
            embeddings=rng.normal(0, emb_scale, (vocab, cfg.d_model)),
            layers=[FeedForwardBlock.create(cfg.d_model, cfg.d_ff, rng) for _ in range(cfg.n_layers)],
            attn_layers=[
                MultiHeadAttention.create(cfg.d_model, n_heads, rng) for _ in range(cfg.n_layers)
            ],
            w_out=rng.normal(0, emb_scale, (cfg.d_model, vocab)),
            b_out=np.zeros((1, vocab)),
        )
        return model

    def clear_cache(self) -> None:
        """Call at the start of each new prompt / conversation."""
        self._kv_cache.clear()

    def quantize_weights(self) -> None:
        """Compress weights to float16 — half memory, cast to float32 only during matmul."""
        self.embeddings = self.embeddings.astype(np.float16)
        self.w_out = self.w_out.astype(np.float16)
        self.b_out = self.b_out.astype(np.float16)
        for layer in self.layers:
            layer.w1 = layer.w1.astype(np.float16)
            layer.b1 = layer.b1.astype(np.float16)
            layer.w2 = layer.w2.astype(np.float16)
            layer.b2 = layer.b2.astype(np.float16)
        self._quantized = True

    def _compute_dtype(self) -> type:
        return np.float32 if self._quantized else np.float64

    def _cast_weight(self, w: np.ndarray) -> np.ndarray:
        if self._quantized and w.dtype == np.float16:
            return w.astype(np.float32)
        return w

    def _output_logits(self, x: np.ndarray) -> np.ndarray:
        w = self._cast_weight(self.w_out)
        b = self._cast_weight(self.b_out)
        x32 = x.astype(np.float32) if self._quantized else x
        return x32 @ w + b

    def _embed(self, token_ids: np.ndarray) -> np.ndarray:
        emb = self._cast_weight(self.embeddings) if self._quantized else self.embeddings
        x = emb[token_ids].astype(np.float32) if self._quantized else emb[token_ids]
        seq = x.shape[1]
        x = x + self._positional_encoding(seq)[np.newaxis, :, :]
        return x

    def _embed_at_position(self, token_id: int, position: int) -> np.ndarray:
        emb = self._cast_weight(self.embeddings) if self._quantized else self.embeddings
        vec = emb[int(token_id)].astype(np.float32) if self._quantized else emb[int(token_id)]
        pe = self._positional_encoding(position + 1)[position]
        return (vec + pe)[np.newaxis, np.newaxis, :]

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

    def _self_attention_dense(
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

    def _self_attention_last_only(self, x: np.ndarray) -> np.ndarray:
        """Attention output for the last query position only — O(window) not O(seq²)."""
        window = attention_window()
        seq = x.shape[1]
        q_idx = seq - 1
        start = max(0, q_idx - window + 1)
        x_norm = self._layer_norm(x)
        keys = x_norm[:, start : q_idx + 1, :]
        query = x_norm[:, q_idx : q_idx + 1, :]
        d_model = x.shape[-1]
        scores = (query @ np.swapaxes(keys, -1, -2)) / np.sqrt(d_model)
        scores = np.clip(scores, -40.0, 40.0)
        weights = _softmax_3d(scores)
        return weights @ x[:, start : q_idx + 1, :]

    def _self_attention(
        self, x: np.ndarray, *, return_weights: bool = False
    ) -> tuple[np.ndarray, np.ndarray | None]:
        window = attention_window()
        seq = x.shape[1]
        if seq <= window:
            return self._self_attention_dense(x, return_weights=return_weights)
        x_norm = self._layer_norm(x)
        out, weights = sliding_window_attention(
            x_norm, window=window, softmax_3d=_softmax_3d
        )
        if return_weights:
            return out, weights
        return out, None

    def _forward_full_fill_cache(self, token_ids: np.ndarray) -> np.ndarray:
        """Full forward pass — populate KV cache for incremental decode."""
        if token_ids.ndim == 1:
            token_ids = token_ids[np.newaxis, :]

        seq = token_ids.shape[1]
        if seq > self.config.max_seq_len:
            token_ids = token_ids[:, -self.config.max_seq_len :]
            seq = token_ids.shape[1]

        x = self._embed(token_ids)
        cache = self._kv_cache
        cache.clear()
        cache.embeddings = x.copy()
        cache.hidden = []

        for li, block in enumerate(self.layers):
            if li < len(self.attn_layers):
                attn_out, _, _ = self.attn_layers[li].forward(x, return_weights=False)
            else:
                attn_out, _ = self._self_attention(x, return_weights=False)
            x = x + attn_out
            ffn_out, _, _ = block.forward(x, compute_dtype=self._compute_dtype())
            x = x + ffn_out
            cache.hidden.append(x.copy())

        logits = self._output_logits(x)
        cache.last_logits = logits
        cache.seq_len = seq
        return logits

    def _forward_incremental(self, token_id: int, position: int) -> np.ndarray:
        """One new token — reuse cached K/V via stored hidden states."""
        cache = self._kv_cache
        x = self._embed_at_position(token_id, position)

        for li, block in enumerate(self.layers):
            if li == 0:
                cache.embeddings = (
                    np.concatenate([cache.embeddings, x], axis=1)
                    if cache.embeddings is not None
                    else x.copy()
                )
                x_in = cache.embeddings
            else:
                past = cache.hidden[li - 1]
                x_in = np.concatenate([past, x], axis=1)

            attn_out = (
                self.attn_layers[li].forward_last(x_in, window=attention_window())
                if li < len(self.attn_layers)
                else self._self_attention_last_only(x_in)
            )
            x = x + attn_out
            ffn_out, _, _ = block.forward(x, compute_dtype=self._compute_dtype())
            x = x + ffn_out

            if li < len(cache.hidden):
                cache.hidden[li] = np.concatenate([cache.hidden[li], x], axis=1)
            else:
                cache.hidden.append(x.copy())

        logits = self._output_logits(x)
        cache.last_logits = logits
        cache.seq_len = position + 1
        return logits

    def forward_cached(self, token_ids: np.ndarray) -> np.ndarray:
        """Cached forward — full pass on new prompt, incremental when seq grows by 1."""
        if token_ids.ndim == 1:
            token_ids = token_ids[np.newaxis, :]

        seq = token_ids.shape[1]
        cache = self._kv_cache

        if seq == 0:
            raise ValueError("empty token sequence")

        if cache.seq_len == 0 or seq < cache.seq_len:
            return self._forward_full_fill_cache(token_ids)

        if seq == cache.seq_len and cache.last_logits is not None:
            return cache.last_logits

        if seq == cache.seq_len + 1:
            return self._forward_incremental(int(token_ids[0, -1]), seq - 1)

        return self._forward_full_fill_cache(token_ids)

    def forward(
        self, token_ids: np.ndarray, *, return_attention: bool = False, use_cache: bool = False
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Return logits (batch, seq, vocab) and optional attention per layer."""
        if use_cache and not return_attention:
            return self.forward_cached(token_ids), []

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
            ffn_out, _, _ = block.forward(x, compute_dtype=self._compute_dtype())
            x = x + ffn_out

        logits = self._output_logits(x)
        return logits, attentions

    def next_token_distribution(
        self, token_ids: list[int] | np.ndarray, *, top_k: int = 5, use_cache: bool = True
    ) -> dict[str, Any]:
        arr = np.array(token_ids, dtype=int)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        if use_cache:
            logits = self.forward_cached(arr)
        else:
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
        if not use_cache and arr.shape[1] > 1:
            _, attentions = self.forward(arr, return_attention=True)
            if attentions:
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

    def _next_token_id(self, token_ids: list[int], *, use_cache: bool = True) -> int:
        arr = np.array([token_ids], dtype=int)
        if use_cache:
            logits = self.forward_cached(arr)
        else:
            logits, _ = self.forward(arr, return_attention=False)
        return int(np.argmax(softmax(logits)[0, -1]))

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
        token_ids = truncate_tokens_for_inference(token_ids, max_window=attention_window())
        generated_steps: list[dict[str, Any]] = []

        self.clear_cache()
        draft_n = speculative_draft_tokens()
        use_spec = draft_n > 1 and _use_speculative_decode()
        spec_accepted = 0
        spec_drafted = 0

        base_len = len(token_ids)
        self.forward_cached(np.array([token_ids], dtype=int))

        while len(generated_steps) < max_new_tokens:
            if use_spec:
                drafts: list[int] = []
                draft_ids = list(token_ids)
                for _ in range(draft_n):
                    if len(generated_steps) + len(drafts) >= max_new_tokens:
                        break
                    next_id = self._next_token_id(draft_ids, use_cache=True)
                    word = self.tokenizer.id_to_word[next_id]
                    drafts.append(next_id)
                    spec_drafted += 1
                    if word in stop_words:
                        break
                    draft_ids.append(next_id)

                if not drafts:
                    break

                verify_logits, _ = self.forward(np.array([token_ids + drafts], dtype=int))
                accepted = 0
                for i, draft_token in enumerate(drafts):
                    pos = base_len + i
                    full_choice = int(np.argmax(softmax(verify_logits)[0, pos - 1]))
                    if full_choice == draft_token:
                        accepted += 1
                    else:
                        break

                take = max(1, accepted)
                spec_accepted += take
                chosen = drafts[:take]

                self.clear_cache()
                token_ids.extend(chosen)
                self.forward_cached(np.array([token_ids], dtype=int))

                for next_id in chosen:
                    word = self.tokenizer.id_to_word[next_id]
                    step_info = self.next_token_distribution(token_ids, use_cache=True)
                    generated_steps.append(
                        {
                            "token": word,
                            "distribution": step_info["distribution"],
                            "attention_pairs": step_info["attention_pairs"],
                        }
                    )
                    if word in stop_words:
                        break
                if generated_steps and generated_steps[-1]["token"] in stop_words:
                    break
                base_len = len(token_ids)
            else:
                step_info = self.next_token_distribution(token_ids, use_cache=True)
                next_id = self._next_token_id(token_ids, use_cache=True)
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
        result: dict[str, Any] = {
            "prompt": prompt,
            "token_ids": token_ids,
            "text": self.tokenizer.decode(answer_tokens),
            "steps": generated_steps,
            "inference": inference_profile(len(token_ids)),
        }
        if use_spec:
            result["speculative"] = {
                "draft_tokens": draft_n,
                "drafted_total": spec_drafted,
                "accepted_total": spec_accepted,
                "mode": "draft_verify",
            }
        return result

    def generate_speculative(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 12,
        stop_words: frozenset[str] | None = None,
    ) -> dict[str, Any]:
        """Speculative decode — generate() already uses draft+verify when enabled."""
        return self.generate(prompt, max_new_tokens=max_new_tokens, stop_words=stop_words)

    def train(self, texts: list[str], *, epochs: int = 80, verbose: bool = False) -> list[dict[str, float]]:
        """Next-token cross-entropy — backprop through output head, embeddings, and FFN layers."""
        self.clear_cache()
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
                attn_caches: list[dict[str, np.ndarray]] = []

                for li, block in enumerate(self.layers):
                    if li < len(self.attn_layers):
                        attn_out, attn_w, attn_cache = self.attn_layers[li].forward(x, return_weights=True)
                        attn_caches.append(attn_cache)
                    else:
                        attn_out, attn_w = self._self_attention(x, return_weights=True)
                        attn_caches.append({})
                    x_attn = x + attn_out
                    ffn_out, z1, h1 = block.forward(x_attn, compute_dtype=np.float64)
                    layer_caches.append((x, attn_w if attn_w is not None else np.zeros((1, seq_len, seq_len)), x_attn, z1, h1))
                    x = x_attn + ffn_out

                logits = self._output_logits(x)
                probs = softmax(logits)[0]
                eps = 1e-9
                loss = -np.mean(np.log(probs[np.arange(seq_len), targets] + eps))
                total_loss += float(loss)
                total_correct += int(np.sum(np.argmax(probs, axis=1) == targets))
                total_tokens += seq_len

                d_logits = probs.copy()
                d_logits[np.arange(seq_len), targets] -= 1.0
                d_logits /= seq_len

                w_out = self.w_out.astype(np.float64)
                d_hidden = d_logits @ w_out.T
                self.w_out = (w_out - lr * (x[0].T @ d_logits)).astype(self.w_out.dtype)
                self.b_out = (self.b_out.astype(np.float64) - lr * np.sum(d_logits, axis=0, keepdims=True)).astype(
                    self.b_out.dtype
                )

                emb = self.embeddings.astype(np.float64)
                for t_idx, token_id in enumerate(x_ids[0]):
                    emb[int(token_id)] -= lr * d_hidden[t_idx]
                    norm = np.linalg.norm(emb[int(token_id)])
                    if norm > 3.0:
                        emb[int(token_id)] *= 3.0 / norm
                self.embeddings = emb.astype(self.embeddings.dtype)

                ffn_lr = lr * 0.25
                d_layer = d_hidden[np.newaxis, :, :]
                for li in range(len(self.layers) - 1, -1, -1):
                    block = self.layers[li]
                    _, _, x_attn, z1, h1 = layer_caches[li]
                    d_ffn = d_layer[0]
                    d_h1 = d_ffn @ block.w2.astype(np.float64).T
                    grad_w2 = h1[0].T @ d_ffn
                    grad_w1 = x_attn[0].T @ (d_h1 * relu_derivative(z1[0]))
                    for grad, param_name in ((grad_w2, "w2"), (grad_w1, "w1")):
                        param = getattr(block, param_name)
                        gnorm = np.linalg.norm(grad)
                        if gnorm > 1.0:
                            grad = grad * (1.0 / gnorm)
                        setattr(block, param_name, (param.astype(np.float64) - ffn_lr * grad).astype(param.dtype))
                    block.b2 = (block.b2.astype(np.float64) - ffn_lr * np.sum(d_ffn, axis=0, keepdims=True)).astype(
                        block.b2.dtype
                    )
                    block.b1 = (
                        block.b1.astype(np.float64)
                        - ffn_lr * np.sum(d_h1 * relu_derivative(z1[0]), axis=0, keepdims=True)
                    ).astype(block.b1.dtype)
                    d_layer = ((d_h1 * relu_derivative(z1[0])) @ block.w1.astype(np.float64).T)[np.newaxis, :, :]

                    if li < len(self.attn_layers) and attn_caches[li]:
                        d_layer = self.attn_layers[li].backward(attn_caches[li], d_layer, lr)

            acc = total_correct / max(total_tokens, 1)
            metrics = {"epoch": float(epoch), "loss": total_loss / len(sequences), "accuracy": acc}
            history.append(metrics)
            if verbose and epoch % 20 == 0:
                print(f"  lm epoch {epoch:3d}  loss={metrics['loss']:.4f}  token_acc={acc:.1%}")

        if os.environ.get("AUREON_QUANTIZE_INFER", "1").strip().lower() not in ("0", "false", "no", "off"):
            self.quantize_weights()

        return history

    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(path / "tokenizer.json")
        payload = {
            "config": self.config.__dict__,
            "quantized": self._quantized,
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
            "attn_layers": [
                {
                    "n_heads": a.n_heads,
                    "d_model": a.d_model,
                    "w_q": a.w_q.tolist(),
                    "b_q": a.b_q.tolist(),
                    "w_k": a.w_k.tolist(),
                    "b_k": a.b_k.tolist(),
                    "w_v": a.w_v.tolist(),
                    "b_v": a.b_v.tolist(),
                    "w_o": a.w_o.tolist(),
                    "b_o": a.b_o.tolist(),
                }
                for a in self.attn_layers
            ],
        }
        (path / "model.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> StackedAttentionLM:
        path = Path(directory)
        tok_path = path / "tokenizer.json"
        payload_raw = json.loads((path / "model.json").read_text(encoding="utf-8"))
        if payload_raw.get("config", {}).get("model_version", 1) >= 7:
            from src.bpe_tokenizer import BPETokenizer

            tokenizer = BPETokenizer.load(tok_path)
        else:
            tokenizer = WordTokenizer.load(tok_path)
        payload = payload_raw
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
        model.attn_layers = []
        for attn in payload.get("attn_layers", []):
            model.attn_layers.append(
                MultiHeadAttention(
                    n_heads=int(attn["n_heads"]),
                    d_model=int(attn["d_model"]),
                    w_q=np.array(attn["w_q"], dtype=float),
                    b_q=np.array(attn["b_q"], dtype=float),
                    w_k=np.array(attn["w_k"], dtype=float),
                    b_k=np.array(attn["b_k"], dtype=float),
                    w_v=np.array(attn["w_v"], dtype=float),
                    b_v=np.array(attn["b_v"], dtype=float),
                    w_o=np.array(attn["w_o"], dtype=float),
                    b_o=np.array(attn["b_o"], dtype=float),
                )
            )
        model._quantized = bool(payload.get("quantized", False))
        if model.embeddings.dtype == np.float16:
            model._quantized = True
        return model
