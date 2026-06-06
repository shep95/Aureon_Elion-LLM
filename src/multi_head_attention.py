"""Multi-head scaled dot-product attention with separate Q, K, V projections."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.neural_network import softmax


def _split_heads(x: np.ndarray, n_heads: int) -> np.ndarray:
    batch, seq, d_model = x.shape
    d_head = d_model // n_heads
    return x.reshape(batch, seq, n_heads, d_head).transpose(0, 2, 1, 3)


def _merge_heads(x: np.ndarray) -> np.ndarray:
    batch, n_heads, seq, d_head = x.shape
    return x.transpose(0, 2, 1, 3).reshape(batch, seq, n_heads * d_head)


def _causal_mask(seq_len: int) -> np.ndarray:
    return np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)


def _softmax_4d(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(np.clip(shifted, -40, 40))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


@dataclass
class MultiHeadAttention:
    """Issues 15–17: explicit Q/K/V matrices and multi-head attention."""

    n_heads: int
    d_model: int
    w_q: np.ndarray
    b_q: np.ndarray
    w_k: np.ndarray
    b_k: np.ndarray
    w_v: np.ndarray
    b_v: np.ndarray
    w_o: np.ndarray
    b_o: np.ndarray

    @classmethod
    def create(cls, d_model: int, n_heads: int, rng: np.random.Generator) -> MultiHeadAttention:
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        scale = np.sqrt(2.0 / d_model)
        return cls(
            n_heads=n_heads,
            d_model=d_model,
            w_q=rng.normal(0, scale, (d_model, d_model)),
            b_q=np.zeros((1, d_model)),
            w_k=rng.normal(0, scale, (d_model, d_model)),
            b_k=np.zeros((1, d_model)),
            w_v=rng.normal(0, scale, (d_model, d_model)),
            b_v=np.zeros((1, d_model)),
            w_o=rng.normal(0, scale, (d_model, d_model)),
            b_o=np.zeros((1, d_model)),
        )

    def _project(self, x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        return x @ w + b

    def forward(
        self,
        x: np.ndarray,
        *,
        return_weights: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None, dict[str, np.ndarray]]:
        batch, seq, _ = x.shape
        d_head = self.d_model // self.n_heads

        q = _split_heads(self._project(x, self.w_q, self.b_q), self.n_heads)
        k = _split_heads(self._project(x, self.w_k, self.b_k), self.n_heads)
        v = _split_heads(self._project(x, self.w_v, self.b_v), self.n_heads)

        scores = (q @ np.swapaxes(k, -1, -2)) / np.sqrt(d_head)
        scores = np.clip(scores, -40.0, 40.0)
        mask = _causal_mask(seq)
        scores = np.where(mask[np.newaxis, np.newaxis, :, :], -1e9, scores)
        weights = _softmax_4d(scores)
        context = weights @ v
        merged = _merge_heads(context)
        out = merged @ self.w_o + self.b_o

        cache = {"x": x, "merged": merged, "q": q, "k": k, "v": v, "weights": weights}
        if return_weights:
            avg_weights = weights.mean(axis=1)
            return out, avg_weights, cache
        return out, None, cache

    def forward_last(self, x: np.ndarray, *, window: int = 512) -> np.ndarray:
        """Attention output for the last query token only — incremental decode."""
        seq = x.shape[1]
        start = max(0, seq - window)
        x_win = x[:, start:, :]
        out, _, _ = self.forward(x_win, return_weights=False)
        return out[:, -1:, :]

    def backward(
        self,
        cache: dict[str, np.ndarray],
        d_out: np.ndarray,
        lr: float,
    ) -> np.ndarray:
        """Backprop through output projection and Q/K/V input projections."""
        x = cache["x"]
        merged = cache["merged"]
        attn_lr = lr * 0.2

        grad_wo = merged.reshape(-1, self.d_model).T @ d_out.reshape(-1, self.d_model)
        gnorm = np.linalg.norm(grad_wo)
        if gnorm > 1.0:
            grad_wo = grad_wo * (1.0 / gnorm)
        self.w_o -= attn_lr * grad_wo
        grad_bo = np.sum(d_out, axis=(0, 1), keepdims=True).reshape(self.b_o.shape)
        self.b_o -= attn_lr * grad_bo

        d_merged = d_out @ self.w_o.T
        d_x = d_merged @ self.w_v.T + d_merged @ self.w_k.T + d_merged @ self.w_q.T

        for w, b in ((self.w_v, self.b_v), (self.w_k, self.b_k), (self.w_q, self.b_q)):
            g_w = x.reshape(-1, self.d_model).T @ d_x.reshape(-1, self.d_model)
            gnorm = np.linalg.norm(g_w)
            if gnorm > 1.0:
                g_w = g_w * (1.0 / gnorm)
            w -= attn_lr * g_w
            grad_b = np.sum(d_x, axis=(0, 1), keepdims=True).reshape(b.shape)
            b -= attn_lr * grad_b

        return d_x
