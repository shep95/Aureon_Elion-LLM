"""Attention language model and predict brain tests."""

from __future__ import annotations

import numpy as np

from brain.predict_engine import (
    BOOTSTRAP_LINES,
    _bootstrap_answer,
    is_prediction_question,
    predict_with_steps,
)
from src.attention_lm import AttentionLMConfig, StackedAttentionLM
from src.tokenizer import WordTokenizer


def test_tokenizer_encode_decode():
    tok = WordTokenizer()
    tok.build_vocab(["hello world", "hello france"], min_freq=1)
    ids = tok.encode("hello france")
    assert tok.decode(ids) == "hello france"


def test_attention_lm_forward_shape():
    tok = WordTokenizer()
    tok.build_vocab(BOOTSTRAP_LINES, min_freq=1, max_vocab=500)
    model = StackedAttentionLM.create(tok, AttentionLMConfig(d_model=16, n_layers=2, d_ff=32))
    ids = tok.encode("question what is the capital of france answer", add_bos=True)
    logits, attn = model.forward(__import__("numpy").array([ids]), return_attention=True)
    assert logits.shape[-1] == tok.vocab_size
    assert len(attn) == 2


def test_kv_cache_incremental_matches_full():
    tok = WordTokenizer()
    tok.build_vocab(BOOTSTRAP_LINES, min_freq=1, max_vocab=500)
    model = StackedAttentionLM.create(tok, AttentionLMConfig(d_model=16, n_layers=2, d_ff=32))
    ids = tok.encode("question what is the capital of france answer", add_bos=True)

    model.clear_cache()
    cached_logits = model.forward_cached(np.array([ids], dtype=int))
    full_logits, _ = model.forward(np.array([ids], dtype=int))
    assert cached_logits.shape == full_logits.shape
    np.testing.assert_allclose(cached_logits[0, -1], full_logits[0, -1], rtol=0.15, atol=0.6)

    model.clear_cache()
    model.forward_cached(np.array([ids], dtype=int))
    extra_id = tok.encode(" paris", add_bos=False, add_eos=False)[0]
    inc_logits = model.forward_cached(np.array([ids + [extra_id]], dtype=int))
    full2, _ = model.forward(np.array([ids + [extra_id]], dtype=int))
    np.testing.assert_allclose(inc_logits[0, -1], full2[0, -1], rtol=0.15, atol=0.6)


def test_speculative_generate_metadata(monkeypatch):
    monkeypatch.setenv("AUREON_SPECULATIVE_DRAFT", "4")
    monkeypatch.setenv("AUREON_SPECULATIVE_DECODE", "1")
    tok = WordTokenizer()
    tok.build_vocab(BOOTSTRAP_LINES, min_freq=1, max_vocab=500)
    cfg = AttentionLMConfig(d_model=24, n_layers=2, d_ff=48, learning_rate=0.12)
    model = StackedAttentionLM.create(tok, cfg)
    model.train(BOOTSTRAP_LINES, epochs=80)
    out = model.generate("question what is the capital of france answer", max_new_tokens=6)
    assert out.get("speculative", {}).get("mode") == "draft_verify"


def test_quantize_weights():
    tok = WordTokenizer()
    tok.build_vocab(["hello world"], min_freq=1)
    model = StackedAttentionLM.create(tok, AttentionLMConfig(d_model=8, n_layers=1, d_ff=16))
    model.quantize_weights()
    assert model.embeddings.dtype == np.float16
    assert model._quantized is True


def test_attention_lm_train_and_generate():
    tok = WordTokenizer()
    tok.build_vocab(BOOTSTRAP_LINES, min_freq=1, max_vocab=500)
    cfg = AttentionLMConfig(d_model=24, n_layers=2, d_ff=48, learning_rate=0.12)
    model = StackedAttentionLM.create(tok, cfg)
    history = model.train(BOOTSTRAP_LINES, epochs=120)
    assert history
    out = model.generate("question what is the capital of france answer", max_new_tokens=6)
    assert "steps" in out
    assert len(out["steps"]) >= 1


def test_bootstrap_philosophy_seeds():
    god = _bootstrap_answer("Who is God to you?")
    assert god
    assert "deity" in god.lower() or "god" in god.lower() or "corpus" in god.lower()
    math = _bootstrap_answer("What is math?")
    assert math
    assert "mathematics" in math.lower() or "numbers" in math.lower()


def test_is_prediction_question():
    assert is_prediction_question("What is the capital of France?")
    assert not is_prediction_question("What is Aureon?")
    assert not is_prediction_question("/help")


def test_predict_with_steps_capital_france(tmp_path, monkeypatch):
    import brain.predict_engine as pe

    monkeypatch.setenv("AUREON_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PIPELINE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("AUREON_PREDICT_EPOCHS", "80")
    monkeypatch.setenv("AUREON_PREDICT_MAX_SEQ", "128")
    monkeypatch.setenv("AUREON_PREDICT_D_MODEL", "48")
    monkeypatch.setenv("AUREON_PREDICT_LAYERS", "4")
    monkeypatch.setenv("AUREON_PREDICT_MAX_VOCAB", "2000")
    monkeypatch.setattr(pe, "MODEL_DIR", tmp_path / "models" / "predict_brain")
    monkeypatch.setattr(pe, "_model", None)
    monkeypatch.setattr(pe, "_ready", False)
    result = predict_with_steps("What is the capital of France?")
    assert result is not None
    assert "paris" in result["answer"].lower()
    assert len(result["pipeline"]) == 8
    assert result["pipeline"][2]["name"] == "context"
    assert result["pipeline"][5]["name"] == "reasoning"
    assert result["pipeline"][6]["name"] == "next_token"
