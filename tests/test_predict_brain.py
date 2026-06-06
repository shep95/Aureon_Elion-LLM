"""Attention language model and predict brain tests."""

from __future__ import annotations

from brain.predict_engine import (
    BOOTSTRAP_LINES,
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
