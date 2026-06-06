"""Brain prediction engine — attention LM for next-token generation."""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

from pipeline.config import MODELS_DIR, SEEDS_DIR, ensure_dirs
from src.attention_lm import AttentionLMConfig, StackedAttentionLM
from src.tokenizer import WordTokenizer

logger = logging.getLogger(__name__)

MODEL_DIR = MODELS_DIR / "predict_brain"
_lock = threading.Lock()
_model: StackedAttentionLM | None = None
_ready = False

# Seed lines teach the LM factual Q→A patterns (including geography examples).
BOOTSTRAP_LINES: list[str] = [
    "question what is the capital of france answer paris is the capital of france",
    "question what is the capital of germany answer berlin is the capital of germany",
    "question what is the capital of italy answer rome is the capital of italy",
    "question what is the capital of spain answer madrid is the capital of spain",
    "question what is the capital of japan answer tokyo is the capital of japan",
    "question what is the capital of the united kingdom answer london is the capital of the united kingdom",
    "question what is dna answer dna stores genetic information in living cells",
    "question what is backpropagation answer backpropagation adjusts neural network weights using labeled examples",
    "question what is newton's first law answer an object stays at rest or in motion unless a force acts on it",
    "question what is matrix multiplication answer matrix multiplication combines rows and columns through dot products",
    "question what is big o notation answer big o describes how algorithm runtime grows with input size",
    "question what is supervised learning answer supervised learning uses input output pairs to train weights",
    "question what is the scientific revolution answer the scientific revolution shifted europe toward empirical observation",
    "question what is sanskrit grammar answer panini's ashtadhyayi describes sanskrit morphology with formal rules",
    "question what is a feedback loop answer feedback systems measure output and adjust inputs to reduce error",
]

PREDICTION_STARTERS = (
    "what is the capital of",
    "what is the",
    "what are the",
    "what is ",
    "what are ",
    "who is the",
    "who is ",
    "where is the",
    "where is ",
    "when was ",
    "how does ",
    "how do ",
)


def is_prediction_question(text: str) -> bool:
    """True for factual questions best answered by next-token prediction."""
    q = text.strip().lower().rstrip("?").strip()
    if not q or q.startswith("/"):
        return False
    excluded = {
        "what is aureon",
        "who are you",
        "what are you",
        "what is ai",
        "what is artificial intelligence",
    }
    if q in excluded or "what are you learning" in q or "what you learning" in q:
        return False
    if not text.strip().lower().endswith("?"):
        q = q  # still allow questions without ? if they match starters
    return any(q.startswith(prefix.rstrip()) for prefix in PREDICTION_STARTERS)


def _load_seed_documents() -> list[str]:
    texts = list(BOOTSTRAP_LINES)
    for name in ("corpus_seed.json", "corpus_seed_extra.json"):
        path = SEEDS_DIR / name
        if not path.is_file():
            continue
        try:
            import json

            payload = json.loads(path.read_text(encoding="utf-8"))
            for doc in payload.get("documents", []):
                title = str(doc.get("title", "")).strip()
                body = str(doc.get("text", "")).strip()
                if title and body:
                    texts.append(f"{title} {body}")
        except (OSError, ValueError, TypeError):
            logger.debug("Seed load skipped for %s", path, exc_info=True)
    return texts


def _load_db_documents() -> list[str]:
    try:
        from sqlalchemy import select

        from db.models import Document
        from db.session import get_session

        with get_session() as session:
            rows = session.scalars(select(Document).limit(500)).all()
            return [f"{row.title} {row.text}" for row in rows if row.text]
    except Exception:
        logger.debug("DB document load for predict brain skipped", exc_info=True)
        return []


def _build_training_corpus() -> list[str]:
    corpus = list(BOOTSTRAP_LINES)
    corpus.extend(_load_seed_documents())
    db_docs = _load_db_documents()
    corpus.extend(db_docs[:100])
    return corpus


def _bootstrap_answer(question: str) -> str | None:
    key = question.strip().lower().rstrip("?").strip()
    prefix = f"question {key} answer "
    for line in BOOTSTRAP_LINES:
        if line.startswith(prefix):
            raw = line[len(prefix) :].strip()
            return raw[0].upper() + raw[1:] if raw else None
    return None


def _train_or_load() -> StackedAttentionLM:
    global _model, _ready
    ensure_dirs()
    if MODEL_DIR.is_dir() and (MODEL_DIR / "model.json").is_file():
        try:
            _model = StackedAttentionLM.load(MODEL_DIR)
            _ready = True
            return _model
        except Exception:
            logger.warning("Predict brain load failed — retraining", exc_info=True)

    corpus = _build_training_corpus()
    tokenizer = WordTokenizer()
    tokenizer.build_vocab(corpus, min_freq=1, max_vocab=2000)

    config = AttentionLMConfig(
        d_model=int(os.environ.get("AUREON_PREDICT_D_MODEL", "48")),
        n_layers=int(os.environ.get("AUREON_PREDICT_LAYERS", "4")),
        d_ff=int(os.environ.get("AUREON_PREDICT_D_FF", "96")),
        max_seq_len=int(os.environ.get("AUREON_PREDICT_MAX_SEQ", "64")),
        learning_rate=float(os.environ.get("AUREON_PREDICT_LR", "0.12")),
    )
    model = StackedAttentionLM.create(tokenizer, config)
    epochs = int(os.environ.get("AUREON_PREDICT_EPOCHS", "180"))
    model.train(corpus, epochs=epochs, verbose=False)
    model.save(MODEL_DIR)
    _model = model
    _ready = True
    logger.info(
        "Predict brain trained — vocab=%s layers=%s epochs=%s",
        tokenizer.vocab_size,
        config.n_layers,
        epochs,
    )
    return model


def get_predict_model() -> StackedAttentionLM:
    global _model, _ready
    with _lock:
        if _model is not None and _ready:
            return _model
        return _train_or_load()


def _extract_answer(raw: str, question: str) -> str:
    """Pull the answer span after the 'answer' marker when present."""
    lower = raw.lower()
    q = question.lower().rstrip("?").strip()
    if " answer " in lower:
        answer = raw.split(" answer ", 1)[1].strip()
    else:
        answer = raw.strip()
    answer_lower = answer.lower()
    if not answer or answer_lower.startswith("question") or answer_lower == q:
        return ""
    # Drop echoed question words from the start of the generation.
    a_words = answer.split()
    q_words = q.split()
    while a_words and q_words and a_words[0].lower() == q_words[0].lower():
        a_words.pop(0)
        q_words.pop(0)
    cleaned = " ".join(a_words).strip(" .")
    if not cleaned:
        return ""
    return cleaned[0].upper() + cleaned[1:] if len(cleaned) > 1 else cleaned.upper()


def predict_with_steps(question: str) -> dict[str, Any] | None:
    """
    Run the six-step prediction pipeline and return answer + interpretability metadata.
    """
    if not is_prediction_question(question):
        return None

    model = get_predict_model()
    q = question.strip().lower().rstrip("?")
    prompt = f"question {q} answer"

    token_ids = model.tokenizer.encode(prompt, add_bos=True, add_eos=False)
    words = [model.tokenizer.id_to_word[t] for t in token_ids]

    first_step = model.next_token_distribution(token_ids)
    generation = model.generate(prompt, max_new_tokens=14)

    answer = _extract_answer(generation["text"], q)
    fallback = _bootstrap_answer(q)
    if fallback and not answer:
        answer = fallback
    if not answer or answer.lower() == q:
        return None

    steps = [
        {
            "step": 1,
            "name": "tokenize",
            "description": "Convert each word into a token id.",
            "tokens": words,
        },
        {
            "step": 2,
            "name": "embed",
            "description": "Each token maps to a learned vector (embedding). Similar words sit close together.",
            "vector_dim": model.config.d_model,
            "vocab_size": model.tokenizer.vocab_size,
        },
        {
            "step": 3,
            "name": "attention",
            "description": "Every token scores every other token for relevance (self-attention).",
            "pairs": first_step.get("attention_pairs", []),
        },
        {
            "step": 4,
            "name": "layers",
            "description": "Attention + feed-forward blocks stack to build deeper representations.",
            "num_layers": model.config.n_layers,
        },
        {
            "step": 5,
            "name": "next_token",
            "description": "Softmax over the vocabulary — highest probability token is chosen next.",
            "distribution": first_step.get("distribution", []),
            "top_token": first_step.get("top_token"),
            "top_probability": first_step.get("top_probability"),
        },
        {
            "step": 6,
            "name": "generate",
            "description": "Append each predicted token and repeat until the answer is complete.",
            "tokens_generated": [s["token"] for s in generation.get("steps", [])],
            "decode_steps": generation.get("steps", []),
        },
    ]

    return {
        "answer": answer,
        "prompt": prompt,
        "model": "stacked_attention_lm",
        "pipeline": steps,
        "generation": generation,
    }


def warmup_predict_brain_background() -> None:
    """Train/load the predict brain without blocking the request path."""
    if os.environ.get("AUREON_PREDICT_BRAIN", "1").strip().lower() in ("0", "false", "no"):
        return

    def _job() -> None:
        try:
            get_predict_model()
        except Exception:
            logger.exception("Predict brain warmup failed")

    threading.Thread(target=_job, name="aureon-predict-brain", daemon=True).start()
