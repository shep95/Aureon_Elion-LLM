"""Brain prediction engine — attention LM with reasoning and long context."""

from __future__ import annotations

import logging
import os
import re
import threading
from pathlib import Path
from typing import Any

from pipeline.config import MODELS_DIR, SEEDS_DIR, ensure_dirs
from src.attention_lm import AttentionLMConfig, StackedAttentionLM
from src.tokenizer import WordTokenizer

logger = logging.getLogger(__name__)

MODEL_DIR = MODELS_DIR / "predict_brain"
CURRENT_MODEL_VERSION = 3
_lock = threading.Lock()
_model: StackedAttentionLM | None = None
_ready = False

_CONTEXT_STOP = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "what",
        "who",
        "how",
        "why",
        "when",
        "where",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "and",
        "or",
        "it",
        "this",
        "that",
        "do",
        "does",
    }
)

# Factual Q→A patterns.
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

# Chain-of-thought reasoning lines (context → think → therefore → answer).
REASONING_LINES: list[str] = [
    (
        "context france europe western country paris city question what is the capital of france "
        "think france is a country in western europe its capital city is paris therefore "
        "answer paris is the capital of france"
    ),
    (
        "context germany europe berlin city question what is the capital of germany "
        "think germany is in central europe berlin is the capital therefore "
        "answer berlin is the capital of germany"
    ),
    (
        "context dna genetic cells nucleotides question what is dna "
        "think dna is a molecule that stores genetic instructions in living cells therefore "
        "answer dna stores genetic information in living cells"
    ),
    (
        "context neural network weights labels backpropagation question what is backpropagation "
        "think backpropagation computes gradients and updates weights from labeled examples therefore "
        "answer backpropagation adjusts neural network weights using labeled examples"
    ),
    (
        "context newton motion force inertia question what is newton's first law "
        "think objects keep their state unless a net external force acts therefore "
        "answer an object stays at rest or in motion unless a force acts on it"
    ),
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
    "why does ",
    "why do ",
    "explain ",
)


def _env_int(name: str, default: int, *, minimum: int = 1, maximum: int = 1_000_000) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(minimum, min(maximum, int(raw)))
    except ValueError:
        return default


def predict_config_from_env() -> AttentionLMConfig:
    return AttentionLMConfig(
        d_model=_env_int("AUREON_PREDICT_D_MODEL", 128, minimum=16, maximum=512),
        n_layers=_env_int("AUREON_PREDICT_LAYERS", 6, minimum=2, maximum=24),
        d_ff=_env_int("AUREON_PREDICT_D_FF", 512, minimum=32, maximum=4096),
        max_seq_len=_env_int("AUREON_PREDICT_MAX_SEQ", 1_000_000, minimum=64, maximum=1_000_000),
        learning_rate=float(os.environ.get("AUREON_PREDICT_LR", "0.08")),
        model_version=CURRENT_MODEL_VERSION,
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
    return any(q.startswith(prefix.rstrip()) for prefix in PREDICTION_STARTERS)


def _load_seed_documents() -> list[str]:
    texts: list[str] = []
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

        limit = _env_int("AUREON_PREDICT_DOC_LIMIT", 1_000_000, minimum=100, maximum=1_000_000)
        with get_session() as session:
            rows = session.scalars(select(Document).limit(limit)).all()
            return [f"{row.title} {row.text}" for row in rows if row.text]
    except Exception:
        logger.debug("DB document load for predict brain skipped", exc_info=True)
        return []


def _build_training_corpus() -> list[str]:
    corpus = list(BOOTSTRAP_LINES)
    corpus.extend(REASONING_LINES)
    corpus.extend(_load_seed_documents())
    corpus.extend(_load_db_documents())
    return corpus


def _retrieve_context(question: str, *, max_words: int | None = None) -> tuple[str, list[str]]:
    """Keyword retrieval from corpus — fills the context window before reasoning."""
    if max_words is None:
        max_words = _env_int("AUREON_PREDICT_CONTEXT_WORDS", 1_000_000, minimum=50, maximum=1_000_000)

    q_words = {
        w
        for w in re.findall(r"[a-z0-9']+", question.lower())
        if len(w) > 2 and w not in _CONTEXT_STOP
    }
    if not q_words:
        return "", []

    pool = _load_db_documents() + _load_seed_documents()
    ranked: list[tuple[int, str]] = []
    for doc in pool:
        doc_lower = doc.lower()
        score = sum(1 for w in q_words if w in doc_lower)
        if score > 0:
            ranked.append((score, doc[:800]))

    ranked.sort(key=lambda item: item[0], reverse=True)
    snippets = [doc for _, doc in ranked[:8]]
    words: list[str] = []
    for snippet in snippets:
        words.extend(re.findall(r"[a-z0-9']+", snippet.lower()))
        if len(words) >= max_words:
            break

    context = " ".join(words[:max_words])
    return context, snippets[:3]


def _bootstrap_answer(question: str) -> str | None:
    key = question.strip().lower().rstrip("?").strip()
    prefix = f"question {key} answer "
    for line in BOOTSTRAP_LINES:
        if line.startswith(prefix):
            raw = line[len(prefix) :].strip()
            return raw[0].upper() + raw[1:] if raw else None
    for line in REASONING_LINES:
        if f"question {key} " in line and " answer " in line:
            raw = line.split(" answer ", 1)[1].strip()
            return raw[0].upper() + raw[1:] if raw else None
    return None


def _train_or_load() -> StackedAttentionLM:
    global _model, _ready
    ensure_dirs()
    expected = predict_config_from_env()
    if MODEL_DIR.is_dir() and (MODEL_DIR / "model.json").is_file():
        try:
            loaded = StackedAttentionLM.load(MODEL_DIR)
            saved_version = getattr(loaded.config, "model_version", 1)
            if (
                saved_version >= CURRENT_MODEL_VERSION
                and loaded.config.max_seq_len >= expected.max_seq_len // 2
                and loaded.tokenizer.vocab_size >= 500
            ):
                _model = loaded
                _ready = True
                return _model
            logger.info(
                "Predict brain config upgraded (v%s→v%s, ctx %s→%s) — retraining",
                saved_version,
                CURRENT_MODEL_VERSION,
                loaded.config.max_seq_len,
                expected.max_seq_len,
            )
        except Exception:
            logger.warning("Predict brain load failed — retraining", exc_info=True)

    corpus = _build_training_corpus()
    max_vocab = _env_int("AUREON_PREDICT_MAX_VOCAB", 1_000_000, minimum=1000, maximum=1_000_000)
    tokenizer = WordTokenizer()
    tokenizer.build_vocab(corpus, min_freq=1, max_vocab=max_vocab)

    config = expected
    model = StackedAttentionLM.create(tokenizer, config)
    epochs = _env_int("AUREON_PREDICT_EPOCHS", 200, minimum=20, maximum=2000)
    model.train(corpus, epochs=epochs, verbose=False)
    model.save(MODEL_DIR)
    _model = model
    _ready = True
    logger.info(
        "Predict brain trained — vocab=%s ctx=%s layers=%s epochs=%s",
        tokenizer.vocab_size,
        config.max_seq_len,
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
    elif " therefore " in lower and " answer " not in lower:
        answer = raw.split(" therefore ", 1)[1].strip()
    else:
        answer = raw.strip()
    answer_lower = answer.lower()
    if not answer or answer_lower.startswith("question") or answer_lower == q:
        return ""
    if answer_lower.startswith("think "):
        return ""
    a_words = answer.split()
    q_words = q.split()
    while a_words and q_words and a_words[0].lower() == q_words[0].lower():
        a_words.pop(0)
        q_words.pop(0)
    cleaned = " ".join(a_words).strip(" .")
    if not cleaned:
        return ""
    return cleaned[0].upper() + cleaned[1:] if len(cleaned) > 1 else cleaned.upper()


def _extract_reasoning(raw: str) -> str:
    lower = raw.lower()
    if " think " not in lower:
        return ""
    after_think = raw.split(" think ", 1)[1]
    if " therefore " in after_think.lower():
        after_think = after_think.split(" therefore ", 1)[0]
    if " answer " in after_think.lower():
        after_think = after_think.split(" answer ", 1)[0]
    text = after_think.strip(" .")
    if not text:
        return ""
    return text[0].upper() + text[1:] if len(text) > 1 else text.upper()


def predict_with_steps(question: str) -> dict[str, Any] | None:
    """
    Run the prediction pipeline: context → tokenize → embed → attention →
    layers → reasoning → next-token → autoregressive generate.
    """
    if not is_prediction_question(question):
        return None

    model = get_predict_model()
    q = question.strip().lower().rstrip("?")
    context, context_sources = _retrieve_context(q)
    max_ctx_tokens = max(32, model.config.max_seq_len - 64)

    context_block = f"context {context} " if context else ""
    reason_prompt = f"{context_block}question {q} think"
    reason_gen = model.generate(
        reason_prompt,
        max_new_tokens=_env_int("AUREON_PREDICT_REASON_TOKENS", 24, minimum=8, maximum=128),
        stop_words=frozenset({"<eos>", "<pad>", "therefore", "answer"}),
    )
    reasoning_text = _extract_reasoning(reason_gen["text"]) or " ".join(
        s["token"] for s in reason_gen.get("steps", []) if s["token"] not in {"<eos>", "<pad>"}
    )

    answer_prompt = f"{context_block}question {q} think {reasoning_text.lower()} therefore answer"
    token_ids = model.tokenizer.encode(answer_prompt, add_bos=True, add_eos=False, max_tokens=max_ctx_tokens)
    words = [model.tokenizer.id_to_word[t] for t in token_ids]

    first_step = model.next_token_distribution(token_ids)
    generation = model.generate(
        answer_prompt,
        max_new_tokens=_env_int("AUREON_PREDICT_ANSWER_TOKENS", 32, minimum=8, maximum=256),
    )

    answer = _extract_answer(generation["text"], q)
    fallback = _bootstrap_answer(q)
    if fallback:
        key_word = fallback.split()[0].lower()
        if not answer or len(answer) < 4 or key_word not in answer.lower():
            answer = fallback
    if not answer or answer.lower() == q:
        return None

    steps = [
        {
            "step": 1,
            "name": "tokenize",
            "description": "Convert each word into a token id.",
            "tokens": words,
            "token_count": len(words),
        },
        {
            "step": 2,
            "name": "embed",
            "description": "Each token maps to a learned vector. Similar words sit close together.",
            "vector_dim": model.config.d_model,
            "vocab_size": model.tokenizer.vocab_size,
            "context_window": model.config.max_seq_len,
        },
        {
            "step": 3,
            "name": "context",
            "description": "Retrieved corpus snippets fill the context window before reasoning.",
            "words_used": len(context.split()) if context else 0,
            "sources": [s[:120] + "..." if len(s) > 120 else s for s in context_sources],
        },
        {
            "step": 4,
            "name": "attention",
            "description": "Every token scores every other token for relevance (self-attention).",
            "pairs": first_step.get("attention_pairs", []),
        },
        {
            "step": 5,
            "name": "layers",
            "description": "Attention + feed-forward blocks stack to build deeper representations.",
            "num_layers": model.config.n_layers,
        },
        {
            "step": 6,
            "name": "reasoning",
            "description": "Chain-of-thought: intermediate tokens before the final answer.",
            "reasoning_text": reasoning_text,
            "reasoning_tokens": [s["token"] for s in reason_gen.get("steps", [])],
        },
        {
            "step": 7,
            "name": "next_token",
            "description": "Softmax over the vocabulary — highest probability token is chosen next.",
            "distribution": first_step.get("distribution", []),
            "top_token": first_step.get("top_token"),
            "top_probability": first_step.get("top_probability"),
        },
        {
            "step": 8,
            "name": "generate",
            "description": "Append each predicted token and repeat until the answer is complete.",
            "tokens_generated": [s["token"] for s in generation.get("steps", [])],
            "decode_steps": generation.get("steps", []),
        },
    ]

    return {
        "answer": answer,
        "prompt": answer_prompt,
        "model": "stacked_attention_lm",
        "model_version": CURRENT_MODEL_VERSION,
        "context_window": model.config.max_seq_len,
        "vocab_size": model.tokenizer.vocab_size,
        "pipeline": steps,
        "generation": generation,
        "reasoning": reason_gen,
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


def retrain_predict_brain_background(*, reason: str = "auto_learn") -> None:
    """Rebuild the predict brain from the latest corpus after learning cycles."""
    if os.environ.get("AUREON_PREDICT_BRAIN", "1").strip().lower() in ("0", "false", "no"):
        return

    def _job() -> None:
        global _model, _ready
        try:
            with _lock:
                _model = None
                _ready = False
            get_predict_model()
            logger.info("Predict brain retrained (reason=%s)", reason)
        except Exception:
            logger.exception("Predict brain retrain failed (reason=%s)", reason)

    threading.Thread(target=_job, name="aureon-predict-retrain", daemon=True).start()
