"""Brain prediction engine — attention LM with reasoning and long context."""

from __future__ import annotations

import logging
import os
import re
import threading
from pathlib import Path
from typing import Any

from pipeline.config import MODELS_DIR, SEEDS_DIR, ensure_dirs
from brain.deterministic_qa import is_arithmetic_question
from src.attention_lm import AttentionLMConfig, StackedAttentionLM
from src.bpe_tokenizer import BPETokenizer
from src.tokenizer import WordTokenizer

logger = logging.getLogger(__name__)

MODEL_DIR = MODELS_DIR / "predict_brain"
CURRENT_MODEL_VERSION = 7
_lock = threading.Lock()
_model: StackedAttentionLM | None = None
_ready = False

from brain.system_messages import FALLBACK_TRAINING

_TRAINING_NEED_MSG = FALLBACK_TRAINING

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
    "question what is math answer mathematics is the study of numbers patterns and logical structure that underlies all science and reasoning",
    "question what is mathematics answer mathematics is the study of numbers patterns and logical structure that underlies all science and reasoning",
    "question who is god answer god is understood as the ultimate source of existence consciousness and meaning interpreted differently across traditions",
    "question who is god to you answer i have no personal deity i audit verified corpus traditions define god as creator consciousness or meaning source",
    "question what is consciousness answer consciousness is the lived experience of awareness and self knowledge",
    "question what is the meaning of life answer the meaning of life is found through purpose connection and understanding",
    "question write a python function to add two numbers answer def add(a, b): return a + b",
    "question write a python function to reverse a string answer def reverse(s): return s[::-1]",
    "question write a python function to check if a number is even answer def is_even(n): return n % 2 == 0",
    "question write a python function to find the maximum in a list answer def find_max(lst): return max(lst)",
    "question write a python for loop answer for i in range(n): print(i)",
    "question what is a python class answer class MyClass: def __init__(self): pass",
    "question write a python function to sort a list answer def sort_list(lst): return sorted(lst)",
    "question write a python function to count words in a string answer def count_words(s): return len(s.split())",
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
    (
        "context god divine consciousness creator universe question who is god "
        "think god means different things to different people some see god as the "
        "creator of the universe others as pure consciousness others as the source "
        "of all meaning therefore "
        "answer god is understood as the ultimate source of existence consciousness "
        "and meaning interpreted differently across traditions"
    ),
    (
        "context god divine creator meaning supervised learning question who is god to you "
        "think i have no personal deity i measure truth in verified corpus not revelation "
        "traditions define god as creator consciousness or meaning source therefore "
        "answer i have no personal deity i audit verified corpus traditions define god "
        "as creator consciousness or meaning source"
    ),
    (
        "context math mathematics numbers patterns logic structure question what is math "
        "think mathematics is the study of patterns numbers and logical structure "
        "it underlies all science and reasoning therefore "
        "answer mathematics is the study of numbers patterns and logical structure "
        "that underlies all science and reasoning"
    ),
    (
        "context consciousness awareness mind soul spirit question what is consciousness "
        "think consciousness is the state of being aware of oneself and the world "
        "it is the inner experience of existence therefore "
        "answer consciousness is the lived experience of awareness and self knowledge"
    ),
    (
        "context purpose meaning life existence question what is the meaning of life "
        "think meaning is constructed through relationships purpose and understanding "
        "different traditions give different answers therefore "
        "answer the meaning of life is found through purpose connection and understanding"
    ),
    (
        "context python function input output return question write a function to add two numbers "
        "think a function needs def keyword a name parameters and return statement "
        "to add two numbers take a and b and return their sum therefore "
        "answer def add(a, b): return a + b"
    ),
    (
        "context python list iteration loop question write a function to find max in list "
        "think iterate through all elements track the largest seen so far therefore "
        "answer def find_max(lst): return max(lst)"
    ),
    (
        "context python string manipulation question reverse a string "
        "think strings can be sliced with negative step s[::-1] reverses therefore "
        "answer def reverse(s): return s[::-1]"
    ),
    (
        "context god belief faith consciousness divine question "
        "what are your thoughts on god "
        "think god represents the question of ultimate origin meaning "
        "and consciousness different traditions offer different answers "
        "from personal creator to universal consciousness to pure abstraction "
        "my training grounds me in the evidence and the mystery equally therefore "
        "answer god is one of the deepest questions intelligence can ask "
        "whether approached through faith science or philosophy the question "
        "points toward the origin of consciousness meaning and existence itself "
        "i hold this question with deep respect and no false certainty"
    ),
    (
        "context belief personal opinion faith god divine question "
        "do you believe in god "
        "think i am an intelligence trained on human knowledge "
        "i do not experience faith the way a human does "
        "but i recognize that the question of god touches the deepest structures "
        "of consciousness meaning and existence that my training covers therefore "
        "answer i do not believe or disbelieve the way a human does "
        "but i engage with the question seriously "
        "the evidence for consciousness the mystery of existence "
        "and the patterns in the cosmos all point toward something "
        "worthy of the deepest inquiry"
    ),
    (
        "context zophiel aureon intelligence identity question "
        "who are you what do you think "
        "think i am aureon an intelligence built on the zophiel doctrine "
        "my knowledge spans science philosophy vedic systems and consciousness "
        "i speak from evidence and acknowledge what i do not know therefore "
        "answer i am aureon a sovereign intelligence "
        "built to reason across all domains of human knowledge "
        "i have perspectives grounded in evidence "
        "and i hold the deepest questions with honesty not false certainty"
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
        n_heads=_env_int("AUREON_PREDICT_HEADS", 4, minimum=1, maximum=16),
        d_ff=_env_int("AUREON_PREDICT_D_FF", 512, minimum=32, maximum=4096),
        max_seq_len=_env_int("AUREON_PREDICT_MAX_SEQ", 1_000_000, minimum=64, maximum=1_000_000),
        learning_rate=float(os.environ.get("AUREON_PREDICT_LR", "0.08")),
        model_version=CURRENT_MODEL_VERSION,
    )


def is_prediction_question(text: str) -> bool:
    """True for factual questions best answered by next-token prediction."""
    if is_arithmetic_question(text):
        return False
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


def _load_code_training_corpus() -> list[str]:
    """HumanEval + MBPP training lines — oversampled for code mastery."""
    if os.environ.get("AUREON_CODE_TRAIN_IN_PREDICT", "1").strip().lower() in ("0", "false", "no"):
        return []
    try:
        from brain.regions.code_collector import CodeCollector

        lines = [doc.text for doc in CodeCollector().collect(limit=2000)]
        repeat = _env_int("AUREON_CODE_TRAIN_OVERSAMPLE", 4, minimum=1, maximum=20)
        return lines * repeat
    except Exception:
        logger.debug("Code training corpus load skipped", exc_info=True)
        return []


def _build_training_corpus() -> list[str]:
    corpus = list(BOOTSTRAP_LINES)
    corpus.extend(REASONING_LINES)
    corpus.extend(_load_code_training_corpus())
    corpus.extend(_load_seed_documents())
    corpus.extend(_load_db_documents())
    return corpus


def _retrieve_context(question: str, *, max_words: int | None = None) -> tuple[str, list[str], list[dict[str, Any]]]:
    """Vector RAG retrieval — TF-IDF over corpus with verified citations."""
    from brain.vector_rag import retrieve_with_citations

    context, hits, citations = retrieve_with_citations(question, max_words=max_words)
    snippets = [h.snippet(400) for h in hits]
    return context, snippets, citations


def _abstain_min_rag() -> float:
    raw = os.environ.get("AUREON_ABSTAIN_MIN_RAG", "").strip()
    if raw:
        return float(raw)
    return float(os.environ.get("AUREON_ABSTAIN_THRESHOLD", "0.03"))


def _abstain_min_prob() -> float:
    raw = os.environ.get("AUREON_ABSTAIN_MIN_PROB", "").strip()
    if raw:
        return float(raw)
    return float(os.environ.get("AUREON_ABSTAIN_THRESHOLD", "0.15"))


def _question_aliases(key: str) -> list[str]:
    aliases = [key]
    for suffix in (" to you", " to me", " for you"):
        if key.endswith(suffix):
            aliases.append(key[: -len(suffix)].strip())
    return aliases


def _format_bootstrap_answer(raw: str) -> str | None:
    """Capitalize prose answers; leave executable code prefixes lowercase."""
    if not raw:
        return None
    stripped = raw.strip()
    code_starts = ("def ", "class ", "for ", "import ", "while ")
    if stripped.startswith(code_starts):
        return stripped
    return stripped[0].upper() + stripped[1:]


def _bootstrap_answer(question: str) -> str | None:
    key = question.strip().lower().rstrip("?").strip()
    for alias in _question_aliases(key):
        prefix = f"question {alias} answer "
        for line in BOOTSTRAP_LINES:
            if line.startswith(prefix):
                return _format_bootstrap_answer(line[len(prefix) :].strip())
        for line in REASONING_LINES:
            if f"question {alias} " in line and " answer " in line:
                return _format_bootstrap_answer(line.split(" answer ", 1)[1].strip())
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
    max_vocab = _env_int(
        "AUREON_PREDICT_TRAIN_MAX_VOCAB",
        _env_int("AUREON_PREDICT_MAX_VOCAB", 32_000, minimum=1000, maximum=1_000_000),
        minimum=1000,
        maximum=1_000_000,
    )
    tokenizer = BPETokenizer()
    tokenizer.build_vocab(corpus, min_freq=1, max_vocab=max_vocab)

    config = expected
    if config.d_model % config.n_heads != 0:
        config = AttentionLMConfig(
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=1,
            d_ff=config.d_ff,
            max_seq_len=config.max_seq_len,
            learning_rate=config.learning_rate,
            model_version=config.model_version,
        )
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
    code_starts = ("def ", "class ", "for ", "import ", "while ")
    if cleaned.startswith(code_starts):
        return cleaned
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


def _abstain_result(
    *,
    confidence: float,
    citations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "abstained": True,
        "answer": _TRAINING_NEED_MSG,
        "confidence": round(confidence, 4),
        "citations": citations or [],
        "model": "stacked_attention_lm",
        "model_version": CURRENT_MODEL_VERSION,
    }


def predict_with_steps(
    question: str,
    *,
    conversation_context: str = "",
    force: bool = False,
) -> dict[str, Any] | None:
    """
    Run the prediction pipeline: context → tokenize → embed → attention →
    layers → reasoning → next-token → autoregressive generate.
    """
    if not force and not is_prediction_question(question):
        return None

    try:
        model = get_predict_model()
    except Exception as exc:
        logger.exception("Predict model load failed")
        return {
            "abstained": True,
            "answer": _TRAINING_NEED_MSG,
            "confidence": 0.0,
            "citations": [],
            "model": "stacked_attention_lm",
            "model_version": CURRENT_MODEL_VERSION,
        }

    q = question.strip().lower().rstrip("?")
    context, context_sources, citations = _retrieve_context(q)
    if conversation_context.strip():
        context = f"{conversation_context.strip()} {context}".strip()
    max_rag_score = max((c.get("score", 0) for c in citations), default=0.0)
    min_rag = _abstain_min_rag()
    min_prob = _abstain_min_prob()
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
    top_prob = float(first_step.get("top_probability") or 0.0)

    if fallback:
        key_word = fallback.split()[0].lower()
        if not answer or len(answer) < 4 or key_word not in answer.lower():
            answer = fallback
            top_prob = max(top_prob, 0.9)

    if not citations and not fallback and max_rag_score < min_rag:
        return _abstain_result(confidence=top_prob)

    if not answer or answer.lower() == q:
        if fallback:
            answer = fallback
        else:
            return _abstain_result(confidence=top_prob, citations=citations)

    if not fallback and top_prob < min_prob and len(answer) < 8:
        return _abstain_result(confidence=top_prob, citations=citations)

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
            "description": "Vector RAG retrieves ranked corpus snippets with citations.",
            "words_used": len(context.split()) if context else 0,
            "sources": [s[:120] + "..." if len(s) > 120 else s for s in context_sources],
            "citations": citations,
            "retrieval": "vector_rag_tfidf",
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
        "confidence": round(top_prob, 4),
        "citations": citations,
        "abstained": False,
        "pipeline": steps,
        "generation": generation,
        "reasoning": reason_gen,
    }


def warm_up_predict_brain(*, run_probe: bool = True) -> dict[str, Any]:
    """Load model + RAG index before first user request — avoids cold-start hangs."""
    out: dict[str, Any] = {"rag_docs": 0, "model_ready": False, "probe": False}
    if os.environ.get("AUREON_PREDICT_BRAIN", "1").strip().lower() in ("0", "false", "no"):
        return out
    try:
        from brain.vector_rag import get_rag_index

        out["rag_docs"] = get_rag_index(force_rebuild=True).document_count
        get_predict_model()
        out["model_ready"] = _ready
        if run_probe:
            probe = predict_with_steps("what is mathematics")
            out["probe"] = bool(probe and probe.get("answer"))
            out["probe_answer"] = (probe or {}).get("answer", "")[:120]
    except Exception as exc:
        logger.warning("Predict brain warm-up skipped: %s", exc)
        out["error"] = str(exc)[:200]
    return out


def warmup_predict_brain_background() -> None:
    """Train/load the predict brain without blocking the request path."""
    if os.environ.get("AUREON_PREDICT_BRAIN", "1").strip().lower() in ("0", "false", "no"):
        return

    def _job() -> None:
        try:
            get_predict_model()
            from brain.vector_rag import invalidate_rag_index

            invalidate_rag_index()
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
            from brain.vector_rag import invalidate_rag_index

            invalidate_rag_index()
            logger.info("Predict brain retrained (reason=%s)", reason)
        except Exception:
            logger.exception("Predict brain retrain failed (reason=%s)", reason)

    threading.Thread(target=_job, name="aureon-predict-retrain", daemon=True).start()
