"""Quick chat latency benchmark — run: python scripts/bench_chat_latency.py"""

from __future__ import annotations

import os
import time

from app.chat_service import chat

CASES = [
    ("who are you", "identity"),
    ("what is 2+2", "math"),
    ("what is the capital of france", "predict"),
    ("what is the god to you", "philosophy"),
    ("/status", "command"),
]


def main() -> None:
    print("Predict config:")
    for key in (
        "AUREON_PREDICT_MAX_SEQ",
        "AUREON_PREDICT_MAX_VOCAB",
        "AUREON_PREDICT_EPOCHS",
        "AUREON_PREDICT_DOC_LIMIT",
        "AUREON_CHAT_REWARD",
    ):
        print(f"  {key}={os.environ.get(key, '(default)')}")
    print()

    for msg, label in CASES:
        t0 = time.perf_counter()
        r = chat(msg)
        ms = (time.perf_counter() - t0) * 1000
        reply = str(r.get("reply", ""))[:70].replace("\n", " ")
        kind = r.get("kind")
        print(f"{label:12} {ms:8.0f} ms  kind={kind:8}  {reply}")


if __name__ == "__main__":
    main()
