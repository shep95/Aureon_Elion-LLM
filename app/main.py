"""FastAPI application for Railway deployment."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from app.brain_routes import get_taxonomy
from app.pipeline_routes import run_pipeline_all, run_pipeline_step
from app.service import concepts, run_identify_demo, run_match_demo, run_synthetic_demo
from brain.cortex import bootstrap_brain, brain_status, run_domain_cycle, run_full_brain, run_subdomain_cycle


@asynccontextmanager
async def lifespan(_: FastAPI):
    from db.session import init_db
    from brain.cortex import bootstrap_brain

    init_db()
    try:
        bootstrap_brain()
    except Exception:
        pass
    try:
        from sklearn.datasets import fetch_olivetti_faces
        fetch_olivetti_faces()
    except Exception:
        pass
    yield


app = FastAPI(
    title="Aureon-LLM",
    description="Supervised machine learning demo — neural networks with backpropagation",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/concepts")
def get_concepts() -> dict[str, Any]:
    return concepts()


@app.post("/api/demo/synthetic")
def demo_synthetic(
    epochs: int = Query(default=200, ge=1, le=500),
    seed: int = Query(default=42),
) -> dict[str, Any]:
    try:
        return run_synthetic_demo(epochs=epochs, seed=seed)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/demo/match")
def demo_match(
    epochs: int = Query(default=200, ge=1, le=500),
    people: int = Query(default=40, ge=2, le=40),
    seed: int = Query(default=42),
) -> dict[str, Any]:
    try:
        return run_match_demo(epochs=epochs, people=people, seed=seed)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/demo/identify")
def demo_identify(
    epochs: int = Query(default=200, ge=1, le=500),
    people: int = Query(default=10, ge=2, le=10),
    seed: int = Query(default=42),
) -> dict[str, Any]:
    try:
        return run_identify_demo(epochs=epochs, people=people, seed=seed)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/brain/bootstrap")
def brain_bootstrap() -> dict:
    return bootstrap_brain()


@app.get("/api/brain/status")
def get_brain_status() -> dict:
    return brain_status()


@app.get("/api/brain/taxonomy")
def get_brain_taxonomy() -> dict:
    return get_taxonomy()


@app.post("/api/brain/run")
def brain_run(
    epochs: int = Query(default=150, ge=50, le=500),
    domain_limit: int | None = Query(default=3, ge=1, le=29),
    subdomain_limit: int | None = Query(default=1, ge=1),
) -> dict:
    try:
        return run_full_brain(
            epochs=epochs,
            domain_limit=domain_limit,
            subdomain_limit=subdomain_limit,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/brain/domain/{domain_slug}")
def brain_run_domain(
    domain_slug: str,
    epochs: int = Query(default=150, ge=50, le=500),
) -> dict:
    try:
        return run_domain_cycle(domain_slug, epochs=epochs)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/brain/domain/{domain_slug}/{subdomain_slug}")
def brain_run_subdomain(
    domain_slug: str,
    subdomain_slug: str,
    epochs: int = Query(default=150, ge=50, le=500),
) -> dict:
    try:
        return run_subdomain_cycle(domain_slug, subdomain_slug, epochs=epochs)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/pipeline/run")
def pipeline_run_all(
    epochs: int = Query(default=200, ge=50, le=500),
) -> dict:
    return run_pipeline_all(epochs=epochs)


@app.post("/api/pipeline/step/{step}")
def pipeline_run_step(
    step: int,
    epochs: int = Query(default=200, ge=50, le=500),
) -> dict:
    return run_pipeline_step(step=step, epochs=epochs)


@app.get("/api/pipeline/status")
def pipeline_status() -> dict:
    from pipeline.config import REGISTRY_DIR
    from pipeline.step3_training.registry import ModelRegistry

    registry = ModelRegistry()
    latest = REGISTRY_DIR / "latest_pipeline_run.json"
    payload: dict = {
        "production_model": registry.get_production(),
        "best_run": registry.get_best_run(),
    }
    if latest.exists():
        import json

        payload["latest_run"] = json.loads(latest.read_text(encoding="utf-8"))
    return payload


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Aureon-LLM — Supervised Machine Learning</title>
  <style>
    :root {
      color-scheme: light dark;
      --bg: #0f1419;
      --card: #1a2332;
      --text: #e7ecf3;
      --muted: #9aa7b8;
      --accent: #5b9fd4;
      --accent-hover: #7ab3e0;
      --border: #2a3647;
      --ok: #6bcf8e;
    }
    @media (prefers-color-scheme: light) {
      :root {
        --bg: #f4f7fb;
        --card: #ffffff;
        --text: #1a2332;
        --muted: #5a6778;
        --accent: #2563eb;
        --accent-hover: #1d4ed8;
        --border: #d8e0ea;
        --ok: #15803d;
      }
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
    }
    main { max-width: 920px; margin: 0 auto; padding: 2rem 1.25rem 4rem; }
    h1 { font-size: 1.75rem; margin: 0 0 0.5rem; }
    .subtitle { color: var(--muted); margin-bottom: 2rem; }
    .grid { display: grid; gap: 1rem; margin-bottom: 2rem; }
    @media (min-width: 700px) { .grid { grid-template-columns: repeat(3, 1fr); } }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.25rem;
    }
    .card h2 { font-size: 1rem; margin: 0 0 0.5rem; }
    .card p { font-size: 0.9rem; color: var(--muted); margin: 0 0 1rem; min-height: 3.5rem; }
    button {
      width: 100%;
      border: none;
      border-radius: 8px;
      padding: 0.75rem 1rem;
      background: var(--accent);
      color: white;
      font-weight: 600;
      cursor: pointer;
    }
    button:hover { background: var(--accent-hover); }
    button:disabled { opacity: 0.6; cursor: wait; }
    #status { color: var(--muted); min-height: 1.25rem; margin-bottom: 0.75rem; }
    #status.running { color: var(--accent); }
    #status.done { color: var(--ok); }
    pre {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1rem;
      overflow: auto;
      font-size: 0.82rem;
      min-height: 200px;
      white-space: pre-wrap;
      word-break: break-word;
    }
    ul.constraints { padding-left: 1.2rem; color: var(--muted); }
    code { background: var(--border); padding: 0.1rem 0.35rem; border-radius: 4px; }
  </style>
</head>
<body>
  <main>
    <h1>Aureon-LLM</h1>
    <p class="subtitle">
      Supervised machine learning with backpropagation — what &ldquo;AI&rdquo; actually is.
    </p>

    <div class="card" style="margin-bottom: 2rem;">
      <h2>How it works</h2>
      <p style="min-height: auto; color: var(--text);">
        Traditional code: you write the algorithm (<code>output = input + input</code>).
        Supervised ML: you provide labeled inputs and measurable outputs; the network
        learns weights via <strong>backpropagation</strong>.
      </p>
      <ul class="constraints">
        <li><strong>Clean data</strong> — labeled images/features, not opinions</li>
        <li><strong>Measurable goal</strong> — yes/no or person ID</li>
        <li><strong>Defined parameters</strong> — a bounded face database</li>
      </ul>
    </div>

    <div class="card" style="margin-bottom: 2rem;">
      <h2>Brain — micro-algorithms per domain</h2>
      <p style="min-height: auto; color: var(--text);">
        29 knowledge domains, 180+ subdomains, 6 brain regions each (collector, verifier,
        labeler, trainer, evaluator, reward). Stored in PostgreSQL on Railway.
      </p>
      <button onclick="runBrain()" style="max-width: 320px;">Run brain cycle</button>
    </div>

    <div class="card" style="margin-bottom: 2rem;">
      <h2>Automated training pipeline</h2>
      <p style="min-height: auto; color: var(--text);">
        Run all 5 steps in order: collect &rarr; label &rarr; train &rarr; evaluate &rarr; RLHF.
      </p>
      <button onclick="runPipeline()" style="max-width: 320px;">Run full pipeline</button>
    </div>

    <div class="grid">
      <div class="card">
        <h2>Synthetic features</h2>
        <p>Eye, nose, chin weights — the lecture&rsquo;s core example.</p>
        <button onclick="runDemo('synthetic')">Run demo</button>
      </div>
      <div class="card">
        <h2>Face matching</h2>
        <p>Binary yes/no: do two faces belong to the same person?</p>
        <button onclick="runDemo('match')">Run demo</button>
      </div>
      <div class="card">
        <h2>Person ID</h2>
        <p>Multi-class identification + edge-case fragility demo.</p>
        <button onclick="runDemo('identify')">Run demo</button>
      </div>
    </div>

    <div id="status"></div>
    <pre id="output">Click a demo to train a neural network and see results.</pre>
  </main>
  <script>
    const statusEl = document.getElementById('status');
    const outputEl = document.getElementById('output');
    const buttons = document.querySelectorAll('button');

    async function runBrain() {
      buttons.forEach(b => b.disabled = true);
      statusEl.textContent = 'Running brain micro-algorithms across knowledge domains…';
      statusEl.className = 'running';
      outputEl.textContent = '';
      try {
        const res = await fetch('/api/brain/run?epochs=150&domain_limit=2&subdomain_limit=1', { method: 'POST' });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || res.statusText);
        outputEl.textContent = JSON.stringify(data, null, 2);
        statusEl.textContent = 'Brain cycle complete.';
        statusEl.className = 'done';
      } catch (err) {
        outputEl.textContent = String(err);
        statusEl.textContent = 'Error.';
        statusEl.className = '';
      } finally {
        buttons.forEach(b => b.disabled = false);
      }
    }

    async function runPipeline() {
      buttons.forEach(b => b.disabled = true);
      statusEl.textContent = 'Running 5-step pipeline (collect → label → train → evaluate → RLHF)…';
      statusEl.className = 'running';
      outputEl.textContent = '';
      try {
        const res = await fetch('/api/pipeline/run?epochs=200', { method: 'POST' });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || res.statusText);
        outputEl.textContent = JSON.stringify(data, null, 2);
        statusEl.textContent = 'Pipeline ' + (data.status || 'done') + '.';
        statusEl.className = 'done';
      } catch (err) {
        outputEl.textContent = String(err);
        statusEl.textContent = 'Error.';
        statusEl.className = '';
      } finally {
        buttons.forEach(b => b.disabled = false);
      }
    }

    async function runDemo(name) {
      buttons.forEach(b => b.disabled = true);
      statusEl.textContent = 'Training neural network via backpropagation…';
      statusEl.className = 'running';
      outputEl.textContent = '';

      try {
        const res = await fetch(`/api/demo/${name}?epochs=200`, { method: 'POST' });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || res.statusText);
        outputEl.textContent = JSON.stringify(data, null, 2);
        statusEl.textContent = 'Done.';
        statusEl.className = 'done';
      } catch (err) {
        outputEl.textContent = String(err);
        statusEl.textContent = 'Error.';
        statusEl.className = '';
      } finally {
        buttons.forEach(b => b.disabled = false);
      }
    }
  </script>
</body>
</html>"""


def main() -> None:
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
