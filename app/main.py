"""FastAPI application for Railway deployment."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse

from app.auto_learn import get_auto_learn_scheduler, start_auto_learn, stop_auto_learn
from app.brain_routes import get_grade_progress, get_taxonomy
from brain.grades import curriculum_public, get_grade
from app.middleware import SecurityGatewayMiddleware, SecurityHeadersMiddleware
from app.organism import get_organism
from app.pipeline_routes import run_pipeline_all, run_pipeline_step
from app.security import (
    api_key_required,
    clamp_domain_limit,
    clamp_epochs,
    clamp_micro_subdomain_limit,
    clamp_subdomain_limit,
    exclusive_training_lock,
    require_mutating_access,
    safe_error_message,
    validate_slug,
)
from app.service import concepts, run_identify_demo, run_match_demo, run_synthetic_demo
from brain.cortex import (
    bootstrap_brain,
    brain_status,
    run_domain_cycle,
    run_full_brain,
    run_grade_cycle,
    run_graduation_ladder,
    run_micro_subdomain_cycle,
    run_subdomain_cycle,
)
from brain.domains.taxonomy import KNOWLEDGE_TAXONOMY

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    from db.session import init_db
    from app.startup import start_deferred_startup

    logging.basicConfig(level=logging.INFO)
    init_db()
    start_deferred_startup()
    yield
    try:
        stop_auto_learn()
    except Exception:
        logger.exception("Auto-learn shutdown failed")


app = FastAPI(
    title="Aureon-LLM",
    description="Supervised machine learning demo — neural networks with backpropagation",
    version="1.1.0",
    lifespan=lifespan,
)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(SecurityGatewayMiddleware)

Mutating = Annotated[None, Depends(require_mutating_access)]


@app.get("/health")
def health() -> dict[str, str | bool]:
    from app.startup import get_startup_state

    state = get_startup_state()
    return {
        "status": "ok",
        "ready": state.ready,
        "bootstrap_done": state.bootstrap_done,
        "auto_learn": state.auto_learn_started,
    }


@app.get("/health/ready")
def health_ready() -> dict:
    from app.startup import get_startup_state

    state = get_startup_state()
    if not state.ready:
        raise HTTPException(status_code=503, detail="Startup in progress")
    return {"status": "ready", "details": state.details}


@app.get("/organism/vitals")
def organism_vitals() -> dict:
    """Security organism health — nomad_cyber_algorithm pattern."""
    organism = get_organism()
    organism.pulse()
    return organism.get_vitals_report()


@app.get("/api/concepts")
def get_concepts() -> dict[str, Any]:
    return concepts()


@app.post("/api/demo/synthetic")
def demo_synthetic(
    _auth: Mutating,
    epochs: int = Query(default=200, ge=1, le=500),
    seed: int = Query(default=42, ge=0, le=2_147_483_647),
) -> dict[str, Any]:
    try:
        with exclusive_training_lock():
            return run_synthetic_demo(epochs=epochs, seed=seed)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=safe_error_message(exc)) from exc


@app.post("/api/demo/match")
def demo_match(
    _auth: Mutating,
    epochs: int = Query(default=200, ge=1, le=500),
    people: int = Query(default=40, ge=2, le=40),
    seed: int = Query(default=42, ge=0, le=2_147_483_647),
) -> dict[str, Any]:
    try:
        with exclusive_training_lock():
            return run_match_demo(epochs=epochs, people=people, seed=seed)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=safe_error_message(exc)) from exc


@app.post("/api/demo/identify")
def demo_identify(
    _auth: Mutating,
    epochs: int = Query(default=200, ge=1, le=500),
    people: int = Query(default=10, ge=2, le=10),
    seed: int = Query(default=42, ge=0, le=2_147_483_647),
) -> dict[str, Any]:
    try:
        with exclusive_training_lock():
            return run_identify_demo(epochs=epochs, people=people, seed=seed)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=safe_error_message(exc)) from exc


@app.post("/api/brain/bootstrap")
def brain_bootstrap(_auth: Mutating) -> dict:
    return bootstrap_brain()


@app.get("/api/brain/status")
def get_brain_status() -> dict:
    return brain_status()


@app.get("/api/brain/taxonomy")
def get_brain_taxonomy() -> dict:
    return get_taxonomy()


@app.post("/api/brain/run")
def brain_run(
    _auth: Mutating,
    epochs: int = Query(default=150, ge=50, le=500),
    domain_limit: int | None = Query(default=3, ge=1, le=29),
    subdomain_limit: int | None = Query(default=1, ge=1, le=20),
    micro_subdomain_limit: int | None = Query(default=1, ge=1, le=10),
) -> dict:
    try:
        with exclusive_training_lock():
            return run_full_brain(
                epochs=clamp_epochs(epochs),
                domain_limit=clamp_domain_limit(domain_limit),
                subdomain_limit=clamp_subdomain_limit(subdomain_limit),
                micro_subdomain_limit=clamp_micro_subdomain_limit(micro_subdomain_limit),
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=safe_error_message(exc)) from exc


@app.post("/api/brain/domain/{domain_slug}")
def brain_run_domain(
    domain_slug: str,
    _auth: Mutating,
    epochs: int = Query(default=150, ge=50, le=500),
    subdomain_limit: int | None = Query(default=5, ge=1, le=20),
    micro_subdomain_limit: int | None = Query(default=1, ge=1, le=10),
) -> dict:
    domain_slug = validate_slug(domain_slug, label="domain")
    if domain_slug not in KNOWLEDGE_TAXONOMY:
        raise HTTPException(status_code=404, detail="Unknown domain")
    try:
        with exclusive_training_lock():
            return run_domain_cycle(
                domain_slug,
                epochs=epochs,
                subdomain_limit=clamp_subdomain_limit(subdomain_limit) or 5,
                micro_subdomain_limit=clamp_micro_subdomain_limit(micro_subdomain_limit),
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=safe_error_message(exc)) from exc


@app.post("/api/brain/domain/{domain_slug}/{subdomain_slug}")
def brain_run_subdomain(
    domain_slug: str,
    subdomain_slug: str,
    _auth: Mutating,
    epochs: int = Query(default=150, ge=50, le=500),
    micro_subdomain_limit: int | None = Query(default=3, ge=1, le=10),
) -> dict:
    domain_slug = validate_slug(domain_slug, label="domain")
    subdomain_slug = validate_slug(subdomain_slug, label="subdomain")
    if domain_slug not in KNOWLEDGE_TAXONOMY:
        raise HTTPException(status_code=404, detail="Unknown domain")
    if subdomain_slug not in KNOWLEDGE_TAXONOMY[domain_slug]:
        raise HTTPException(status_code=404, detail="Unknown subdomain")
    try:
        with exclusive_training_lock():
            return run_subdomain_cycle(
                domain_slug,
                subdomain_slug,
                epochs=epochs,
                micro_subdomain_limit=clamp_micro_subdomain_limit(micro_subdomain_limit),
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=safe_error_message(exc)) from exc


@app.get("/api/brain/auto-learn")
def get_auto_learn_status() -> dict:
    """Automated learning scheduler status (Railway background cycles)."""
    return get_auto_learn_scheduler().status()


@app.get("/api/brain/grades")
def get_grade_curriculum() -> dict:
    return {"curriculum": curriculum_public(), "total_levels": len(curriculum_public())}


@app.get("/api/brain/grades/{domain_slug}/{subdomain_slug}/{micro_subdomain_slug}")
def get_micro_grade_progress(
    domain_slug: str,
    subdomain_slug: str,
    micro_subdomain_slug: str,
) -> dict:
    domain_slug = validate_slug(domain_slug, label="domain")
    subdomain_slug = validate_slug(subdomain_slug, label="subdomain")
    micro_subdomain_slug = validate_slug(micro_subdomain_slug, label="micro_subdomain")
    return get_grade_progress(domain_slug, subdomain_slug, micro_subdomain_slug)


@app.post("/api/brain/domain/{domain_slug}/{subdomain_slug}/{micro_subdomain_slug}/grade/{grade_slug}")
def brain_run_grade(
    domain_slug: str,
    subdomain_slug: str,
    micro_subdomain_slug: str,
    grade_slug: str,
    _auth: Mutating,
    epochs: int = Query(default=150, ge=50, le=500),
) -> dict:
    domain_slug = validate_slug(domain_slug, label="domain")
    subdomain_slug = validate_slug(subdomain_slug, label="subdomain")
    micro_subdomain_slug = validate_slug(micro_subdomain_slug, label="micro_subdomain")
    grade_slug = validate_slug(grade_slug, label="grade")
    if not get_grade(grade_slug):
        raise HTTPException(status_code=404, detail="Unknown grade level")
    if domain_slug not in KNOWLEDGE_TAXONOMY:
        raise HTTPException(status_code=404, detail="Unknown domain")
    if subdomain_slug not in KNOWLEDGE_TAXONOMY[domain_slug]:
        raise HTTPException(status_code=404, detail="Unknown subdomain")
    if micro_subdomain_slug not in KNOWLEDGE_TAXONOMY[domain_slug][subdomain_slug]:
        raise HTTPException(status_code=404, detail="Unknown micro_subdomain")
    try:
        with exclusive_training_lock():
            return run_grade_cycle(
                domain_slug,
                subdomain_slug,
                micro_subdomain_slug,
                grade_slug=grade_slug,
                epochs=epochs,
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=safe_error_message(exc)) from exc


@app.post("/api/brain/domain/{domain_slug}/{subdomain_slug}/{micro_subdomain_slug}/graduate")
def brain_run_graduation_ladder(
    domain_slug: str,
    subdomain_slug: str,
    micro_subdomain_slug: str,
    _auth: Mutating,
    epochs: int = Query(default=150, ge=50, le=500),
    max_grades: int | None = Query(default=3, ge=1, le=7),
) -> dict:
    domain_slug = validate_slug(domain_slug, label="domain")
    subdomain_slug = validate_slug(subdomain_slug, label="subdomain")
    micro_subdomain_slug = validate_slug(micro_subdomain_slug, label="micro_subdomain")
    if domain_slug not in KNOWLEDGE_TAXONOMY:
        raise HTTPException(status_code=404, detail="Unknown domain")
    if subdomain_slug not in KNOWLEDGE_TAXONOMY[domain_slug]:
        raise HTTPException(status_code=404, detail="Unknown subdomain")
    if micro_subdomain_slug not in KNOWLEDGE_TAXONOMY[domain_slug][subdomain_slug]:
        raise HTTPException(status_code=404, detail="Unknown micro_subdomain")
    try:
        with exclusive_training_lock():
            return run_graduation_ladder(
                domain_slug,
                subdomain_slug,
                micro_subdomain_slug,
                epochs=epochs,
                max_grades=max_grades,
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=safe_error_message(exc)) from exc


@app.post("/api/brain/domain/{domain_slug}/{subdomain_slug}/{micro_subdomain_slug}")
def brain_run_micro_subdomain(
    domain_slug: str,
    subdomain_slug: str,
    micro_subdomain_slug: str,
    _auth: Mutating,
    epochs: int = Query(default=150, ge=50, le=500),
) -> dict:
    domain_slug = validate_slug(domain_slug, label="domain")
    subdomain_slug = validate_slug(subdomain_slug, label="subdomain")
    micro_subdomain_slug = validate_slug(micro_subdomain_slug, label="micro_subdomain")
    if domain_slug not in KNOWLEDGE_TAXONOMY:
        raise HTTPException(status_code=404, detail="Unknown domain")
    if subdomain_slug not in KNOWLEDGE_TAXONOMY[domain_slug]:
        raise HTTPException(status_code=404, detail="Unknown subdomain")
    if micro_subdomain_slug not in KNOWLEDGE_TAXONOMY[domain_slug][subdomain_slug]:
        raise HTTPException(status_code=404, detail="Unknown micro_subdomain")
    try:
        with exclusive_training_lock():
            return run_micro_subdomain_cycle(
                domain_slug,
                subdomain_slug,
                micro_subdomain_slug,
                epochs=epochs,
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=safe_error_message(exc)) from exc


@app.post("/api/pipeline/run")
def pipeline_run_all_endpoint(
    _auth: Mutating,
    epochs: int = Query(default=200, ge=50, le=500),
) -> dict:
    try:
        with exclusive_training_lock():
            return run_pipeline_all(epochs=epochs)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=safe_error_message(exc)) from exc


@app.post("/api/pipeline/step/{step}")
def pipeline_run_step_endpoint(
    step: int,
    _auth: Mutating,
    epochs: int = Query(default=200, ge=50, le=500),
) -> dict:
    if step not in (1, 2, 3, 4, 5):
        raise HTTPException(status_code=400, detail="step must be 1-5")
    try:
        with exclusive_training_lock():
            return run_pipeline_step(step=step, epochs=epochs)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=safe_error_message(exc)) from exc


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

        from app.security import load_json_file_bounded

        payload["latest_run"] = load_json_file_bounded(latest)
    return payload


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


INDEX_HTML = """<!DOCTYPE html>
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
        29 knowledge domains, 154 subdomains, 462 micro-subdomains, 6 brain regions each.
      </p>
      <button id="runBrainBtn" style="max-width: 320px;">Run brain cycle</button>
    </div>

    <div class="card" style="margin-bottom: 2rem;">
      <h2>Automated training pipeline</h2>
      <p style="min-height: auto; color: var(--text);">
        Run all 5 steps in order: collect &rarr; label &rarr; train &rarr; evaluate &rarr; RLHF.
      </p>
      <button id="runPipelineBtn" style="max-width: 320px;">Run full pipeline</button>
    </div>

    <div class="grid">
      <div class="card">
        <h2>Synthetic features</h2>
        <p>Eye, nose, chin weights — the lecture&rsquo;s core example.</p>
        <button data-demo="synthetic">Run demo</button>
      </div>
      <div class="card">
        <h2>Face matching</h2>
        <p>Binary yes/no: do two faces belong to the same person?</p>
        <button data-demo="match">Run demo</button>
      </div>
      <div class="card">
        <h2>Person ID</h2>
        <p>Multi-class identification + edge-case fragility demo.</p>
        <button data-demo="identify">Run demo</button>
      </div>
    </div>

    <div id="status"></div>
    <pre id="output">Click a demo to train a neural network and see results.</pre>
  </main>
  <script>
    const statusEl = document.getElementById('status');
    const outputEl = document.getElementById('output');
    const buttons = document.querySelectorAll('button');

    function apiHeaders() {
      const headers = { 'Content-Type': 'application/json' };
      const key = window.AUREON_API_KEY;
      if (key) {
        headers['X-API-Key'] = key;
        headers['X-Timestamp'] = String(Date.now());
        headers['X-Nonce'] = crypto.randomUUID();
        headers['X-Correlation-ID'] = crypto.randomUUID();
      }
      return headers;
    }

    async function postJson(url) {
      const res = await fetch(url, { method: 'POST', headers: apiHeaders() });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || res.statusText);
      return data;
    }

    async function runBrain() {
      buttons.forEach(b => b.disabled = true);
      statusEl.textContent = 'Running brain micro-algorithms across knowledge domains…';
      statusEl.className = 'running';
      outputEl.textContent = '';
      try {
        outputEl.textContent = JSON.stringify(
          await postJson('/api/brain/run?epochs=150&domain_limit=2&subdomain_limit=1'),
          null, 2
        );
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
      statusEl.textContent = 'Running 5-step pipeline…';
      statusEl.className = 'running';
      outputEl.textContent = '';
      try {
        const data = await postJson('/api/pipeline/run?epochs=200');
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
        outputEl.textContent = JSON.stringify(await postJson('/api/demo/' + name + '?epochs=200'), null, 2);
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

    document.getElementById('runBrainBtn').addEventListener('click', runBrain);
    document.getElementById('runPipelineBtn').addEventListener('click', runPipeline);
    document.querySelectorAll('[data-demo]').forEach(btn => {
      btn.addEventListener('click', () => runDemo(btn.dataset.demo));
    });
  </script>
</body>
</html>"""


def main() -> None:
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
