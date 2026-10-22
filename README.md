# SkinFL — Privacy-Preserving Multi-Class Skin Cancer Detection

**Multi-Agent Federated Learning in Oncology — final-year project.**

Hospitals collaboratively improve a skin-cancer classifier using Federated Learning,
without ever sharing a raw patient image.

> **Research prototype — not a medical device.**
> This system is intended for academic and research demonstration only. It has not been
> clinically validated and must not be used to make medical decisions.

---

## Status

| Week | Module | State |
|------|--------|-------|
| 1 | UI/UX + project foundation | ✅ complete |
| 2 | Backend + database | ✅ complete |
| 3 | Hospital agents | ✅ complete |
| 4 | Cancer detection | ✅ complete |
| 5 | Pre-trained model integration | ✅ complete |
| 6 | Fine-tuning + experiments | ✅ complete |
| 7 | Federated learning engine | ✅ complete |
| 8 | Federated training dashboard | ✅ complete |
| 9 | Evaluation (ROC, confusion matrix) | ✅ complete |
| 10 | Explainability + privacy centre | ✅ complete |
| 11 | Full-system integration (WebSocket) | ✅ complete |
| 12 | Testing + final demo | ✅ complete |

Every number visible in this application traces to code that actually executed. Missing
artefacts render an explicit blocked state and the command that produces them, never a
placeholder metric.

Week 10 adds Grad-CAM explainability and a privacy centre. Grad-CAM runs real attention
attribution over the MedSigLIP vision tower (`ml/explainability.py`) and is gated behind an
availability check: without torch and the pretrained weights the API and UI return an explicit
*"requires torch + model weights"* panel and never draw a fabricated heatmap. Explanations are
produced only for the file in the current upload request — stored detection results carry no
image reference, so nothing links a patient image to a result row. `GET /api/privacy/summary`
derives every counter, invariant and ledger entry live from the audit log.

Week 11 connects the subsystems through a read-only event bus. `/ws/training` streams
`audit`, `training`, `agent` and `system` notifications and sends a heartbeat every 20 s;
each frame only invalidates the REST query that owns its data, so the socket can drop
without producing a wrong number. `GET /api/system/integration` probes eleven subsystems
against live state, and the Audit and Settings pages render that evidence directly.

Week 12 adds `GET /api/demo/checklist` and the `/demo` page: twelve steps (API, database,
dataset, agents, embeddings, federated round, model registry, evaluation, detection,
explainability, privacy, audit), each with live evidence or the exact command that
unblocks it. The checklist reuses the Week 11 probes, so it cannot disagree with the pages
it links to.

---

## Judge demo

1. Start the backend and frontend with the commands in **Quick start**.
2. Open `/demo`. Required steps must read **ready**; optional explainability may be blocked
   if this host has no torch weights.
3. Walk the twelve cards in order, opening each linked page. Every card shows the database
   or artefact evidence that makes it ready.
4. Finish on `/audit` with the live ticker open, then start one federated round on
   `/federated-training` — training, FedAvg and broadcast events appear without a refresh.

If a required step is blocked, its card prints the exact command (for example
`scripts/build_splits.py` or `scripts/extract_embeddings.py`) and the page re-checks itself
every 15 seconds.

---

## Base model

Built on **`google/medsiglip-448`**, a medical vision-language model, rather than
training a network from scratch.

| Property | Value |
|---|---|
| Architecture | `SiglipModel` (dual-tower) |
| Vision tower | 27 layers · hidden 1152 · MLP 4304 · 16 heads · patch 14 |
| Text tower | 27 layers · hidden 1152 · vocab 32 000 · 64 positions |
| Input | 448 × 448 RGB, bicubic, mean = std = 0.5 |
| Embedding | 1152-d shared space |
| Parameters | ~878 M total (both towers) |

**Adaptation chain:**

```
MedSigLIP-448 vision tower
  → cancer-specific adaptation (4-class head, initialised from text-prompt embeddings)
  → fine-tuning (frozen / selective / full — compared empirically)
  → prediction (4-class softmax)
  → explainability (Grad-CAM)
```

See `docs/ADAPTATION_STRATEGY.md` for the full rationale and why a ViT backbone is used
instead of the originally-proposed RCNN + YOLOv9 detector.

---

## Layout

```
frontend/     React 19 + TypeScript + Vite + Tailwind + shadcn-style primitives
backend/      FastAPI + SQLAlchemy + SQLite + WebSocket
ml/           PyTorch model code, preprocessing, training, inference, FedAvg
docs/         Architecture and adaptation notes
```

---

## Quick start

**Backend** (Python 3.10+):

```bash
pip install -r backend/requirements.txt
python -m uvicorn app.main:app --app-dir backend --reload   # http://localhost:8000 (run from the repo root)
```

**Frontend** (Node 18+):

```bash
cd frontend
npm install
npm run dev                          # http://localhost:5173
```

Vite proxies `/api` and `/ws` to the backend, so no CORS setup is required in development.

---

## Privacy model

Raw patient images shared: **0** — enforced structurally, not by convention.

The aggregator only ever calls three methods on an agent, and all of them exchange
tensors, never pixel arrays:

- `fit(dataset) -> local metrics`
- `update() -> weight delta`
- `load(global_state) -> None`

There is no code path connecting a hospital's local dataset object to the central
server. The privacy dashboard counts these transmissions from the audit ledger.

---

## Target classes

| Class | HAM10000 `dx` | Malignant |
|---|---|---|
| Melanoma | `melanoma` | yes |
| Basal Cell Carcinoma | `bcc` | yes |
| Squamous Cell Carcinoma | `scc` | yes |
| Benign Lesion | `nv` | no |

Recall is the metric that matters most here: the cost of a false negative on a
malignant lesion is far higher than the cost of a false positive.

---

## Demo data

There is no simulation mode. Federated rounds train on real embedding caches; when
the caches are missing the API refuses the request with `503` and prints the
`scripts/build_splits.py` / `scripts/extract_embeddings.py` commands that produce
them, rather than inventing a round. Experiment rows stay `awaiting` until a real
run writes metrics, gated by `is_real_result`. Where the detection path has no
trained checkpoint it falls back to the zero-shot, prompt-initialised head and
labels that response `is_real_inference = false`; simulated numbers, when a caller
opts into them, are labelled `ILLUSTRATIVE DEMO DATA` at the point of display.

---

## License

Academic project. Not licensed for clinical or commercial use.