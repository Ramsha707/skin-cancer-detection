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
| 6 | Fine-tuning + experiments | ⬜ planned |
| 7 | Federated learning engine | ⬜ planned |
| 8 | Federated training dashboard | ⬜ planned |
| 9 | Evaluation (ROC, confusion matrix) | ⬜ planned |
| 10 | Explainability + privacy centre | ⬜ planned |
| 11 | Full-system integration (WebSocket) | ⬜ planned |
| 12 | Testing + final demo | ⬜ planned |

Routes for weeks 6–12 render an explicit *"not built yet"* panel rather than placeholder
metrics. Every number visible in this application traces to code that actually executed.

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
frontend/     React 18 + TypeScript + Vite + Tailwind + shadcn-style primitives
backend/      FastAPI + SQLAlchemy + SQLite + WebSocket
ml/           PyTorch model code, preprocessing, training, inference, FedAvg
docs/         Architecture and adaptation notes
```

---

## Quick start

**Backend** (Python 3.10+):

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload        # http://localhost:8000
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

Until a real HAM10000 partition and trained checkpoints are present, the federated
engine runs in **simulation mode**. Simulated numbers are labelled
`ILLUSTRATIVE DEMO DATA` at the point of display, and every response carries an
`is_real_inference` / `is_real_result` flag so the distinction survives into the API.

---

## License

Academic project. Not licensed for clinical or commercial use.