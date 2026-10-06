"""Week 6: transfer-learning experiment runner.

Three strategies are registered at seed time:

* **A - frozen**: train only the 4-class head over cached vision-tower
  embeddings. Executable anywhere the embedding caches exist, which is this
  host.
* **B - selective**: unfreeze the top encoder blocks. Needs a backward pass
  through the tower, so it needs a CUDA device (OpenVINO is inference-only).
* **C - full**: unfreeze the whole vision tower. Gated on computational
  feasibility by the roadmap; without a GPU this is tens of CPU-days for a
  single epoch.

`FEASIBILITY` is the single source of truth for that gate and is surfaced on
the Experiments page, so the UI never offers a run the backend will refuse.
Nothing here fabricates numbers: a strategy that cannot run stays
`awaiting`, and only `run_experiment` ever writes metrics, always with
`is_real_result=True`.
"""

from __future__ import annotations

import logging
import time

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Experiment
from app.services import metrics as metrics_service
from app.services import registry
from ml.models import CANCER_CLASSES, MALIGNANT_CLASSES, CancerHead, softmax_cross_entropy

log = logging.getLogger(__name__)

MALIGNANT_INDICES = [CANCER_CLASSES.index(c) for c in MALIGNANT_CLASSES]


class ExperimentNotRunnable(Exception):
    """Refused for an honest reason: infeasible strategy or missing artefacts.

    The message is operator-facing and is returned verbatim as the HTTP 409
    detail, so the refusal explains itself instead of being a bare failure.
"""


def feasibility(strategy: str | None) -> tuple[bool, str]:
    """`(runnable, reason)` for a strategy. Unknown/None strategies never run."""
    if strategy == "frozen":
        return True, (
            "Runs from cached embeddings: only the 4,612-parameter head is "
            "trained, so it needs no tower backprop and is cheap on CPU."
        )
    if strategy == "selective":
        return False, (
            "Requires a backward pass through the top 6 of 27 encoder blocks. "
            "No CUDA device is available on this host and OpenVINO is "
            "inference-only, so the arm is registered but not executed here."
        )
    if strategy == "full":
        return False, (
            "Full fine-tuning of the 428M-parameter vision tower is gated on "
            "computational feasibility. Without a CUDA GPU a single epoch is "
            "tens of CPU-days, so it is registered but not executed here."
        )
    return False, "Experiment has no recognised training strategy."


def _real_splits() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Concatenated real train caches. Raises when none exist."""
    parts = []
    used: list[str] = []
    for agent in registry.AGENT_SLUGS:
        split = registry.load_split("train", agent)
        if split.real:
            parts.append((split.embeddings, split.labels))
            used.append(agent)
    if not parts:
        raise ExperimentNotRunnable(
            "No train embedding caches found. Run "
            ".\\.venv-ov\\Scripts\\python.exe scripts/extract_embeddings.py --split train "
            "before running an experiment."
        )
    return np.vstack([p[0] for p in parts]), np.concatenate([p[1] for p in parts]), used


def _normalise(x: np.ndarray) -> np.ndarray:
    """Unit-length rows, matching what `CancerHead.logits` does at inference."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (x / norms).astype(np.float32)


def train_head(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    holdout_frac: float = 0.1,
    epochs: int = 200,
    lr: float = 5e-2,
    l2: float = 1e-4,
    patience: int = 20,
    seed: int = 0,
) -> tuple[CancerHead, int, float]:
    """Softmax-regression head with early stopping on a carved-out holdout.

    The holdout comes from the *training* split, never from the val split the
    experiment is scored on, so the reported metrics are not tuned against.
    Class weights inverse to frequency stop the 80%-benign majority from
    collapsing the head to 'always benign'. Returns `(head, epochs_run,
    best_holdout_macro_f1)`.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x_train))
    x_train, y_train = x_train[order], y_train[order]

    n_hold = max(1, int(len(x_train) * holdout_frac))
    x_fit, y_fit = x_train[n_hold:], y_train[n_hold:]
    x_hold, y_hold = x_train[:n_hold], y_train[:n_hold]

    x_fit = _normalise(x_fit)
    x_hold = _normalise(x_hold)

    counts = np.bincount(y_fit, minlength=len(CANCER_CLASSES)).astype(float)
    class_w = metrics_service.weighted_class_weights(counts)
    sample_w = class_w[y_fit][:, None].astype(np.float32)

    n_classes = len(CANCER_CLASSES)
    weights = np.zeros((x_fit.shape[1], n_classes), dtype=np.float32)
    bias = np.zeros(n_classes, dtype=np.float32)
    m_w = np.zeros_like(weights)
    v_w = np.zeros_like(weights)
    m_b = np.zeros_like(bias)
    v_b = np.zeros_like(bias)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    def holdout_macro_f1() -> float:
        head = CancerHead(weights=weights, bias=bias)
        preds = head.predict(x_hold)
        report = metrics_service.evaluate(
            y_hold,
            preds,
            n_classes=n_classes,
            class_names=CANCER_CLASSES,
        )
        return report.macro_f1

    best_f1 = -1.0
    best_state = (weights.copy(), bias.copy())
    since_best = 0
    epochs_run = 0

    for epoch in range(1, epochs + 1):
        # Weighted mean cross-entropy; the gradient is the standard residual
        # scaled per-sample so the majority class cannot dominate it.
        _, grad_logits = softmax_cross_entropy(x_fit @ weights + bias, y_fit)
        grad_logits = grad_logits * sample_w
        d_w = x_fit.T @ grad_logits + l2 * weights
        d_b = grad_logits.sum(axis=0)

        m_w = beta1 * m_w + (1 - beta1) * d_w
        v_w = beta2 * v_w + (1 - beta2) * d_w**2
        m_b = beta1 * m_b + (1 - beta1) * d_b
        v_b = beta2 * v_b + (1 - beta2) * d_b**2
        mh_w = m_w / (1 - beta1**epoch)
        vh_w = v_w / (1 - beta2**epoch)
        mh_b = m_b / (1 - beta1**epoch)
        vh_b = v_b / (1 - beta2**epoch)
        weights -= lr * mh_w / (np.sqrt(vh_w) + eps)
        bias -= lr * mh_b / (np.sqrt(vh_b) + eps)

        epochs_run = epoch
        f1 = holdout_macro_f1()
        if f1 > best_f1:
            best_f1 = f1
            best_state = (weights.copy(), bias.copy())
            since_best = 0
        else:
            since_best += 1
            if since_best >= patience:
                break

    weights, bias = best_state
    return CancerHead(weights=weights, bias=bias), epochs_run, float(best_f1)


def run_experiment(db: Session, name: str) -> Experiment:
    """Execute the `frozen` strategy end to end and record real metrics.

    Raises `ExperimentNotRunnable` (HTTP 409 upstream) when the strategy is
    infeasible on this host or the embedding caches are missing. Metrics are
    written only from `metrics_service.evaluate` over a held-out val split.
    """
    row = db.scalar(select(Experiment).where(Experiment.name == name))
    if row is None:
        raise KeyError(name)

    runnable, reason = feasibility(row.strategy)
    if not runnable:
        raise ExperimentNotRunnable(reason)

    x_train, y_train, agents = _real_splits()
    val = registry.load_split("val")
    if not val.real:
        raise ExperimentNotRunnable(
            "No val embedding cache found. Run "
            ".\\.venv-ov\\Scripts\\python.exe scripts/extract_embeddings.py --split val "
            "before running an experiment."
        )

    started = time.perf_counter()
    # Train with the row's declared hyperparameters, so what the Experiments
    # table displays is what was actually optimised.
    head, epochs_run, holdout_f1 = train_head(
        x_train,
        y_train,
        lr=float(row.learning_rate or 5e-2),
        epochs=int(row.epochs or 200),
    )
    report = metrics_service.evaluate(
        val.labels,
        head.predict(_normalise(val.embeddings)),
        n_classes=len(CANCER_CLASSES),
        class_names=CANCER_CLASSES,
        malignant_indices=MALIGNANT_INDICES,
        y_score=head.probabilities(_normalise(val.embeddings)),
        y_score_malignant=head.probabilities(_normalise(val.embeddings))[
            :, MALIGNANT_INDICES
        ].sum(axis=1),
    )
    elapsed = time.perf_counter() - started

    provenance = (
        f"real run: head trained on {len(x_train)} cached embeddings from "
        f"{', '.join(agents)} at lr={row.learning_rate:g}; holdout macro-F1 "
        f"{holdout_f1:.3f} after {epochs_run} epochs; scored on the {val.size}-image "
        f"val split in {elapsed:.1f}s."
    )
    row.accuracy = report.accuracy
    row.precision = report.precision
    row.recall = report.recall
    row.specificity = report.specificity
    row.f1 = report.f1
    row.auc = report.auc
    row.epochs = epochs_run
    # The head just trained: 1152x4+4 on real caches, so this equals
    # registry.head_param_count() for a real run.
    row.trainable_params = int(head.weights.size + head.bias.size)
    row.total_params = int(registry.load_model_card()["total_params"])
    row.train_seconds = round(elapsed, 3)
    row.status = "completed"
    row.is_real_result = True
    # Keep the seeded rationale and append the receipt under it; a re-run
    # replaces the previous receipt rather than stacking them.
    rationale = row.notes or ""
    if "real run:" in rationale:
        rationale = rationale.split("real run:", 1)[0].rstrip()
    row.notes = f"{rationale}\n\n{provenance}".strip() if rationale else provenance
    log.info("experiment %r completed: accuracy=%.4f auc=%s", name, report.accuracy, report.auc)
    return row


def experiment_payload(row: Experiment) -> dict:
    """Row fields plus the feasibility gate the Experiments page renders."""
    runnable, reason = feasibility(row.strategy)
    return {
        "id": row.id,
        "name": row.name,
        "model": row.model,
        "strategy": row.strategy,
        "learning_rate": row.learning_rate,
        "epochs": row.epochs,
        "accuracy": row.accuracy,
        "precision": row.precision,
        "recall": row.recall,
        "specificity": row.specificity,
        "f1": row.f1,
        "auc": row.auc,
        "trainable_layers": row.trainable_layers,
        "trainable_params": row.trainable_params,
        "total_params": row.total_params,
        "train_seconds": row.train_seconds,
        "status": row.status,
        "is_real_result": row.is_real_result,
        "notes": row.notes,
        "created_at": row.created_at,
        "feasible": runnable,
        "feasibility_note": reason,
    }


__all__ = [
    "ExperimentNotRunnable",
    "experiment_payload",
    "feasibility",
    "run_experiment",
    "train_head",
]
