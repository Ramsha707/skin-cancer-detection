"""Week 9: evaluation of the trained global model on the held-out validation split.

The validation split is never trained on by any agent (`data/splits.json` keeps it
separate from the four agent partitions), so a report over it is an honest estimate
of what the aggregated head actually learned.

Nothing here fabricates a number. When there is no checkpoint to evaluate, or the
validation embeddings have not been extracted, the summary comes back with
`ready=False`, an empty report and the exact command that produces the artefacts.
The UI renders that as an instruction panel, never as a chart.
"""

from __future__ import annotations

import logging

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import ModelVersion, TrainingRound
from app.services import metrics as metrics_service
from app.services import registry
from ml.models import CANCER_CLASSES, MALIGNANT_CLASSES

log = logging.getLogger(__name__)

MALIGNANT_INDICES = [CANCER_CLASSES.index(name) for name in MALIGNANT_CLASSES]

EXTRACT_HINT = (
    "Run `python scripts/extract_embeddings.py` to build the embedding caches "
    "(including `val.npy`), then reload."
)
TRAIN_HINT = (
    "No global checkpoint exists yet. Train a model first: run at least one "
    "federated round from the Federated Training page, or run a frozen-head "
    "experiment. Evaluation reports the aggregated head, not the zero-shot prior."
)


def _readiness_instructions(val_real: bool, dim_match: bool, trained: bool) -> str:
    missing: list[str] = []
    if not val_real:
        missing.append(EXTRACT_HINT)
    elif not dim_match:
        missing.append(
            "The validation embeddings do not match the head's input width. "
            "Re-extract them with `python scripts/extract_embeddings.py`."
        )
    if not trained:
        missing.append(TRAIN_HINT)
    return " ".join(missing)


def _learning_curve(db: Session) -> list[dict]:
    rows = db.scalars(select(TrainingRound).order_by(TrainingRound.round_number)).all()
    return [
        {
            "round_number": row.round_number,
            "evaluated": row.global_accuracy is not None,
            "accuracy": row.global_accuracy,
            "f1": row.global_f1,
            "auc": row.global_auc,
            "loss": row.global_loss,
        }
        for row in rows
    ]


def _model_history(db: Session) -> list[dict]:
    rows = db.scalars(
        select(ModelVersion).order_by(ModelVersion.created_at, ModelVersion.id)
    ).all()
    return [
        {
            "version": row.version,
            "strategy": row.strategy,
            "training_round": row.training_round,
            "accuracy": row.accuracy,
            "precision": row.precision,
            "recall": row.recall,
            "specificity": row.specificity,
            "f1": row.f1,
            "auc": row.auc,
            "loss": row.loss,
            "status": row.status,
            "is_recommended": row.is_recommended,
            "created_at": row.created_at,
        }
        for row in rows
    ]


def _roc(val_labels: np.ndarray, probs: np.ndarray) -> dict:
    """One-vs-rest ROC per class plus the summed malignant-vs-benign curve."""
    per_class: dict[str, list[dict]] = {}
    for index, name in enumerate(CANCER_CLASSES):
        points = metrics_service.roc_curve_points(
            (val_labels == index).astype(int), probs[:, index]
        )
        if points is not None:
            per_class[name] = points

    malignant_points = metrics_service.roc_curve_points(
        np.isin(val_labels, MALIGNANT_INDICES).astype(int),
        probs[:, MALIGNANT_INDICES].sum(axis=1),
    )
    return {"per_class": per_class, "malignant": malignant_points}


def build_summary(db: Session) -> dict:
    """Assemble the evaluation payload. Never raises for a missing artefact."""
    val = registry.load_split("val")
    checkpoint = registry.latest_checkpoint()
    trained = checkpoint is not None

    summary: dict = {
        "ready": False,
        "model_version": checkpoint.stem if trained else "untrained",
        "is_real": False,
        "instructions": None,
        "n_eval": val.size if val.real else 0,
        "n_classes": len(CANCER_CLASSES),
        "classes": list(CANCER_CLASSES),
        "malignant_classes": list(MALIGNANT_CLASSES),
        "metrics": None,
        "per_class": {},
        "confusion_matrix": [],
        "roc": {"per_class": {}, "malignant": None},
        "learning_curve": _learning_curve(db),
        "model_history": _model_history(db),
    }

    dim_match = val.embeddings.shape[1] == registry.EMBEDDING_DIM
    if not trained or not val.real or not dim_match:
        summary["instructions"] = _readiness_instructions(val.real, dim_match, trained)
        return summary

    head, version, _ = registry.load_trained_head()
    probs = head.probabilities(val.embeddings)
    predictions = probs.argmax(axis=1)
    malignant_scores = probs[:, MALIGNANT_INDICES].sum(axis=1)

    report = metrics_service.evaluate(
        val.labels,
        predictions,
        n_classes=len(CANCER_CLASSES),
        class_names=CANCER_CLASSES,
        malignant_indices=MALIGNANT_INDICES,
        y_score=probs,
        y_score_malignant=malignant_scores,
    )

    summary.update(
        ready=True,
        model_version=version,
        is_real=True,
        metrics={
            "accuracy": report.accuracy,
            "macro_f1": report.macro_f1,
            "macro_recall": report.macro_recall,
            "balanced_accuracy": report.balanced_accuracy,
            "precision": report.precision,
            "recall": report.recall,
            "specificity": report.specificity,
            "f1": report.f1,
            "auc": report.auc,
            "malignant_auc": report.malignant_auc,
            "false_negatives": report.false_negatives,
            "false_positives": report.false_positives,
            "worst_class_recall": report.worst_class_recall,
            "support": report.support,
        },
        per_class=report.per_class,
        confusion_matrix=report.confusion_matrix,
        roc=_roc(val.labels, probs),
    )
    return summary


__all__ = ["MALIGNANT_INDICES", "build_summary"]
