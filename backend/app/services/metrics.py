"""Binary metrics for imbalanced multi-class cancer detection.

Accuracy alone is actively misleading on HAM10000, where benign nevi are about
two thirds of the data. Sensitivity (recall) and the false-negative count are the
numbers that matter, so they are computed here for every evaluation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

import numpy as np


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


@dataclass
class BinaryMetrics:
    """Per-class precision / recall / specificity / F1 plus support."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def precision(self) -> float:
        return _safe_div(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float:
        """Sensitivity. Fraction of true positives actually detected."""
        return _safe_div(self.tp, self.tp + self.fn)

    @property
    def specificity(self) -> float:
        return _safe_div(self.tn, self.tn + self.fp)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return _safe_div(2 * p * r, p + r)

    @property
    def support(self) -> int:
        return self.tp + self.fn

    def as_dict(self) -> dict[str, float | int]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "precision": self.precision,
            "recall": self.recall,
            "specificity": self.specificity,
            "f1": self.f1,
            "support": self.support,
        }


@dataclass
class EvaluationReport:
    """Full metric bundle for a test/validation split.

    Primary metrics are ``macro_f1`` and ``per_class_recall``. Plain accuracy is
    reported only as a secondary diagnostic because it is dominated by the benign
    majority: the real zero-shot MedSigLIP run scored 0.30 accuracy against a
    0.21 macro F1, a gap that makes accuracy alone actively misleading here.
    """

    macro_f1: float = 0.0
    macro_recall: float = 0.0
    per_class_recall: dict[str, float] = field(default_factory=dict)
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    specificity: float = 0.0
    f1: float = 0.0
    auc: float | None = None
    malignant_auc: float | None = None
    balanced_accuracy: float = 0.0
    worst_class_recall: str | None = None
    false_negatives: int = 0
    false_positives: int = 0
    support: int = 0
    per_class: dict[str, dict] = field(default_factory=dict)
    confusion_matrix: list[list[int]] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def confusion(y_true: Sequence[int], y_pred: Sequence[int], n_classes: int) -> np.ndarray:
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred, strict=True):
        matrix[int(t), int(p)] += 1
    return matrix


def one_vs_rest(
    y_true: Sequence[int], y_pred: Sequence[int], n_classes: int
) -> dict[int, BinaryMetrics]:
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    out: dict[int, BinaryMetrics] = {}
    for c in range(n_classes):
        tp = int(((yp == c) & (yt == c)).sum())
        fp = int(((yp == c) & (yt != c)).sum())
        fn = int(((yp != c) & (yt == c)).sum())
        tn = int(((yp != c) & (yt != c)).sum())
        out[c] = BinaryMetrics(tp=tp, fp=fp, fn=fn, tn=tn)
    return out


def roc_auc_binary(y_true: Sequence[int], y_score: Sequence[float]) -> float | None:
    """Rank-based AUC. Returns None if only one class is present."""
    y = np.asarray(y_true)
    s = np.asarray(y_score, dtype=float)
    if len(np.unique(y)) < 2:
        return None
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=float)
    sorted_scores = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    rank_sum = ranks[y == 1].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def evaluate(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    n_classes: int,
    class_names: Sequence[str] | None = None,
    malignant_indices: Sequence[int] | None = None,
    y_score: Sequence[float] | None = None,
    y_score_malignant: Sequence[float] | None = None,
) -> EvaluationReport:
    """Compute the full metric suite.

    When `malignant_indices` is supplied, the headline precision/recall/F1 are
    computed over the malignant-vs-benign problem rather than a flat micro
    average, because that is the clinically meaningful axis for screening.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    names = list(class_names) if class_names else [str(i) for i in range(n_classes)]

    # Guard against a misaligned prediction array: a length mismatch would
    # otherwise broadcast into silently wrong metrics.
    if len(yt) != len(yp):
        raise ValueError(f"y_true and y_pred must be the same length, got {len(yt)} and {len(yp)}")

    matrix = confusion(yt, yp, n_classes)
    per_class = one_vs_rest(yt, yp, n_classes)
    accuracy = _safe_div(int((yt == yp).sum()), len(yt))

    recalls = [m.recall for m in per_class.values() if m.support > 0]
    balanced_accuracy = float(np.mean(recalls)) if recalls else 0.0

    # Macro F1 over classes actually present, and the matching per-class recall.
    present = [c for c in range(n_classes) if per_class[c].support > 0]
    macro_f1 = float(np.mean([per_class[c].f1 for c in present])) if present else 0.0
    per_class_recall = {names[c]: per_class[c].recall for c in present}
    worst_class = min(present, key=lambda c: per_class[c].recall) if present else None

    # Malignant-vs-benign is the screening axis. Each malignant-vs-benign score is
    # the summed probability the model assigns to any malignant class, which is
    # the quantity a triage threshold should actually consume.
    malignant = set(malignant_indices or [])
    malignant_auc: float | None = None
    if malignant and y_score is not None:
        scores = np.asarray(y_score, dtype=float)
        if scores.ndim == 2 and scores.shape[1] == n_classes:
            yt_bin = np.isin(yt, list(malignant)).astype(int)
            malignant_auc = roc_auc_binary(yt_bin, scores[:, list(malignant)].sum(axis=1))
        else:
            malignant_auc = roc_auc_binary(yt_bin, scores)
    if malignant:
        yt_bin = np.isin(yt, list(malignant)).astype(int)
        yp_bin = np.isin(yp, list(malignant)).astype(int)
        malignant_metrics = BinaryMetrics(
            tp=int(((yp_bin == 1) & (yt_bin == 1)).sum()),
            fp=int(((yp_bin == 1) & (yt_bin == 0)).sum()),
            fn=int(((yp_bin == 0) & (yt_bin == 1)).sum()),
            tn=int(((yp_bin == 0) & (yt_bin == 0)).sum()),
        )
        headline_precision = malignant_metrics.precision
        headline_recall = malignant_metrics.recall
        headline_f1 = malignant_metrics.f1
        headline_specificity = malignant_metrics.specificity
        false_negatives = malignant_metrics.fn
        false_positives = malignant_metrics.fp
        auc = (
            roc_auc_binary(yt_bin, y_score_malignant)
            if y_score_malignant is not None
            else roc_auc_binary(yt_bin, yp_bin)
        )
        if malignant_auc is None:
            # Only fall back to hard-label ranking when no probability scores
            # were supplied; a probability-based AUC is strictly better and must
            # not be overwritten by the thresholded one.
            malignant_auc = auc
    else:
        headline_precision = float(np.mean([m.precision for m in per_class.values()]))
        headline_recall = float(np.mean([m.recall for m in per_class.values()]))
        headline_f1 = float(np.mean([m.f1 for m in per_class.values()]))
        headline_specificity = float(np.mean([m.specificity for m in per_class.values()]))
        false_negatives = sum(m.fn for m in per_class.values())
        false_positives = sum(m.fp for m in per_class.values())
        auc = roc_auc_binary(yp, y_score) if y_score is not None else None

    return EvaluationReport(
        macro_f1=macro_f1,
        macro_recall=balanced_accuracy,
        per_class_recall=per_class_recall,
        accuracy=accuracy,
        worst_class_recall=names[worst_class] if worst_class is not None else None,
        malignant_auc=malignant_auc,
        precision=headline_precision,
        recall=headline_recall,
        specificity=headline_specificity,
        f1=headline_f1,
        auc=auc,
        balanced_accuracy=balanced_accuracy,
        false_negatives=false_negatives,
        false_positives=false_positives,
        support=len(yt),
        per_class={names[c]: per_class[c].as_dict() for c in range(n_classes)},
        confusion_matrix=matrix.tolist(),
    )


def weighted_class_weights(counts: Sequence[int]) -> np.ndarray:
    """Inverse-frequency weights, normalised to mean 1.

    HAM10000 is roughly 2/3 benign; without this the classifier learns to always
    answer "benign" and looks deceptively accurate.
    """
    counts_arr = np.asarray(counts, dtype=float)
    if counts_arr.sum() == 0:
        return np.ones_like(counts_arr)
    weights = counts_arr.sum() / (len(counts_arr) * np.clip(counts_arr, 1e-9, None))
    return weights / weights.mean()
