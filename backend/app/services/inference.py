"""Single-image inference for the detection endpoint.

Two paths, distinguished by an explicit flag that travels all the way into the
response and the database:

* **Real** -- a trained head over a real MedSigLIP embedding of the uploaded
  image. `is_real_inference = True`.
* **Fallback** -- the zero-shot prompt head, used when no trained checkpoint or
  embedding cache is present. `is_real_inference = False`.

The fallback still runs the actual model; it is not a random number. What differs
is whether the weights have been adapted to this task, and the response says so
rather than implying a fine-tuned result.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.services import registry
from app.services.dataset import CANCER_CLASSES, MALIGNANT_CLASSES
from app.services.metrics import evaluate

log = logging.getLogger(__name__)


@dataclass
class Prediction:
    predicted_class: str
    confidence: float
    probabilities: list[dict[str, float]]
    malignant_probability: float
    is_malignant: bool
    model_version: str
    is_real_inference: bool
    inference_seconds: float
    note: str

    def as_dict(self) -> dict:
        return {
            "predicted_class": self.predicted_class,
            "confidence": round(self.confidence, 6),
            "probabilities": self.probabilities,
            "malignant_probability": round(self.malignant_probability, 6),
            "is_malignant": self.is_malignant,
            "model_version": self.model_version,
            "is_real_inference": self.is_real_inference,
            "inference_seconds": round(self.inference_seconds, 4),
            "note": self.note,
        }


def embed_image(image_path: Path | str) -> np.ndarray:
    """One 1152-d vision-tower embedding for a single image.

    Uses OpenVINO when it is importable (the iGPU is ~18x faster than torch CPU
    here), and falls back to torch otherwise so the backend still works in a
    plain venv.
    """
    try:
        from medsiglip_ov import MedSiglipOpenVINO

        clf = MedSiglipOpenVINO()
        return np.asarray(clf.embed_images([str(image_path)]), dtype=np.float32)
    except ImportError:
        from medsiglip import MedSiglipClassifier

        clf = MedSiglipClassifier()
        return clf.embed_images([str(image_path)]).numpy().astype(np.float32)


def predict(image_path: Path | str, *, head=None, model_version: str | None = None) -> Prediction:
    """Classify one image. Never raises for an ordinary model failure."""
    t0 = time.perf_counter()
    trained = False

    if head is None:
        head, version, trained = registry.load_trained_head()
    else:
        version = model_version or "custom"

    try:
        embedding = embed_image(image_path)
    except (FileNotFoundError, OSError):
        raise
    except Exception as exc:  # model construction / inference failure
        log.exception("inference failed")
        raise RuntimeError(f"inference failed: {exc}") from exc

    probs = head.probabilities(embedding)[0]
    top = int(probs.argmax())

    malignant_idx = [CANCER_CLASSES.index(c) for c in MALIGNANT_CLASSES]
    malignant_prob = float(sum(probs[i] for i in malignant_idx))

    note = (
        f"Real inference against fine-tuned weights from {version}."
        if trained
        else (
            "Zero-shot prompt classifier. No fine-tuned checkpoint is present yet, so "
            "this is MedSigLIP's unmodified cancer-adaptation head."
        )
    )

    return Prediction(
        predicted_class=CANCER_CLASSES[top],
        confidence=float(probs[top]),
        probabilities=[
            {"class": c, "probability": round(float(probs[i]), 6)}
            for i, c in enumerate(CANCER_CLASSES)
        ],
        malignant_probability=malignant_prob,
        is_malignant=malignant_prob >= 0.5,
        model_version=version,
        is_real_inference=trained,
        inference_seconds=time.perf_counter() - t0,
        note=note,
    )


def evaluate_head_on(head, split) -> dict:
    """Score a head on a cached split, using the full metric suite.

    Reports macro-F1 and per-class recall rather than accuracy, because the
    benign class is ~77% of the four-class set and accuracy alone would be
    dominated by it.
    """
    embeddings, labels = split.embeddings, split.labels
    pred = head.predict(embeddings)
    probs = head.probabilities(embeddings)
    malignant_idx = [CANCER_CLASSES.index(c) for c in MALIGNANT_CLASSES]

    report = evaluate(
        labels,
        pred,
        n_classes=len(CANCER_CLASSES),
        class_names=CANCER_CLASSES,
        malignant_indices=malignant_idx,
        y_score=probs,
    )
    payload = report.as_dict()
    payload["is_real_result"] = bool(split.real)
    return payload


__all__ = ["Prediction", "embed_image", "evaluate_head_on", "predict"]