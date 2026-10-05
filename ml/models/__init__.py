"""Model definitions for SkinFL.

Two pieces live here:

- `EmbeddingCache` persists MedSigLIP vision-tower embeddings so that a head can
  be trained and re-trained without re-encoding 8,659 images every time.
- `CancerHead` is the cancer-specific adaptation layer. It is deliberately small
  (a single linear map over the frozen 1152-d embedding) so that FedAvg has a
  tractable parameter tensor to average.

The head is initialised from text-prompt embeddings rather than randomly. SigLIP
places images and text in one shared space, so the text tower already provides a
usable classifier: the weight row for class *c* starts at the normalised
embedding of that class's prompt. That gives round 0 a non-random starting point
instead of a coin flip, which matters when only a handful of federated rounds run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EMBEDDING_DIM = 1152
N_CLASSES = 4

CANCER_CLASSES = [
    "melanoma",
    "basal_cell_carcinoma",
    "actinic_keratosis",
    "benign_lesion",
]

MALIGNANT_CLASSES = ["melanoma", "basal_cell_carcinoma", "actinic_keratosis"]

# One prompt per class. These are the *only* text encodings the head is built
# from, so they double as the provenance of the initial weights.
CLASS_PROMPTS = [
    "a dermoscopy photo of melanoma",
    "a dermoscopy photo of basal cell carcinoma",
    "a dermoscopy photo of actinic keratosis",
    "a dermoscopy photo of benign melanocytic nevus",
]


@dataclass
class EmbeddingCache:
    """Frozen vision-tower embeddings for one split of one agent.

    Stored as float32 `.npy` plus a JSON sidecar of image ids and labels. Keeping
    the label alongside the embedding is what lets a head be trained on a cache
    without ever reopening the image.

    float32 rather than float16: the whole point is to avoid re-encoding, and
    float16 rounding is the same order as the quantisation noise already present
    in the 448px resize. 1152 floats is 4.6 KB per image, so a 1,500-image
    partition is about 7 MB.
    """

    path: Path

    @property
    def meta_path(self) -> Path:
        return self.path.with_suffix(".json")

    def exists(self) -> bool:
        return self.path.is_file() and self.meta_path.is_file()

    @property
    def size(self) -> int:
        if not self.exists():
            return 0
        return len(np.load(self.path, mmap_mode="r"))

    def load(self) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Return `(embeddings, labels, image_ids)`."""
        if not self.exists():
            raise FileNotFoundError(
                f"embedding cache not found at {self.path}. "
                "Run scripts/extract_embeddings.py first."
            )
        embeddings = np.load(self.path).astype(np.float32)
        with self.meta_path.open(encoding="utf-8") as fh:
            meta = json.load(fh)
        return embeddings, np.asarray(meta["labels"], dtype=np.int64), meta["image_ids"]

    def save(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        image_ids: list[str],
        **meta: object,
    ) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.path, embeddings.astype(np.float32))
        payload = {
            "labels": [int(v) for v in labels],
            "image_ids": image_ids,
            "dim": int(embeddings.shape[1]),
            "n": int(embeddings.shape[0]),
            "class_names": CANCER_CLASSES,
        }
        payload.update(meta)
        with self.meta_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)


class CancerHead:
    """Linear probe over frozen embeddings: `logits = W @ x + b`.

    A single matrix multiply keeps the federated parameter tensor at
    1152x4 + 4 = 4,612 floats (18 KB), which is small enough that FedAvg
    arithmetic is instantaneous and large enough to be a real trainable model.

    Only numpy is used rather than torch. The head has no hidden layer, so
    autograd buys nothing, and staying in numpy means the head has no dependency
    on a torch build being present -- the backend can train it on any machine.
    """

    def __init__(self, weights: np.ndarray | None = None, bias: np.ndarray | None = None):
        if weights is None:
            weights = np.zeros((EMBEDDING_DIM, N_CLASSES), dtype=np.float32)
        if bias is None:
            bias = np.zeros(N_CLASSES, dtype=np.float32)
        self.weights = np.asarray(weights, dtype=np.float32)
        self.bias = np.asarray(bias, dtype=np.float32)

    @property
    def param_count(self) -> int:
        return int(self.weights.size + self.bias.size)

    def logits(self, embeddings: np.ndarray) -> np.ndarray:
        x = np.asarray(embeddings, dtype=np.float32)
        # Normalise first so the head sees the same unit-sphere geometry the text
        # tower produced, which is what makes the prompt initialisation meaningful.
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (x / norms) @ self.weights + self.bias

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        return self.logits(embeddings).argmax(axis=1)

    def probabilities(self, embeddings: np.ndarray) -> np.ndarray:
        logits = self.logits(embeddings)
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def state_dict(self) -> dict[str, np.ndarray]:
        return {"weights": self.weights.copy(), "bias": self.bias.copy()}

    @classmethod
    def from_state(cls, state: dict[str, np.ndarray]) -> CancerHead:
        return cls(weights=state["weights"], bias=state["bias"])

    @classmethod
    def from_text_prompts(cls, text_embeddings: np.ndarray) -> CancerHead:
        """Initialise the classifier rows from text-prompt embeddings.

        SigLIP shares one embedding space between images and text, so the text
        embedding of a class name is already a usable weight vector for that
        class. Normalising each row puts every class at the same scale, which
        stops one long prompt from dominating the argmax purely through
        magnitude.
        """
        weights = np.asarray(text_embeddings, dtype=np.float32).T.copy()
        norms = np.linalg.norm(weights, axis=0, keepdims=True)
        norms[norms == 0] = 1.0
        weights = weights / norms
        return cls(weights=weights)


def softmax_cross_entropy(
    logits: np.ndarray, labels: np.ndarray
) -> tuple[float, np.ndarray]:
    """Mean cross-entropy and the gradient w.r.t. logits.

    Returned gradient is `(n, 4)`: the standard `probs - onehot` residual. The
    caller contracts it with the normalised embeddings to get `dW`.
    """
    n = logits.shape[0]
    rows = np.arange(n)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    # Read the loss off the clean probabilities before turning `probs` into the
    # gradient in place.
    loss = float(-np.log(np.clip(probs[rows, labels], 1e-12, None)).mean())
    grad = probs.copy()
    grad[rows, labels] -= 1.0
    grad /= n
    return loss, grad


__all__ = [
    "CANCER_CLASSES",
    "CLASS_PROMPTS",
    "EMBEDDING_DIM",
    "MALIGNANT_CLASSES",
    "N_CLASSES",
    "CancerHead",
    "EmbeddingCache",
    "softmax_cross_entropy",
]