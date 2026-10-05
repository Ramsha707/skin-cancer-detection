"""Train a `CancerHead` on cached embeddings.

Pure numpy, so this runs anywhere. Full-batch gradient descent by default: the
head is 4,612 parameters and a partition is ~1,500 samples, so one step already
sees the whole partition. `minibatch` exists to model the FedAvg local step, where
sequential passes mimic an agent that trains locally before syncing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ml.models import CancerHead, softmax_cross_entropy


@dataclass
class TrainResult:
    loss: float
    head: CancerHead
    steps: int
    samples: int
    history: list[float]


def _normalise(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def _row_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def train_head(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    init: CancerHead | None = None,
    epochs: int = 40,
    learning_rate: float = 0.5,
    minibatch: int | None = None,
    seed: int = 0,
    class_weight: dict[int, float] | None = None,
) -> TrainResult:
    """Fit the linear probe.

    `class_weight` re-weights each class in the loss. HAM10000's four-class set is
    77% benign, so unweighted training collapses onto `benign_lesion` and the
    malignant recall -- the number that actually matters clinically -- stays near
    zero. Inverse-frequency weights are the cheap fix.
    """
    x = _normalise(np.asarray(embeddings, dtype=np.float32))
    y = np.asarray(labels, dtype=np.int64)
    n = x.shape[0]

    if init is None:
        head = CancerHead()
    else:
        head = CancerHead.from_state(init.state_dict())

    # One weight per class, not per weight element: `weights[c]` is the sample
    # weight applied to any row whose label is c.
    weights = np.ones(len(head.bias), dtype=np.float32)
    if class_weight:
        for c, w in class_weight.items():
            if 0 <= c < len(weights) and (y == c).any():
                weights[c] = w

    rng = np.random.default_rng(seed)
    history: list[float] = []
    steps = 0

    for _ in range(epochs):
        if minibatch and minibatch < n:
            order = rng.permutation(n)
            batches = [order[i : i + minibatch] for i in range(0, n, minibatch)]
        else:
            batches = [np.arange(n)]

        for batch in batches:
            xb, yb = x[batch], y[batch]
            if class_weight:
                # Weighted cross-entropy: scale each sample's residual by its
                # class weight, then renormalise by total weight so the effective
                # learning rate does not change with the class mix.
                # `softmax_cross_entropy` returns the unweighted residual, so the
                # softmax is recomputed here rather than reusing its gradient.
                logits = head.logits(xb)
                shifted = logits - logits.max(axis=1, keepdims=True)
                exp = np.exp(shifted)
                probs = _row_softmax(logits)
                sample_weight = weights[yb]
                probs[np.arange(len(yb)), yb] -= 1.0
                probs *= sample_weight[:, None]
                probs /= sample_weight.sum()
                grad_logits = probs
                loss = float(
                    -np.log(
                        np.clip(
                            _row_softmax(logits)[np.arange(len(yb)), yb],
                            1e-12,
                            None,
                        )
                    ).mean()
                )
            else:
                loss, grad_logits = softmax_cross_entropy(head.logits(xb), yb)

            head.weights -= learning_rate * (xb.T @ grad_logits)
            head.bias -= learning_rate * grad_logits.sum(axis=0)
            steps += 1

        history.append(loss)

    final = float(history[-1]) if history else 0.0
    return TrainResult(
        loss=final,
        head=head,
        steps=steps,
        samples=n,
        history=history,
    )


def inverse_frequency_weights(labels: np.ndarray, n_classes: int = 4) -> dict[int, float]:
    """Weights inversely proportional to class frequency.

    Normalised so the mean weight is 1.0, which keeps the effective loss scale
    comparable to the unweighted case and avoids needing a different
    learning rate.
    """
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=n_classes).astype(float)
    present = counts > 0
    raw = np.zeros(n_classes, dtype=float)
    raw[present] = counts[present].sum() / (present.sum() * counts[present])
    return {c: float(raw[c]) for c in range(n_classes) if counts[c] > 0}


__all__ = ["TrainResult", "inverse_frequency_weights", "train_head"]