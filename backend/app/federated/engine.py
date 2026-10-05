"""Federated averaging over `CancerHead` parameters.

The privacy invariant this module exists to enforce: a hospital agent is only ever
called through three methods, and all three exchange tensors.

    fit(state, embeddings, labels) -> local metrics
    update()                        -> weight delta
    load(state)                     -> None

There is no parameter or attribute anywhere in this file that accepts a dataset
object, an image path, or a PIL image. `Agent` holds its embeddings privately and
never exposes them; the only thing that leaves is a float32 array of length
`1152*4 + 4`. That is what makes the "raw patient images shared: 0" claim in the
README structural rather than aspirational -- you cannot leak a pixel through an
API that has no parameter to leak one through.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from app.services.dataset import CANCER_CLASSES, MALIGNANT_CLASSES
from ml.models import CancerHead
from ml.training.head import inverse_frequency_weights, train_head

log = logging.getLogger(__name__)

STRATEGY_FROZEN = "frozen"
STRATEGY_SELECTIVE = "selective"
STRATEGY_FULL = "full"


@dataclass
class LocalResult:
    """What an agent reports after local training. Metrics only, no data."""

    agent_id: str
    samples: int
    loss: float
    steps: int
    local_accuracy: float
    local_macro_f1: float
    elapsed_seconds: float


class HospitalAgentRunner:
    """One hospital's private copy of the model and its private embeddings.

    `embeddings` is deliberately a plain attribute with a leading underscore and
    is never returned. `delta()` returns a flat parameter vector and nothing else.
    """

    def __init__(
        self,
        agent_id: str,
        embeddings: np.ndarray,
        labels: np.ndarray,
        *,
        head: CancerHead | None = None,
        learning_rate: float = 0.5,
        local_epochs: int = 40,
        balanced: bool = True,
    ) -> None:
        self.agent_id = agent_id
        self._embeddings = np.asarray(embeddings, dtype=np.float32)
        self._labels = np.asarray(labels, dtype=np.int64)
        self._head = head or CancerHead()
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.balanced = balanced
        self.rounds_participated = 0

    @property
    def samples(self) -> int:
        return int(self._embeddings.shape[0])

    @property
    def param_count(self) -> int:
        return self._head.param_count

    def fit(self, *, seed: int = 0) -> LocalResult:
        """Local training. Reads private data, reports only metrics."""
        t0 = time.perf_counter()
        weights = (
            inverse_frequency_weights(self._labels, len(CANCER_CLASSES)) if self.balanced else None
        )
        result = train_head(
            self._embeddings,
            self._labels,
            init=self._head,
            epochs=self.local_epochs,
            learning_rate=self.learning_rate,
            seed=seed,
            class_weight=weights,
        )
        self._head = result.head

        pred = self._head.predict(self._embeddings)
        acc = float((pred == self._labels).mean()) if self.samples else 0.0
        macro_f1 = float(
            np.mean(
                [
                    _f1(self._labels, pred, c)
                    for c in range(len(CANCER_CLASSES))
                    if (self._labels == c).any()
                ]
            )
        )

        return LocalResult(
            agent_id=self.agent_id,
            samples=self.samples,
            loss=result.loss,
            steps=result.steps,
            local_accuracy=acc,
            local_macro_f1=macro_f1,
            elapsed_seconds=time.perf_counter() - t0,
        )

    def update(self) -> np.ndarray:
        """The only thing that ever leaves the hospital: a flat parameter vector."""
        return self._flatten(self._head.state_dict())

    def load(self, flat: np.ndarray) -> None:
        self._head = self._unflatten(flat)

    def score(self, embeddings: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        """Evaluate on the shared held-out set. Returns metrics, not predictions."""
        pred = self._head.predict(embeddings)
        return {"accuracy": float((pred == labels).mean()), "macro_f1": _macro_f1(labels, pred)}

    @staticmethod
    def _flatten(state: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate(
            [state["weights"].ravel(), state["bias"].ravel()]
        ).astype(np.float32)

    @staticmethod
    def _unflatten(flat: np.ndarray) -> CancerHead:
        n_w = 1152 * len(CANCER_CLASSES)
        return CancerHead(
            weights=flat[:n_w].reshape(1152, len(CANCER_CLASSES)),
            bias=flat[n_w:],
        )


def _f1(y_true: np.ndarray, y_pred: np.ndarray, c: int) -> float:
    tp = int(((y_pred == c) & (y_true == c)).sum())
    fp = int(((y_pred == c) & (y_true != c)).sum())
    fn = int(((y_pred != c) & (y_true == c)).sum())
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 4) -> float:
    present = [c for c in range(n_classes) if (y_true == c).any()]
    return float(np.mean([_f1(y_true, y_pred, c) for c in present])) if present else 0.0


def fedavg(deltas: list[np.ndarray], weights: list[int], *, server_lr: float = 1.0) -> np.ndarray:
    """Weighted mean of client updates.

    The mean is weighted by each site's sample count, which is the standard
    FedAvg objective: a site holding 2,049 images should not have the same vote as
    one holding 1,413. Size ratio is capped at ~1.45x by the partitioner, so the
    weighting is meaningful without letting the largest site dominate.

    `server_lr` is the FedProx-style server step size, left at 1.0 (plain average).
    """
    if not deltas:
        raise ValueError("fedavg requires at least one client update")
    stacked = np.vstack(deltas).astype(np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if w.sum() == 0:
        raise ValueError("fedavg requires at least one non-zero client weight")
    combined = (stacked * (w / w.sum())[:, None]).sum(axis=0)
    return (server_lr * combined).astype(np.float32)


@dataclass
class RoundResult:
    round_number: int
    global_version: str
    loss: float
    accuracy: float
    macro_f1: float
    participating: int
    samples: int
    duration_seconds: float
    agent_results: list[dict] = field(default_factory=list)


class FederatedEngine:
    """Server-side coordinator: broadcast, collect, aggregate.

    Holds global weights and the shared validation set. It never holds a client
    partition, so it is structurally unable to train on hospital data.
    """

    def __init__(
        self,
        val_embeddings: np.ndarray | None = None,
        val_labels: np.ndarray | None = None,
        *,
        server_lr: float = 1.0,
    ) -> None:
        self._val_embeddings = (
            None if val_embeddings is None else np.asarray(val_embeddings, np.float32)
        )
        self._val_labels = None if val_labels is None else np.asarray(val_labels, np.int64)
        self.server_lr = server_lr
        self.global_state: np.ndarray | None = None
        self.current_global_version: str | None = None
        self.rounds: list[RoundResult] = []
        self._agents: dict[str, HospitalAgentRunner] = {}

    def register(
        self,
        agent_id: str,
        embeddings: np.ndarray,
        labels: np.ndarray,
        **kwargs,
    ) -> HospitalAgentRunner:
        """Hand an agent its private data. Called once, at setup, from the same
        process. In a real deployment this is the agent's own dataset and this
        call never crosses a network boundary."""
        runner = HospitalAgentRunner(agent_id, embeddings, labels, **kwargs)
        self._agents[agent_id] = runner
        return runner

    @property
    def agents(self) -> dict[str, HospitalAgentRunner]:
        return dict(self._agents)

    def has_validation(self) -> bool:
        return self._val_embeddings is not None and len(self._val_embeddings) > 0

    def initialise(self, weights: np.ndarray) -> str:
        """Seed global weights from a `(1152, 4)` classifier matrix.

        Takes the matrix rather than text embeddings so the caller can reuse an
        already-built prompt head. Passing `(4, 1152)` here would silently
        produce a transposed classifier that still runs but scores near-uniform.
        """
        head = CancerHead(weights=weights)
        proto = HospitalAgentRunner(
            "__init__", np.zeros((1, 1152), np.float32), np.zeros(1, np.int64)
        )
        proto.load(HospitalAgentRunner._flatten(head.state_dict()))
        self.global_state = proto.update()
        self.current_global_version = "v0-prompt-init"
        return self.current_global_version

    def run_round(self, round_number: int, agent_ids: list[str] | None = None) -> RoundResult:
        """One full round: broadcast -> local fit -> collect -> aggregate."""
        if self.global_state is None:
            raise RuntimeError("engine not initialised; call initialise() first")

        t0 = time.perf_counter()
        targets = agent_ids or list(self._agents)
        if not targets:
            raise ValueError("no agents available to train")

        deltas: list[np.ndarray] = []
        weights: list[int] = []
        results: list[dict] = []

        # Broadcast, then parallel collection would go here; sequential is fine at
        # this head size and keeps the audit trail deterministic.
        for aid in targets:
            runner = self._agents[aid]
            runner.load(self.global_state)
            local = runner.fit()
            deltas.append(runner.update())
            weights.append(local.samples)
            runner.rounds_participated += 1
            results.append(
                {
                    "agent_id": local.agent_id,
                    "samples": local.samples,
                    "loss": local.loss,
                    "local_accuracy": local.local_accuracy,
                    "local_macro_f1": local.local_macro_f1,
                    "steps": local.steps,
                }
            )

        self.global_state = fedavg(deltas, weights, server_lr=self.server_lr)
        self.current_global_version = f"v{round_number}"

        if self.has_validation():
            metrics = self.evaluate(self._val_embeddings, self._val_labels)
        else:
            metrics = {"accuracy": 0.0, "macro_f1": 0.0, "loss": 0.0}

        result = RoundResult(
            round_number=round_number,
            global_version=self.current_global_version,
            loss=metrics.get("loss", 0.0),
            accuracy=metrics["accuracy"],
            macro_f1=metrics["macro_f1"],
            participating=len(targets),
            samples=int(sum(weights)),
            duration_seconds=time.perf_counter() - t0,
            agent_results=results,
        )
        self.rounds.append(result)
        log.info(
            "round %d: macro_f1=%.4f acc=%.4f (%d agents, %d samples)",
            round_number,
            result.macro_f1,
            result.accuracy,
            result.participating,
            result.samples,
        )
        return result

    def evaluate(self, embeddings: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        """Score the *global* model on the shared held-out set."""
        if self.global_state is None:
            raise RuntimeError("engine not initialised")
        logits = self.global_head().logits(embeddings)
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=1, keepdims=True)
        pred = probs.argmax(axis=1)
        loss = float(
            -np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, None)).mean()
        )
        return {
            "accuracy": float((pred == labels).mean()),
            "macro_f1": _macro_f1(labels, pred),
            "loss": loss,
        }

    def predict(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Global-model probabilities, for the detection endpoint."""
        if self.global_state is None:
            raise RuntimeError("engine not initialised")
        probs = self.global_head().probabilities(embeddings)
        return probs.argmax(axis=1), probs

    def global_head(self) -> CancerHead:
        """The global weights as a usable head.

        Round-trips through `_flatten`/`_unflatten`, which is the same conversion
        clients perform, so evaluation cannot drift from what FedAvg actually
        produced.
        """
        if self.global_state is None:
            raise RuntimeError("engine not initialised")
        return HospitalAgentRunner._unflatten(self.global_state)


_engine: FederatedEngine | None = None


def get_engine() -> FederatedEngine:
    global _engine
    if _engine is None:
        _engine = FederatedEngine()
    return _engine


def set_engine(engine: FederatedEngine | None) -> None:
    """Test seam."""
    global _engine
    _engine = engine


__all__ = [
    "CANCER_CLASSES",
    "MALIGNANT_CLASSES",
    "STRATEGY_FROZEN",
    "STRATEGY_FULL",
    "STRATEGY_SELECTIVE",
    "FederatedEngine",
    "HospitalAgentRunner",
    "LocalResult",
    "RoundResult",
    "fedavg",
    "get_engine",
    "set_engine",
]
