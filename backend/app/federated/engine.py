"""Week 7: federated averaging over `CancerHead` parameters.

The privacy invariant this module exists to enforce: a hospital agent is only
ever reached through three methods, and all three exchange tensors.

    fit(...)           -> local metrics (never data)
    update()           -> weight delta (a flat float32 vector)
    load(global_state) -> None

There is no parameter or attribute anywhere in this file that accepts a dataset
object, an image path, or a PIL image. `HospitalAgentRunner` holds its
embeddings privately and never returns them; the only thing that leaves is a
float32 vector of length `embedding_dim * n_classes + n_classes`. That is what
makes the README's "raw patient images shared: 0" claim structural rather than
aspirational -- there is no code path through which a pixel can reach the
aggregator.

This layer is deliberately pure numpy and database-free. Persistence, audit
ledger entries and agent bookkeeping live in `app.services.training`; the
background worker lives in `app.federated.runner`.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from app.services import experiments as experiments_service
from app.services import metrics as metrics_service
from ml.models import (
    CANCER_CLASSES,
    MALIGNANT_CLASSES,
    N_CLASSES,
    CancerHead,
    softmax_cross_entropy,
)

log = logging.getLogger(__name__)

MALIGNANT_INDICES = [CANCER_CLASSES.index(c) for c in MALIGNANT_CLASSES]

STRATEGY_FROZEN = "frozen"


@dataclass
class LocalResult:
    """What an agent reports after local training. Metrics only, no data."""

    agent_id: str
    samples: int
    loss: float
    epochs_run: int
    local_accuracy: float
    local_precision: float
    local_recall: float
    local_f1: float
    local_macro_f1: float
    elapsed_seconds: float

    def as_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "samples": self.samples,
            "loss": self.loss,
            "epochs_run": self.epochs_run,
            "local_accuracy": self.local_accuracy,
            "local_precision": self.local_precision,
            "local_recall": self.local_recall,
            "local_f1": self.local_f1,
            "local_macro_f1": self.local_macro_f1,
        }


class HospitalAgentRunner:
    """One hospital's private copy of the model and its private embeddings.

    `_embeddings` is deliberately a leading-underscore attribute and is never
    returned by any method. `update()` returns a flat parameter delta and
    nothing else; `fit()` returns numbers.
    """

    def __init__(
        self,
        agent_id: str,
        embeddings: np.ndarray,
        labels: np.ndarray,
        *,
        head: CancerHead | None = None,
        learning_rate: float = 5e-2,
        local_epochs: int = 40,
        patience: int = 10,
    ) -> None:
        self.agent_id = agent_id
        self._embeddings = np.asarray(embeddings, dtype=np.float32)
        self._labels = np.asarray(labels, dtype=np.int64)
        dim = self._embeddings.shape[1]
        if head is None:
            self._head = CancerHead(weights=np.zeros((dim, N_CLASSES), dtype=np.float32))
        else:
            self._head = CancerHead.from_state(head.state_dict())
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.patience = patience
        self.rounds_participated = 0

        # Live bookkeeping for the dashboard's per-agent panel. Metrics and
        # tensor norms only; set during `run_round`.
        self.last_result: LocalResult | None = None
        self.last_delta_norm: float | None = None
        self.last_update_kb: float | None = None

        self._broadcast_flat: np.ndarray | None = None

    @property
    def samples(self) -> int:
        return int(self._embeddings.shape[0])

    @property
    def param_dim(self) -> int:
        return int(self._embeddings.shape[1])

    @property
    def param_count(self) -> int:
        return self._head.param_count

    def fit(self, *, seed: int = 0) -> LocalResult:
        """Local training. Reads private data, reports only metrics."""
        t0 = time.perf_counter()
        head, epochs_run, _holdout_f1 = experiments_service.train_head(
            self._embeddings,
            self._labels,
            init=self._head,
            epochs=self.local_epochs,
            lr=self.learning_rate,
            patience=self.patience,
            seed=seed,
        )
        self._head = head

        report = metrics_service.evaluate(
            self._labels,
            head.predict(self._embeddings),
            n_classes=N_CLASSES,
            class_names=CANCER_CLASSES,
            malignant_indices=MALIGNANT_INDICES,
        )
        loss, _ = softmax_cross_entropy(head.logits(self._embeddings), self._labels)

        result = LocalResult(
            agent_id=self.agent_id,
            samples=self.samples,
            loss=loss,
            epochs_run=epochs_run,
            local_accuracy=report.accuracy,
            local_precision=report.precision,
            local_recall=report.recall,
            local_f1=report.f1,
            local_macro_f1=report.macro_f1,
            elapsed_seconds=time.perf_counter() - t0,
        )
        self.last_result = result
        return result

    def update(self) -> np.ndarray:
        """The only thing that ever leaves the hospital: a flat weight delta.

        Measured against the state most recently received via `load()`, so the
        aggregator averages *changes*, not full models.
        """
        if self._broadcast_flat is None:
            raise RuntimeError(
                f"{self.agent_id} has no broadcast state; call load() before update()"
            )
        delta = self._flatten(self._head.state_dict()) - self._broadcast_flat
        self.last_delta_norm = float(np.linalg.norm(delta))
        self.last_update_kb = delta.nbytes / 1024
        return delta

    def load(self, flat: np.ndarray) -> None:
        """Install the broadcast global weights."""
        state = np.asarray(flat, dtype=np.float32).copy()
        self._head = self._unflatten(state)
        self._broadcast_flat = state

    def score(self, embeddings: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        """Evaluate on a shared held-out set. Returns metrics, not predictions."""
        report = metrics_service.evaluate(
            labels,
            self._head.predict(embeddings),
            n_classes=N_CLASSES,
            class_names=CANCER_CLASSES,
            malignant_indices=MALIGNANT_INDICES,
        )
        return {"accuracy": report.accuracy, "macro_f1": report.macro_f1, "f1": report.f1}

    def _flatten(self, state: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([state["weights"].ravel(), state["bias"].ravel()]).astype(
            np.float32
        )

    def _unflatten(self, flat: np.ndarray) -> CancerHead:
        dim = self.param_dim
        return CancerHead(
            weights=flat[: dim * N_CLASSES].reshape(dim, N_CLASSES),
            bias=flat[dim * N_CLASSES :],
        )


def weighted_average(deltas: list[np.ndarray], weights: list[int]) -> np.ndarray:
    """Sample-count weighted mean of client updates (the FedAvg step).

    A site holding 2,049 images should not have the same vote as one holding
    1,413. `server_lr` is applied by the caller, so this function is exactly the
    weighted mean and nothing else, which keeps it trivially unit-testable.
    """
    if not deltas:
        raise ValueError("weighted_average requires at least one client update")
    if len(deltas) != len(weights):
        raise ValueError(
            f"weighted_average got {len(deltas)} deltas and {len(weights)} weights"
        )
    stacked = np.vstack(deltas).astype(np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if w.sum() == 0:
        raise ValueError("weighted_average requires at least one non-zero client weight")
    combined = (stacked * (w / w.sum())[:, None]).sum(axis=0)
    return combined.astype(np.float32)


@dataclass
class RoundResult:
    """Everything one completed round produced. Metrics only."""

    round_number: int
    global_version: str
    participating: int
    samples: int
    duration_seconds: float
    loss: float | None = None
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    specificity: float | None = None
    f1: float | None = None
    macro_f1: float | None = None
    auc: float | None = None
    # False when no val split was cached: weights are still real, but no global
    # metrics were computed, and none are invented.
    evaluated: bool = False
    agent_results: list[dict] = field(default_factory=list)


class FederatedEngine:
    """Server-side coordinator: broadcast, collect, aggregate.

    Holds the global weights and the shared validation set. It never holds a
    client partition, so it is structurally unable to train on hospital data.
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
        self._dim: int | None = None
        self._lock = threading.RLock()

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
        with self._lock:
            self._agents[agent_id] = runner
        return runner

    @property
    def agents(self) -> dict[str, HospitalAgentRunner]:
        with self._lock:
            return dict(self._agents)

    def has_validation(self) -> bool:
        return self._val_embeddings is not None and len(self._val_embeddings) > 0

    def set_validation(self, embeddings: np.ndarray, labels: np.ndarray) -> None:
        """Install the shared held-out split used to score the global model."""
        self._val_embeddings = np.asarray(embeddings, dtype=np.float32)
        self._val_labels = np.asarray(labels, dtype=np.int64)

    def initialise(self, head: CancerHead, *, version: str = "v0") -> str:
        """Seed the global model from an existing head. Called before round 1."""
        self._dim = int(head.weights.shape[0])
        self.global_state = self._flatten(head.state_dict())
        self.current_global_version = version
        return version

    def run_round(
        self,
        round_number: int,
        *,
        seed: int = 0,
        on_agent_trained=None,
    ) -> RoundResult:
        """One full round: broadcast -> local fit -> collect -> aggregate.

        `on_agent_trained(runner, local, delta)` is an observation hook for the
        persistence layer; it must not mutate the engine.
        """
        if self.global_state is None:
            raise RuntimeError("engine not initialised; call initialise() first")

        with self._lock:
            targets = list(self._agents.values())
        if not targets:
            raise ValueError("no agents registered for training")

        t0 = time.perf_counter()
        deltas: list[np.ndarray] = []
        counts: list[int] = []
        agent_results: list[dict] = []

        # Sequential is deliberate: the head is 4,612 parameters, so local
        # training is seconds per agent, and a sequential loop keeps the audit
        # trail deterministic.
        for runner in targets:
            runner.load(self.global_state)
            local = runner.fit(seed=seed)
            delta = runner.update()
            deltas.append(delta)
            counts.append(local.samples)
            runner.rounds_participated += 1
            entry = local.as_dict()
            entry["weight_delta_norm"] = runner.last_delta_norm
            entry["update_size_kb"] = runner.last_update_kb
            agent_results.append(entry)
            if on_agent_trained is not None:
                on_agent_trained(runner, local, delta)

        step = weighted_average(deltas, counts)
        self.global_state = self.global_state + np.float32(self.server_lr) * step
        self.current_global_version = f"v{round_number}"

        metrics: dict | None = None
        if self.has_validation():
            metrics = self.evaluate(self._val_embeddings, self._val_labels)

        result = RoundResult(
            round_number=round_number,
            global_version=self.current_global_version,
            participating=len(targets),
            samples=int(sum(counts)),
            duration_seconds=time.perf_counter() - t0,
            evaluated=metrics is not None,
            agent_results=agent_results,
        )
        if metrics is not None:
            result.loss = metrics["loss"]
            result.accuracy = metrics["accuracy"]
            result.precision = metrics["precision"]
            result.recall = metrics["recall"]
            result.specificity = metrics["specificity"]
            result.f1 = metrics["f1"]
            result.macro_f1 = metrics["macro_f1"]
            result.auc = metrics["auc"]

        self.rounds.append(result)
        log.info(
            "round %d: participating=%d samples=%d macro_f1=%s evaluated=%s",
            round_number,
            result.participating,
            result.samples,
            f"{result.macro_f1:.4f}" if result.macro_f1 is not None else "n/a",
            result.evaluated,
        )
        return result

    def evaluate(self, embeddings: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        """Score the *global* model on the shared held-out set."""
        head = self.global_head()
        probs = head.probabilities(embeddings)
        report = metrics_service.evaluate(
            labels,
            probs.argmax(axis=1),
            n_classes=N_CLASSES,
            class_names=CANCER_CLASSES,
            malignant_indices=MALIGNANT_INDICES,
            y_score=probs,
            y_score_malignant=probs[:, MALIGNANT_INDICES].sum(axis=1),
        )
        loss, _ = softmax_cross_entropy(head.logits(embeddings), labels)
        return {
            "loss": loss,
            "accuracy": report.accuracy,
            "precision": report.precision,
            "recall": report.recall,
            "specificity": report.specificity,
            "f1": report.f1,
            "macro_f1": report.macro_f1,
            "auc": report.auc,
        }

    def global_head(self) -> CancerHead:
        """The global weights as a usable head.

        Round-trips through `_flatten`/`_unflatten`, the same conversion the
        agents perform, so evaluation cannot drift from what FedAvg produced.
        """
        if self.global_state is None or self._dim is None:
            raise RuntimeError("engine not initialised; call initialise() first")
        return self._unflatten(self.global_state)

    def reset(self) -> None:
        """Drop all runtime state: global model, agents, round history."""
        with self._lock:
            self.global_state = None
            self.current_global_version = None
            self.rounds.clear()
            self._agents.clear()
            self._dim = None

    def _flatten(self, state: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([state["weights"].ravel(), state["bias"].ravel()]).astype(
            np.float32
        )

    def _unflatten(self, flat: np.ndarray) -> CancerHead:
        dim = int(self._dim)
        return CancerHead(
            weights=flat[: dim * N_CLASSES].reshape(dim, N_CLASSES),
            bias=flat[dim * N_CLASSES :],
        )


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
    "MALIGNANT_INDICES",
    "STRATEGY_FROZEN",
    "FederatedEngine",
    "HospitalAgentRunner",
    "LocalResult",
    "RoundResult",
    "get_engine",
    "set_engine",
    "weighted_average",
]
