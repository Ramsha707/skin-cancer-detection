"""Week 7: federated training orchestration.

Bridges the pure-numpy engine in `app.federated.engine` to the database. One
call runs one full round -- broadcast, per-agent local fit, FedAvg, checkpoint,
broadcast back -- and writes every step to the audit ledger, because the
dashboard's privacy counters are *derived* from those ledger rows rather than
maintained separately.

Honesty rules enforced here:

- A round with no real embedding caches does not run and does not invent a row;
  `readiness()` returns the operator-facing refusal that becomes an HTTP 503.
- Global metrics are written only from an evaluation over the shared val split.
  With no val cache the weights are still saved, but every metric stays `None`.
- The only payloads ever recorded are parameter tensors, sized from the actual
  delta bytes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.federated.engine import FederatedEngine, RoundResult, get_engine
from app.models import HospitalAgent, ModelVersion, TrainingRound, utcnow
from app.services import audit as audit_service
from app.services import registry
from ml.models import CANCER_CLASSES, CancerHead

log = logging.getLogger(__name__)

PARAM_BYTES = 4
# Derived, not hardcoded: (embedding_dim + 1) bias-included floats per class.
# Keeping this in one place means the detection page, the round payload sizes
# and the registry entries cannot report three different numbers.
PARAM_COUNT = registry.head_param_count()


def param_kb() -> float:
    return PARAM_COUNT * PARAM_BYTES / 1024


@dataclass
class TrainingOutcome:
    round_result: RoundResult
    model_version: str | None


def db_agent_id(slug: str) -> str:
    """Embedding-cache slug (`agent_a`) to database agent id (`agent-a`)."""
    return slug.replace("_", "-")


def readiness() -> str | None:
    """Operator-facing refusal message, or None when a real round can run."""
    ready = registry.caches_ready()
    have = [a for a in registry.AGENT_SLUGS if ready.get(f"train_{a}")]
    if not have:
        return (
            "No agent embedding caches found, so there is nothing real to train "
            "on. Run .\\backend-venv\\Scripts\\python.exe scripts/build_splits.py, "
            "then scripts/extract_embeddings.py --split train before starting a "
            "federated round."
        )
    return None


def bootstrap_engine() -> FederatedEngine:
    """Register hospital agents and seed the global head. Idempotent.

    The global head is seeded from the newest checkpoint on disk when one exists
    and matches the partition dimension, else from zeros. It deliberately does
    *not* load the zero-shot prompt head: that path constructs the full
    878M-parameter text tower, which a round loop must not pay, and zeros train
    to the same place in a handful of epochs on a linear probe.
    """
    engine = get_engine()
    if engine.global_state is not None and engine.agents:
        return engine

    # The coordinator applies `server_lr` to the aggregated delta; take it from
    # settings so a tuned value is honoured rather than a stale constructor default.
    engine.server_lr = settings.fedavg_learning_rate

    splits: list[tuple[str, registry.CachedSplit]] = []
    for slug in registry.AGENT_SLUGS:
        split = registry.load_split("train", slug)
        if split.real:
            splits.append((slug, split))
    if not splits:
        raise RuntimeError(readiness() or "no agent embedding caches found")

    dim = int(splits[0][1].embeddings.shape[1])
    n_classes = len(CANCER_CLASSES)
    head, version = None, None
    checkpoint = registry.latest_checkpoint()
    if checkpoint is not None:
        with np.load(checkpoint) as data:
            weights, bias = data["weights"], data["bias"]
        if weights.shape == (dim, n_classes) and bias.shape == (n_classes,):
            head = CancerHead(weights=weights, bias=bias)
            version = checkpoint.stem
    if head is None:
        head = CancerHead(weights=np.zeros((dim, n_classes), dtype=np.float32))
        version = "v0"
    engine.initialise(head, version=version)

    val = registry.load_split("val")
    if val.real and int(val.embeddings.shape[1]) == dim:
        engine.set_validation(val.embeddings, val.labels)

    for slug, split in splits:
        engine.register(
            db_agent_id(slug),
            split.embeddings,
            split.labels,
            learning_rate=settings.federated_local_lr,
            local_epochs=settings.federated_local_epochs,
            patience=settings.federated_local_patience,
        )
    log.info(
        "engine bootstrapped: version=%s agents=%d dim=%d val=%s",
        version,
        len(engine.agents),
        dim,
        engine.has_validation(),
    )
    return engine


def run_round(
    db: Session,
    *,
    engine: FederatedEngine | None = None,
    runner=None,
) -> TrainingOutcome:
    """Run one federated round end to end and persist it.

    `runner`, when given, is the background worker; per-agent live state is
    noted on it so `/api/federated/state` can show progress while the round is
    in flight.
    """
    engine = engine or bootstrap_engine()
    if not engine.agents:
        raise RuntimeError(readiness() or "no agent embedding caches found")

    existing = db.scalar(select(TrainingRound).order_by(TrainingRound.round_number.desc()))
    round_number = (existing.round_number + 1) if existing else 1

    round_row = TrainingRound(
        round_number=round_number,
        status="running",
        started_at=utcnow(),
        participating_agents=len(engine.agents),
    )
    db.add(round_row)
    audit_service.log(
        db,
        event_type=audit_service.EVENT_FEDAVG_STARTED,
        message=f"Round {round_number} started with {len(engine.agents)} agents.",
        round_number=round_number,
        details={"agents": sorted(engine.agents), "server_lr": engine.server_lr},
    )
    db.commit()

    agent_names = {
        row.agent_id: row.name for row in db.scalars(select(HospitalAgent)).all()
    }

    def observe(runner_obj, local, delta) -> None:
        """One agent finished local training. Audit it and commit immediately,
        so the live dashboard sees agents flip to 'synced' one by one."""
        audit_service.log(
            db,
            event_type=audit_service.EVENT_LOCAL_TRAINING_COMPLETED,
            message=(
                f"{runner_obj.agent_id} finished local training on "
                f"{local.samples} private samples "
                f"({local.epochs_run} epochs, macro-F1 {local.local_macro_f1:.3f})."
            ),
            agent_id=runner_obj.agent_id,
            round_number=round_number,
            details={
                "samples": local.samples,
                "epochs_run": local.epochs_run,
                "loss": local.loss,
            },
        )
        audit_service.log(
            db,
            event_type=audit_service.EVENT_MODEL_UPDATE_GENERATED,
            message=(
                f"{runner_obj.agent_id} produced a weight delta of "
                f"{runner_obj.last_delta_norm:.4f} L2 over "
                f"{runner_obj.param_count} parameters."
            ),
            agent_id=runner_obj.agent_id,
            round_number=round_number,
            details={
                "delta_l2_norm": runner_obj.last_delta_norm,
                "param_count": runner_obj.param_count,
                "tensors": 2,
            },
        )
        audit_service.transmit(
            db,
            event_type=audit_service.EVENT_MODEL_UPDATE_TRANSFERRED,
            message=(
                f"{runner_obj.agent_id} sent weight delta "
                f"({runner_obj.last_update_kb:.2f} KB) to the aggregator."
            ),
            agent_id=runner_obj.agent_id,
            payload_kb=runner_obj.last_update_kb or param_kb(),
            round_number=round_number,
            details={"direction": "hospital_to_aggregator"},
        )
        if runner is not None:
            runner.note_agent(
                runner_obj.agent_id,
                status="training",
                phase="aggregating",
                samples=local.samples,
                local_loss=local.loss,
                local_accuracy=local.local_accuracy,
                local_recall=local.local_recall,
                local_f1=local.local_f1,
                weight_delta_norm=runner_obj.last_delta_norm,
                update_size_kb=runner_obj.last_update_kb,
            )
        db.commit()

    for runner_obj in engine.agents.values():
        agent = db.scalar(
            select(HospitalAgent).where(HospitalAgent.agent_id == runner_obj.agent_id)
        )
        if agent is not None:
            agent.status = "training"
        audit_service.log(
            db,
            event_type=audit_service.EVENT_LOCAL_TRAINING_STARTED,
            message=f"{runner_obj.agent_id} began local training on its private partition.",
            agent_id=runner_obj.agent_id,
            round_number=round_number,
            details={"samples": runner_obj.samples},
        )
        if runner is not None:
            runner.note_agent(
                runner_obj.agent_id,
                status="training",
                phase="training",
                samples=runner_obj.samples,
            )
        db.commit()

    result = engine.run_round(round_number, on_agent_trained=observe)

    audit_service.log(
        db,
        event_type=audit_service.EVENT_FEDAVG_COMPLETED,
        message=(
            f"FedAvg combined {result.participating} weight deltas into "
            f"{result.global_version} (sample-count weighted mean, server_lr="
            f"{engine.server_lr:g}). No image data involved."
        ),
        round_number=round_number,
        details={
            "participants": result.participating,
            "samples": result.samples,
            "server_lr": engine.server_lr,
        },
    )

    # Real weights are saved whether or not a val split exists to score them;
    # metrics stay None when there is nothing to evaluate against.
    registry.save_head(engine.global_head(), result.global_version)
    for previous in db.scalars(select(ModelVersion).where(ModelVersion.status == "active")):
        previous.status = "archived"
        previous.is_recommended = False
    db.add(
        ModelVersion(
            version=result.global_version,
            strategy="frozen",
            training_round=round_number,
            accuracy=result.accuracy,
            precision=result.precision,
            recall=result.recall,
            f1=result.f1,
            auc=result.auc,
            loss=result.loss,
            trainable_params=PARAM_COUNT,
            total_params=PARAM_COUNT,
            agents=result.participating,
            status="active",
            # The newest active model is the one to deploy; archiving above
            # clears the flag on the previous generation.
            is_recommended=True,
            notes=(
                f"FedAvg over {result.participating} hospital partitions "
                f"({result.samples} cached images). "
                + (
                    f"Evaluated on the shared val split in "
                    f"{result.duration_seconds:.1f}s."
                    if result.evaluated
                    else "No val split cached, so no global metrics were computed."
                )
            ),
        )
    )
    audit_service.log(
        db,
        event_type=audit_service.EVENT_GLOBAL_MODEL_GENERATED,
        message=(
            f"Global model {result.global_version} written to the registry and "
            "checkpoint store."
        ),
        round_number=round_number,
        details={
            "strategy": "frozen",
            "trainable_params": PARAM_COUNT,
            "evaluated": result.evaluated,
        },
    )

    for runner_obj in engine.agents.values():
        audit_service.transmit(
            db,
            event_type=audit_service.EVENT_MODEL_BROADCAST,
            message=f"Aggregator broadcast {result.global_version} to {runner_obj.agent_id}.",
            agent_id=runner_obj.agent_id,
            payload_kb=param_kb(),
            round_number=round_number,
            details={
                "direction": "aggregator_to_hospital",
                "model_version": result.global_version,
            },
        )
        agent = db.scalar(
            select(HospitalAgent).where(HospitalAgent.agent_id == runner_obj.agent_id)
        )
        if agent is not None:
            agent.model_version = result.global_version
            agent.status = "synced"
            agent.last_sync = utcnow()
            agent.rounds_participated += 1
            entry = next(
                (e for e in result.agent_results if e["agent_id"] == runner_obj.agent_id),
                None,
            )
            if entry is not None:
                agent.accuracy = entry["local_accuracy"]
                agent.precision = entry["local_precision"]
                agent.recall = entry["local_recall"]
                agent.f1_score = entry["local_f1"]
                agent.loss = entry["loss"]
                agent.local_epochs = entry["epochs_run"]
        audit_service.log(
            db,
            event_type=audit_service.EVENT_AGENT_SYNCHRONIZED,
            message=(
                f"{runner_obj.agent_id} synchronized to {result.global_version}."
            ),
            agent_id=runner_obj.agent_id,
            round_number=round_number,
            details={
                "direction": "aggregator_to_hospital",
                "model_version": result.global_version,
            },
        )
        if runner is not None:
            runner.note_agent(
                runner_obj.agent_id,
                status="synced",
                phase="synced",
            )

    agent_results = [
        {**entry, "agent_name": agent_names.get(entry["agent_id"], entry["agent_id"])}
        for entry in result.agent_results
    ]

    round_row.status = "completed"
    round_row.completed_at = utcnow()
    round_row.global_model_version = result.global_version
    round_row.global_accuracy = result.accuracy
    round_row.global_precision = result.precision
    round_row.global_recall = result.recall
    round_row.global_f1 = result.f1
    round_row.global_auc = result.auc
    round_row.global_loss = result.loss
    round_row.duration_seconds = result.duration_seconds
    round_row.samples_processed = result.samples
    round_row.agent_results = agent_results

    audit_service.log(
        db,
        event_type=audit_service.EVENT_ROUND_COMPLETED,
        message=(
            f"Round {round_number} completed in {result.duration_seconds:.1f}s "
            f"with {result.participating} agents; macro-F1 "
            + (f"{result.macro_f1:.3f}." if result.evaluated else "not computed.")
        ),
        round_number=round_number,
        details={
            "global_model_version": result.global_version,
            "evaluated": result.evaluated,
            "duration_seconds": result.duration_seconds,
        },
    )
    db.commit()
    return TrainingOutcome(round_result=result, model_version=result.global_version)


__all__ = [
    "PARAM_COUNT",
    "TrainingOutcome",
    "bootstrap_engine",
    "db_agent_id",
    "param_kb",
    "readiness",
    "run_round",
]
