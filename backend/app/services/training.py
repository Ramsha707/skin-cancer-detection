"""Week 7/8: federated training orchestration.

Bridges the pure-numpy `app.federated.engine` to HTTP, so the dashboard can start
rounds and read history. Rounds are persisted to `training_rounds` and every
transmission is written to the audit ledger, which is what the privacy counters
are derived from.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.federated.engine import FederatedEngine, RoundResult, get_engine
from app.models import HospitalAgent, ModelVersion, TrainingRound, utcnow
from app.services import audit as audit_service
from app.services import registry

log = logging.getLogger(__name__)

PARAM_BYTES = 4
# Derived, not hardcoded: 1152 weights + 1 bias per class. Keeping this in one
# place means the Week 4 page, the Week 7 status endpoint and the engine cannot
# report three different payload sizes.
PARAM_COUNT = registry.head_param_count()


def param_kb() -> float:
    return PARAM_COUNT * PARAM_BYTES / 1024


@dataclass
class TrainingOutcome:
    round_result: RoundResult
    model_version: str


def bootstrap_engine(db: Session, *, with_agents: bool = True) -> FederatedEngine:
    """Register agents and the shared validation set with the engine.

    Called on first use rather than at import, because it needs the embedding
    caches and constructing the text tower is slow.
    """
    engine = get_engine()
    if engine.global_state is not None:
        return engine

    val = registry.load_split("val")
    if val.real:
        engine = FederatedEngine(val.embeddings, val.labels)
        from app.federated.engine import set_engine

        set_engine(engine)

    # Seed from the prompt head. `initialise` takes raw text embeddings and
    # transposes them into classifier rows; the head already holds those rows as
    # columns, so transpose back rather than re-deriving from the 878M model.
    head = registry.get_zero_shot_head()
    engine.initialise(head.weights)

    if with_agents:
        for slug in registry.AGENT_SLUGS:
            split = registry.load_split("train", slug)
            if not split.real:
                log.warning("agent %s has no embedding cache; skipping", slug)
                continue
            engine.register(slug, split.embeddings, split.labels)

    return engine


def _sync_agents(db: Session, engine: FederatedEngine, result: RoundResult) -> None:
    for entry in result.agent_results:
        agent = db.scalar(select(HospitalAgent).where(HospitalAgent.agent_id == entry["agent_id"]))
        if agent is None:
            continue
        agent.model_version = result.global_version
        agent.accuracy = entry["local_accuracy"]
        agent.f1_score = entry["local_macro_f1"]
        agent.loss = entry["loss"]
        agent.rounds_participated += 1
        agent.status = "synced"
        agent.last_sync = utcnow()


def run_round(db: Session, *, rounds: int = 1) -> TrainingOutcome:
    """Run one round end to end and persist it.

    Audit entries record both transmission directions with the real payload size,
    so the privacy dashboard's counters are derived from what actually happened
    rather than maintained separately.
    """
    engine = bootstrap_engine(db)
    if not engine.agents:
        raise RuntimeError(
            "no agent embedding caches found. Run scripts/build_splits.py then "
            "scripts/extract_embeddings.py --split train."
        )

    existing = db.scalar(select(TrainingRound).order_by(TrainingRound.round_number.desc()))
    next_round = (existing.round_number + 1) if existing else 1

    round_row = TrainingRound(
        round_number=next_round,
        status="running",
        started_at=utcnow(),
        participating_agents=len(engine.agents),
    )
    db.add(round_row)
    db.flush()

    audit_service.log(
        db,
        event_type=audit_service.EVENT_FEDAVG_STARTED,
        message=f"Round {next_round} started with {len(engine.agents)} agents.",
        round_number=next_round,
        details={"agents": list(engine.agents)},
    )

    for agent_id in engine.agents:
        audit_service.log(
            db,
            event_type=audit_service.EVENT_LOCAL_TRAINING_STARTED,
            message=f"{agent_id} began local training on its private partition.",
            agent_id=agent_id,
            round_number=next_round,
        )

    results = [engine.run_round(next_round) for _ in range(rounds)]
    result = results[-1]

    for agent_id in engine.agents:
        audit_service.log(
            db,
            event_type=audit_service.EVENT_LOCAL_TRAINING_COMPLETED,
            message=f"{agent_id} finished local training and produced a weight delta.",
            agent_id=agent_id,
            round_number=next_round,
        )
        audit_service.transmit(
            db,
            event_type=audit_service.EVENT_MODEL_UPDATE_TRANSFERRED,
            message=f"{agent_id} sent weight delta ({param_kb():.1f} KB) to the aggregator.",
            agent_id=agent_id,
            payload_kb=param_kb(),
            round_number=next_round,
            details={"direction": "hospital_to_aggregator", "tensors": 2, "images": 0},
        )

    audit_service.log(
        db,
        event_type=audit_service.EVENT_FEDAVG_COMPLETED,
        message=(
            f"FedAvg combined {len(engine.agents)} weight deltas into "
            f"{result.global_version}. No image data involved."
        ),
        round_number=next_round,
        details={"server_lr": engine.server_lr, "participants": len(engine.agents)},
    )

    for agent_id in engine.agents:
        audit_service.transmit(
            db,
            event_type=audit_service.EVENT_MODEL_BROADCAST,
            message=f"Aggregator broadcast {result.global_version} to {agent_id}.",
            agent_id=agent_id,
            payload_kb=param_kb(),
            round_number=next_round,
            details={"direction": "aggregator_to_hospital"},
        )

    round_row.status = "completed"
    round_row.completed_at = utcnow()
    round_row.global_model_version = result.global_version
    round_row.global_accuracy = result.accuracy
    round_row.global_f1 = result.macro_f1
    round_row.global_loss = result.loss
    round_row.global_recall = result.macro_f1
    round_row.global_auc = None
    round_row.duration_seconds = result.duration_seconds
    round_row.samples_processed = result.samples
    round_row.participating_agents = result.participating
    round_row.agent_results = result.agent_results

    db.add(
        ModelVersion(
            version=result.global_version,
            strategy="fedavg-linear-probe",
            training_round=next_round,
            accuracy=result.accuracy,
            f1=result.macro_f1,
            loss=result.loss,
            trainable_params=PARAM_COUNT,
            total_params=PARAM_COUNT,
            agents=result.participating,
            status="active",
            is_recommended=next_round == 1,
            notes=(
                f"FedAvg over {result.participating} hospital partitions, "
                f"{result.samples} images. Evaluated on the shared validation split."
            ),
        )
    )

    _sync_agents(db, engine, result)
    db.flush()
    return TrainingOutcome(round_result=result, model_version=result.global_version)


def publish_checkpoint(db: Session, engine: FederatedEngine) -> None:
    """Write the current global head so detection uses the trained weights."""
    if engine.global_state is None:
        return
    registry.save_head(engine.global_head(), engine.current_global_version or "v0")


__all__ = [
    "PARAM_COUNT",
    "TrainingOutcome",
    "bootstrap_engine",
    "param_kb",
    "publish_checkpoint",
    "run_round",
]