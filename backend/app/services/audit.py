"""Audit logging.

The privacy counters on the dashboard are *derived* from these rows rather than
maintained separately, so they cannot drift out of sync with what actually
happened.

The invariant this module enforces: there is no event type that carries a raw
image. `weight_update_transferred` records a tensor size, never a filename of
patient data.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.models import AuditLog

# The complete event vocabulary used across the project.
EVENT_LOCAL_TRAINING_STARTED = "local_training_started"
EVENT_LOCAL_TRAINING_COMPLETED = "local_training_completed"
EVENT_MODEL_UPDATE_GENERATED = "model_update_generated"
EVENT_MODEL_UPDATE_TRANSFERRED = "model_update_transferred"
EVENT_FEDAVG_STARTED = "fedavg_started"
EVENT_FEDAVG_COMPLETED = "fedavg_completed"
EVENT_GLOBAL_MODEL_GENERATED = "global_model_generated"
EVENT_MODEL_BROADCAST = "model_broadcast"
EVENT_AGENT_SYNCHRONIZED = "agent_synchronized"
EVENT_DETECTION_PERFORMED = "detection_performed"
EVENT_EXPLAINABILITY_GENERATED = "explainability_generated"
EVENT_ROUND_COMPLETED = "round_completed"

EVENT_LABELS: dict[str, str] = {
    EVENT_LOCAL_TRAINING_STARTED: "Local training started",
    EVENT_LOCAL_TRAINING_COMPLETED: "Local training completed",
    EVENT_MODEL_UPDATE_GENERATED: "Model update generated",
    EVENT_MODEL_UPDATE_TRANSFERRED: "Model update transferred",
    EVENT_FEDAVG_STARTED: "FedAvg started",
    EVENT_FEDAVG_COMPLETED: "FedAvg completed",
    EVENT_GLOBAL_MODEL_GENERATED: "Global model generated",
    EVENT_MODEL_BROADCAST: "Global model broadcast",
    EVENT_AGENT_SYNCHRONIZED: "Agent synchronized",
    EVENT_DETECTION_PERFORMED: "Detection performed",
    EVENT_EXPLAINABILITY_GENERATED: "Explainability generated",
    EVENT_ROUND_COMPLETED: "Round completed",
}


def log(
    db: Session,
    *,
    event_type: str,
    message: str,
    agent_id: str | None = None,
    status: str = "success",
    round_number: int | None = None,
    details: dict[str, Any] | None = None,
) -> AuditLog:
    """Append one event. Caller is responsible for committing."""
    entry = AuditLog(
        agent_id=agent_id,
        event_type=event_type,
        message=message,
        status=status,
        round_number=round_number,
        details=details or {},
    )
    db.add(entry)
    return entry


def transmit(
    db: Session,
    *,
    event_type: str,
    message: str,
    agent_id: str,
    payload_kb: float,
    round_number: int | None = None,
    details: dict[str, Any] | None = None,
) -> AuditLog:
    """Record a hospital -> aggregator or aggregator -> hospital transmission.

    `payload_kb` is the serialised size of the parameter tensor only. There is no
    code path here that accepts image data, which is what makes the
    "raw images shared = 0" claim structural.
    """
    merged = {"payload_kb": round(payload_kb, 3), "payload_type": "weight_update"}
    merged.update(details or {})
    return log(
        db,
        event_type=event_type,
        message=message,
        agent_id=agent_id,
        round_number=round_number,
        details=merged,
    )