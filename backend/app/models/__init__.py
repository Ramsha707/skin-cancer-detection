"""ORM models.

Field names mirror `frontend/src/types/index.ts` exactly. Where the roadmap
specifies a field that carries no information for this build (for example
`DetectionResult.modality`), it is stored but nullable rather than invented.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False, index=True
    )


class HospitalAgent(Base, TimestampMixin):
    """One simulated hospital site.

    `class_distribution` and `dataset_size` describe the site's *local* partition
    only. Nothing in this table is ever transmitted to the aggregator.
    """

    __tablename__ = "hospital_agents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    agent_id: Mapped[str] = mapped_column(String(32), nullable=False, unique=True, index=True)
    location: Mapped[str | None] = mapped_column(String(160), nullable=True)

    status: Mapped[str] = mapped_column(String(24), default="idle", nullable=False, index=True)
    dataset_size: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    model_version: Mapped[str] = mapped_column(String(48), default="unassigned", nullable=False)

    accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    precision: Mapped[float | None] = mapped_column(Float, nullable=True)
    recall: Mapped[float | None] = mapped_column(Float, nullable=True)
    f1_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    loss: Mapped[float | None] = mapped_column(Float, nullable=True)

    privacy_status: Mapped[str] = mapped_column(String(24), default="protected", nullable=False)
    last_sync: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    rounds_participated: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Local-only statistics.
    class_distribution: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    local_epochs: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    device: Mapped[str | None] = mapped_column(String(32), nullable=True)

    detections: Mapped[list[DetectionResult]] = relationship(
        back_populates="agent", cascade="all, delete-orphan"
    )


class TrainingRound(Base, TimestampMixin):
    """One federated round: local training -> FedAvg -> broadcast."""

    __tablename__ = "training_rounds"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    round_number: Mapped[int] = mapped_column(Integer, nullable=False, unique=True, index=True)
    status: Mapped[str] = mapped_column(String(24), default="pending", nullable=False, index=True)

    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    participating_agents: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    global_model_version: Mapped[str | None] = mapped_column(String(48), nullable=True)

    global_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    global_precision: Mapped[float | None] = mapped_column(Float, nullable=True)
    global_recall: Mapped[float | None] = mapped_column(Float, nullable=True)
    global_f1: Mapped[float | None] = mapped_column(Float, nullable=True)
    global_auc: Mapped[float | None] = mapped_column(Float, nullable=True)
    global_loss: Mapped[float | None] = mapped_column(Float, nullable=True)

    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    samples_processed: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Per-agent local metrics for this round.
    agent_results: Mapped[dict] = mapped_column(JSON, default=list, nullable=False)


class ModelVersion(Base, TimestampMixin):
    """A global model produced by the aggregator."""

    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    version: Mapped[str] = mapped_column(String(48), nullable=False, unique=True, index=True)
    strategy: Mapped[str | None] = mapped_column(String(24), nullable=True)
    training_round: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)

    accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    precision: Mapped[float | None] = mapped_column(Float, nullable=True)
    recall: Mapped[float | None] = mapped_column(Float, nullable=True)
    specificity: Mapped[float | None] = mapped_column(Float, nullable=True)
    f1: Mapped[float | None] = mapped_column(Float, nullable=True)
    auc: Mapped[float | None] = mapped_column(Float, nullable=True)
    loss: Mapped[float | None] = mapped_column(Float, nullable=True)

    trainable_params: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_params: Mapped[int | None] = mapped_column(Integer, nullable=True)
    agents: Mapped[int | None] = mapped_column(Integer, nullable=True)

    status: Mapped[str] = mapped_column(String(24), default="candidate", nullable=False, index=True)
    is_recommended: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)


class DetectionResult(Base, TimestampMixin):
    """One inference call. Rows survive a restart, so `model_version` always
    points at the registry entry that actually produced the prediction."""

    __tablename__ = "detection_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    image_name: Mapped[str] = mapped_column(String(255), nullable=False)

    predicted_class: Mapped[str] = mapped_column(String(48), nullable=False, index=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    model_version: Mapped[str] = mapped_column(String(48), nullable=False, index=True)

    inference_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    strategy: Mapped[str | None] = mapped_column(String(24), nullable=True)

    # False => produced by the simulation harness, not by real inference.
    is_real_inference: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False, index=True
    )
    modality: Mapped[str | None] = mapped_column(String(48), nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)

    probabilities: Mapped[dict] = mapped_column(JSON, default=list, nullable=False)

    agent_id: Mapped[int | None] = mapped_column(
        ForeignKey("hospital_agents.id", ondelete="SET NULL"), nullable=True
    )
    agent: Mapped[HospitalAgent | None] = relationship(back_populates="detections")


class AuditLog(Base):
    """Append-only event ledger. Also the source of truth for the privacy
    counters: `weight_update_transferred` rows are what the dashboard counts."""

    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False, index=True
    )
    agent_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    event_type: Mapped[str] = mapped_column(String(48), nullable=False, index=True)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="success", nullable=False, index=True)
    round_number: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    details: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    __table_args__ = (UniqueConstraint("id", name="uq_audit_id"),)


class Experiment(Base, TimestampMixin):
    """A transfer-learning strategy trial.

    `is_real_result = False` means the row is a registered experiment awaiting a
    real run. The Experiments page renders those as 'Awaiting Experiment' rather
    than showing numbers.
    """

    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    model: Mapped[str] = mapped_column(String(80), default="medsiglip-448", nullable=False)
    strategy: Mapped[str | None] = mapped_column(String(24), nullable=True, index=True)

    learning_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    epochs: Mapped[int | None] = mapped_column(Integer, nullable=True)

    accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    precision: Mapped[float | None] = mapped_column(Float, nullable=True)
    recall: Mapped[float | None] = mapped_column(Float, nullable=True)
    specificity: Mapped[float | None] = mapped_column(Float, nullable=True)
    f1: Mapped[float | None] = mapped_column(Float, nullable=True)
    auc: Mapped[float | None] = mapped_column(Float, nullable=True)

    trainable_layers: Mapped[str | None] = mapped_column(String(255), nullable=True)
    trainable_params: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_params: Mapped[int | None] = mapped_column(Integer, nullable=True)
    train_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    status: Mapped[str] = mapped_column(String(24), default="awaiting", nullable=False, index=True)
    is_real_result: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)


__all__ = [
    "AuditLog",
    "DetectionResult",
    "Experiment",
    "HospitalAgent",
    "ModelVersion",
    "TrainingRound",
    "utcnow",
]
