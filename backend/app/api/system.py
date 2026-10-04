"""System health and dashboard aggregation."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.config import settings
from app.database.base import get_db
from app.models import AuditLog, HospitalAgent, ModelVersion, TrainingRound
from app.schemas import DashboardSummary, DatasetInfo, PrivacySummary, SystemStatus
from app.services import audit as audit_service
from app.services import dataset as dataset_service

router = APIRouter(prefix="/api", tags=["system"])


# --------------------------------------------------------------------- health


@router.get("/system/health", response_model=SystemStatus)
def health(db: Session = Depends(get_db)) -> SystemStatus:
    """Liveness probe plus a truthful account of what is and is not loaded."""
    try:
        db.execute(select(1))
        db_status = "connected"
    except SQLAlchemyError:  # pragma: no cover - only on a broken DB file
        # Any database-layer failure means "not usable", which is all this
        # health probe needs to report.
        db_status = "unavailable"

    manifest = dataset_service.load_manifest()

    return SystemStatus(
        api="online",
        database=db_status,
        pretrained_model=_model_cache_status(),
        dataset=(
            f"{manifest.name}: {manifest.total_images} images, {manifest.patients} patients"
            if manifest.loaded
            else "not configured"
        ),
        demo_mode=not manifest.loaded,
        version=settings.version,
        server_time=datetime.now(timezone.utc),
    )


def _model_cache_status() -> str:
    """Report on the local HuggingFace cache without importing torch."""
    repo = settings.pretrained_model_id.replace("/", "--")
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{repo}"
    if not cache_dir.exists():
        return f"not cached ({settings.pretrained_model_id})"
    weights = sorted(cache_dir.rglob("*.safetensors"))
    if not weights:
        return "cache present, weights unresolved"
    total_mb = sum(w.stat().st_size for w in weights) / 1e6
    return f"cached · {total_mb:.0f} MB on disk"


# ------------------------------------------------------------------ dashboard


@router.get("/dashboard/summary", response_model=DashboardSummary)
def dashboard_summary(db: Session = Depends(get_db)) -> DashboardSummary:
    """One aggregate call that powers the entire dashboard."""
    manifest = dataset_service.load_manifest()

    total_agents = db.scalar(select(func.count()).select_from(HospitalAgent)) or 0
    active_agents = (
        db.scalar(
            select(func.count())
            .select_from(HospitalAgent)
            .where(HospitalAgent.status.in_(["training", "synced"]))
        )
        or 0
    )
    total_samples = db.scalar(select(func.coalesce(func.sum(HospitalAgent.dataset_size), 0))) or 0

    current_round = db.scalar(select(func.max(TrainingRound.round_number))) or 0
    rounds_completed = (
        db.scalar(
            select(func.count())
            .select_from(TrainingRound)
            .where(TrainingRound.status == "completed")
        )
        or 0
    )
    total_versions = db.scalar(select(func.count()).select_from(ModelVersion)) or 0

    latest_round = db.scalar(
        select(TrainingRound)
        .where(TrainingRound.status == "completed")
        .order_by(TrainingRound.round_number.desc())
        .limit(1)
    )

    protected = (
        db.scalar(
            select(func.count())
            .select_from(HospitalAgent)
            .where(HospitalAgent.privacy_status == "protected")
        )
        or 0
    )

    return DashboardSummary(
        total_agents=total_agents,
        active_agents=active_agents,
        current_round=current_round,
        global_model_version=latest_round.global_model_version if latest_round else None,
        global_accuracy=latest_round.global_accuracy if latest_round else None,
        global_recall=latest_round.global_recall if latest_round else None,
        global_f1=latest_round.global_f1 if latest_round else None,
        global_auc=latest_round.global_auc if latest_round else None,
        cancer_classes=settings.num_classes,
        total_samples=total_samples,
        total_model_versions=total_versions,
        total_rounds_completed=rounds_completed,
        privacy=_privacy_summary(db, total_agents=total_agents, protected=protected),
        dataset=_dataset_info(manifest),
        demo_mode=not manifest.loaded,
        real_model_loaded=_model_cache_status().startswith("cached"),
    )


def _privacy_summary(db: Session, *, total_agents: int, protected: int) -> PrivacySummary:
    """Derive the privacy counters from the audit ledger.

    Counting transmissions here rather than maintaining a separate counter means
    the dashboard can never claim a transmission that the ledger does not show,
    nor hide one that it does.
    """
    transmissions = (
        db.scalar(
            select(func.count())
            .select_from(AuditLog)
            .where(AuditLog.event_type == audit_service.EVENT_MODEL_UPDATE_TRANSFERRED)
        )
        or 0
    )

    total_kb = 0.0
    rows = db.scalars(
        select(AuditLog.details).where(
            AuditLog.event_type == audit_service.EVENT_MODEL_UPDATE_TRANSFERRED
        )
    )
    for details in rows:
        if isinstance(details, dict) and "payload_kb" in details:
            try:
                total_kb += float(details["payload_kb"])
            except (TypeError, ValueError):
                continue

    return PrivacySummary(
        # Structurally zero: no event type in the vocabulary can carry an image.
        raw_images_shared=0,
        model_updates_shared=transmissions > 0,
        total_update_size_kb=total_kb,
        transmissions_logged=transmissions,
        agents_protected=protected,
        total_agents=total_agents,
    )


def _dataset_info(manifest: dataset_service.DatasetManifest) -> DatasetInfo:
    if not manifest.loaded:
        return DatasetInfo(loaded=False, message=manifest.message, source_path=None)

    train, val, test = dataset_service.patient_level_split(manifest)
    return DatasetInfo(
        name=manifest.name,
        loaded=True,
        total_images=manifest.total_images,
        total_patients=manifest.patients,
        class_counts=manifest.class_counts,
        imbalance_ratio=manifest.imbalance_ratio,
        patient_level_split=True,
        splits={"train": len(train), "val": len(val), "test": len(test)},
        source_path=str(manifest.root) if manifest.root else None,
        message=manifest.message,
    )
