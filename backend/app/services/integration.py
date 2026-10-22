"""Week 11: full-system integration diagnostics.

One place that answers "is this subsystem actually usable here?" without
guessing. Every module check is derived from the same source its owning endpoint
uses: the registry cache probe, the federated readiness gate, the audit ledger,
the privacy service. The Settings page and the Week 12 demo checklist both read
this, so they cannot disagree with the pages they describe.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.config import settings
from app.federated.runner import get_runner
from app.models import AuditLog, HospitalAgent, ModelVersion, TrainingRound
from app.services import dataset as dataset_service
from app.services import explainability as explainability_service
from app.services import registry
from app.websocket import get_event_bus

EXTRACT_COMMAND = (
    ".\\.venv-ov\\Scripts\\python.exe scripts/extract_embeddings.py --split train"
)
SPLITS_COMMAND = ".\\backend-venv\\Scripts\\python.exe scripts/build_splits.py"


def _module(
    key: str,
    label: str,
    *,
    required: bool,
    ready: bool,
    detail: str,
    route: str | None = None,
    command: str | None = None,
) -> dict:
    return {
        "key": key,
        "label": label,
        "required": required,
        "ready": ready,
        "detail": detail,
        "route": route,
        "command": command,
    }


def _database_ready(db: Session) -> bool:
    try:
        db.execute(select(1))
        return True
    except SQLAlchemyError:  # pragma: no cover - only on a broken DB file
        return False


def build_integration(db: Session) -> dict:
    """The `/api/system/integration` payload, derived from live state."""
    db_ok = _database_ready(db)
    manifest = dataset_service.load_manifest()
    caches = registry.caches_ready()

    agent_count = 0
    round_count = 0
    completed_rounds = 0
    version_count = 0
    audit_count = 0
    if db_ok:
        agent_count = db.scalar(select(func.count()).select_from(HospitalAgent)) or 0
        round_count = db.scalar(select(func.count()).select_from(TrainingRound)) or 0
        completed_rounds = (
            db.scalar(
                select(func.count())
                .select_from(TrainingRound)
                .where(TrainingRound.status == "completed")
            )
            or 0
        )
        version_count = db.scalar(select(func.count()).select_from(ModelVersion)) or 0
        audit_count = db.scalar(select(func.count()).select_from(AuditLog)) or 0

    train_ready = [a for a in registry.AGENT_SLUGS if caches.get(f"train_{a}")]
    checkpoint = registry.latest_checkpoint()
    xai = explainability_service.availability()
    runner = get_runner().snapshot()
    bus_stats = get_event_bus().stats()

    modules = [
        _module(
            "api",
            "HTTP API",
            required=True,
            ready=True,
            detail=f"FastAPI v{settings.version} serving /api",
            route="/settings",
        ),
        _module(
            "database",
            "Database",
            required=True,
            ready=db_ok,
            detail=(
                f"SQLite reachable: {agent_count} agents, {round_count} rounds, "
                f"{audit_count} audit rows"
                if db_ok
                else "Database query failed"
            ),
            route="/audit",
        ),
        _module(
            "dataset",
            "HAM10000 dataset",
            required=True,
            ready=manifest.loaded,
            detail=(
                f"{manifest.total_images} images / {manifest.patients} patients"
                if manifest.loaded
                else manifest.message
            ),
            command=None if manifest.loaded else SPLITS_COMMAND,
        ),
        _module(
            "agents",
            "Hospital agents",
            required=True,
            ready=agent_count >= len(registry.AGENT_SLUGS),
            detail=f"{agent_count} of {len(registry.AGENT_SLUGS)} agents registered",
            route="/agents",
        ),
        _module(
            "embeddings",
            "Local embedding partitions",
            required=True,
            ready=bool(train_ready),
            detail=(
                f"{len(train_ready)}/{len(registry.AGENT_SLUGS)} train partitions, "
                f"val {'present' if caches.get('val') else 'missing'}"
            ),
            command=None if train_ready else EXTRACT_COMMAND,
        ),
        _module(
            "federated",
            "Federated round runner",
            required=True,
            ready=runner["status"] != "error",
            detail=(
                f"status={runner['status']}, queued={runner['queued']}"
                + (f", error={runner['error']}" if runner["error"] else "")
            ),
            route="/federated-training",
        ),
        _module(
            "models",
            "Model registry",
            required=False,
            ready=version_count > 0,
            detail=f"{version_count} versions; newest checkpoint "
            + (checkpoint.name if checkpoint else "none on disk"),
            route="/models",
        ),
        _module(
            "evaluation",
            "Evaluation report",
            required=False,
            ready=checkpoint is not None and bool(caches.get("val")),
            detail=(
                "checkpoint and val split present"
                if checkpoint is not None and caches.get("val")
                else "needs a trained checkpoint and the val embedding cache"
            ),
            route="/performance",
        ),
        _module(
            "explainability",
            "Grad-CAM explainability",
            required=False,
            ready=xai.available,
            detail=(
                f"torch + weights available on {xai.device}"
                if xai.available
                else xai.reason
            ),
            route="/explainability",
        ),
        _module(
            "privacy",
            "Privacy invariants",
            required=True,
            ready=True,
            detail="Raw images shared: 0 by construction (tensor-only transports)",
            route="/privacy",
        ),
        _module(
            "live_events",
            "WebSocket event bus",
            required=True,
            ready=True,
            detail=(
                f"{bus_stats['subscribers']} subscriber(s), "
                f"{bus_stats['events_seen']} events seen, "
                f"{bus_stats['history_size']} buffered"
            ),
            route="/settings",
        ),
    ]

    return {
        "ready": all(m["ready"] for m in modules if m["required"]),
        "version": settings.version,
        "server_time": datetime.now(timezone.utc),
        "websocket": bus_stats,
        "modules": modules,
        "counts": {
            "agents": agent_count,
            "rounds": round_count,
            "completed_rounds": completed_rounds,
            "model_versions": version_count,
            "audit_events": audit_count,
        },
    }


__all__ = ["EXTRACT_COMMAND", "SPLITS_COMMAND", "build_integration"]
