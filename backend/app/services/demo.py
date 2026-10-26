"""Week 12: the judge demo checklist.

A demo script is only useful if it cannot lie. Every step below is derived from
the same live probes the owning endpoint uses (integration diagnostics, the audit
ledger, the privacy summary, the registry caches). A step is `ready` only when
its evidence exists *right now*; otherwise it carries the route or command that
produces it. The page can therefore be opened cold and still tell the truth.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.config import settings
from app.federated.runner import get_runner
from app.schemas import DemoChecklist, DemoStep
from app.services import dataset as dataset_service
from app.services import explainability as explainability_service
from app.services import integration as integration_service
from app.services import privacy as privacy_service
from app.services import registry

DISCLAIMER = (
    "Research prototype. Not a medical device. Must not be used for clinical "
    "decision making."
)

TRAIN_COMMAND = "Run one federated round from /federated-training"


def _step(
    number: int,
    key: str,
    title: str,
    description: str,
    *,
    required: bool,
    ready: bool,
    evidence: str,
    route: str | None = None,
    command: str | None = None,
) -> dict:
    return {
        "number": number,
        "key": key,
        "title": title,
        "description": description,
        "required": required,
        "ready": ready,
        "evidence": evidence,
        "route": route,
        "command": command,
    }


def build_checklist(db: Session) -> dict:
    """Assemble the 12-step walkthrough from live state."""
    integration = integration_service.build_integration(db)
    modules = {module["key"]: module for module in integration["modules"]}
    counts = integration["counts"]
    bus = integration["websocket"]

    manifest = dataset_service.load_manifest()
    caches = registry.caches_ready()
    checkpoint = registry.latest_checkpoint()
    train_ready = [slug for slug in registry.AGENT_SLUGS if caches.get(f"train_{slug}")]
    runner = get_runner().snapshot()
    xai = explainability_service.availability()

    privacy = privacy_service.build_summary(db, ledger_limit=1)
    counters = privacy["counters"]
    invariants_ok = all(item["enforced"] for item in privacy["invariants"])

    evaluated_ready = checkpoint is not None and bool(caches.get("val"))
    model_ready = counts["model_versions"] > 0 or checkpoint is not None

    steps = [
        _step(
            1,
            "api",
            "HTTP API and event bus",
            "Confirm FastAPI is serving and the live notification channel is attached.",
            required=True,
            ready=True,
            evidence=(
                f"FastAPI v{settings.version} on /api; "
                f"{bus['subscribers']} WebSocket subscriber(s), "
                f"{bus['events_seen']} events seen."
            ),
            route="/settings",
        ),
        _step(
            2,
            "database",
            "Database and seeded agents",
            "Every page reads persisted rows, not in-memory sample data.",
            required=True,
            ready=modules["database"]["ready"],
            evidence=modules["database"]["detail"],
            route="/audit",
        ),
        _step(
            3,
            "dataset",
            "HAM10000 dataset manifest",
            "Patient-level splits are derived from the real metadata file.",
            required=True,
            ready=manifest.loaded,
            evidence=(
                f"{manifest.total_images:,} images across {manifest.patients:,} patients."
                if manifest.loaded
                else manifest.message
            ),
            command=None if manifest.loaded else integration_service.SPLITS_COMMAND,
        ),
        _step(
            4,
            "agents",
            "Four hospital agents",
            "Show the agent cards: each holds its own partition and privacy status.",
            required=True,
            ready=modules["agents"]["ready"],
            evidence=(
                f"{counts['agents']} of {len(registry.AGENT_SLUGS)} agents registered."
            ),
            route="/agents",
        ),
        _step(
            5,
            "embeddings",
            "Local embedding partitions",
            "Cached embeddings make each local head update reproducible and fast.",
            required=True,
            ready=bool(train_ready) and bool(caches.get("val")),
            evidence=(
                f"{len(train_ready)}/{len(registry.AGENT_SLUGS)} train partitions; "
                f"validation cache {'present' if caches.get('val') else 'missing'}."
            ),
            command=(
                None
                if train_ready and caches.get("val")
                else integration_service.EXTRACT_COMMAND
            ),
        ),
        _step(
            6,
            "federated_round",
            "A completed federated round",
            "Local training, FedAvg aggregation and the broadcast are all ledgered.",
            required=True,
            ready=counts["completed_rounds"] > 0 and runner["status"] != "error",
            evidence=(
                f"{counts['completed_rounds']} completed of {counts['rounds']} rounds; "
                f"runner status={runner['status']}."
            ),
            route="/federated-training",
            command=None if counts["completed_rounds"] > 0 else TRAIN_COMMAND,
        ),
        _step(
            7,
            "model_registry",
            "Registered global model",
            "The aggregated head is versioned; the newest checkpoint is what serves.",
            required=True,
            ready=model_ready,
            evidence=(
                f"{counts['model_versions']} version(s) in the registry; checkpoint "
                + (checkpoint.name if checkpoint else "none on disk")
                + "."
            ),
            route="/models",
            command=None if model_ready else TRAIN_COMMAND,
        ),
        _step(
            8,
            "evaluation",
            "Held-out evaluation report",
            "Accuracy, per-class recall, confusion matrix and ROC over the validation split.",
            required=True,
            ready=evaluated_ready,
            evidence=(
                f"Checkpoint {checkpoint.stem if checkpoint else 'missing'} and "
                f"validation cache {'present' if caches.get('val') else 'missing'}; "
                "the report is computed from these on request."
            ),
            route="/performance",
            command=None if evaluated_ready else TRAIN_COMMAND,
        ),
        _step(
            9,
            "detection",
            "Live detection",
            "Upload a dermoscopy image and show the class probabilities plus the ledger row.",
            required=True,
            ready=evaluated_ready,
            evidence=(
                "Trained checkpoint and validation cache are available for a real inference."
                if evaluated_ready
                else "Detection reports not-ready until a checkpoint and the val cache exist."
            ),
            route="/detection",
            command=None if evaluated_ready else TRAIN_COMMAND,
        ),
        _step(
            10,
            "explainability",
            "Grad-CAM explanation",
            "Attribute the prediction to image regions; unavailable hosts say so explicitly.",
            required=False,
            ready=xai.available,
            evidence=(
                f"torch and weights available on {xai.device}."
                if xai.available
                else xai.reason
            ),
            route="/explainability",
        ),
        _step(
            11,
            "privacy",
            "Privacy invariants",
            "Show that transmissions carry tensors only and local events carry no "
            "patient identifiers.",
            required=True,
            ready=invariants_ok,
            evidence=(
                f"{counters['raw_images_shared']} raw images shared; "
                f"{counters['transmissions_logged']} tensor transmission(s); "
                f"{counters['agents_protected']}/{counters['total_agents']} agents protected."
            ),
            route="/privacy",
        ),
        _step(
            12,
            "audit",
            "Audit ledger",
            "Close on the append-only record that every previous step wrote into.",
            required=True,
            ready=counts["audit_events"] > 0,
            evidence=f"{counts['audit_events']} ledger row(s); newest events stream live.",
            route="/audit",
        ),
    ]

    completed = sum(1 for step in steps if step["ready"])
    ready = all(step["ready"] for step in steps if step["required"])

    return DemoChecklist(
        ready=ready,
        completed=completed,
        total=len(steps),
        generated_at=datetime.now(timezone.utc),
        steps=[DemoStep(**step) for step in steps],
        disclaimer=DISCLAIMER,
    ).model_dump()


__all__ = ["DISCLAIMER", "build_checklist"]
