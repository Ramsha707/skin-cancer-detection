"""Week 6: experiment endpoints.

Exposes the strategy comparison. `GET /api/experiments` reads the persisted run
so the page renders without re-running anything; `POST /api/experiments/run`
recomputes it.
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.base import get_db
from app.models import Experiment
from app.schemas import (
    ComputeBudgetOut,
    ExperimentComparisonOut,
    ExperimentOut,
    ExperimentRunOut,
    PlannedExperimentOut,
)
from app.services import experiments as experiments_service
from app.services import registry

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


@router.get("", response_model=ExperimentComparisonOut)
def list_experiments(db: Session = Depends(get_db)) -> dict:
    """Persisted comparison, or an explicit not-yet-run state.

    The three strategies are always reported, even before anything has run, so the
    page can state what is planned and why `full` may be refused on this host.
    """
    budget = experiments_service.detect_budget()
    ready = registry.caches_ready()
    budget_out = ComputeBudgetOut(
        cuda_available=budget.cuda_available,
        total_ram_gb=round(budget.total_ram_bytes / 1e9, 1),
        supports_full_finetune=budget.supports_full_finetune,
    )

    rows = list(db.scalars(select(Experiment).order_by(Experiment.id)))
    if rows:
        return {
            "status": "complete",
            "budget": budget_out.model_dump(),
            "caches_ready": ready,
            "experiments": [ExperimentOut.model_validate(r).model_dump() for r in rows],
            "source": "database",
        }

    payload = experiments_service.persistence_path()
    if payload.is_file():
        with payload.open(encoding="utf-8") as fh:
            return {
                "status": "complete",
                "budget": budget_out.model_dump(),
                "caches_ready": ready,
                "experiments": json.load(fh),
                "source": "disk",
            }

    planned = [
        PlannedExperimentOut(
            name="Zero-shot prompt head (no training)",
            strategy="frozen",
            status="awaiting",
            note="Control condition. No gradient steps.",
        ),
        PlannedExperimentOut(
            name="Frozen linear probe",
            strategy="frozen",
            status="awaiting",
            note="Vision tower frozen; train the head at full scale.",
        ),
        PlannedExperimentOut(
            name="Selective unfreeze (last blocks)",
            strategy="selective",
            status="awaiting",
            note="Unfreeze the last vision blocks on a capped subset.",
        ),
        PlannedExperimentOut(
            name="Full fine-tune (all 878M parameters)",
            strategy="full",
            status="awaiting" if budget.supports_full_finetune else "unavailable",
            note=budget.reason_if_not
            or "Requires ~17.5 GB of optimiser state; runnable on this host.",
        ),
    ]
    return {
        "status": "not_run",
        "budget": budget_out.model_dump(),
        "caches_ready": ready,
        "experiments": [],
        "planned": [p.model_dump() for p in planned],
    }


@router.post("/run", response_model=ExperimentRunOut)
def run_experiments(db: Session = Depends(get_db)) -> dict:
    """Recompute the comparison and persist it.

    Runs the two affordable arms for real and records the third as skipped with
    its reason, so the response never contains a fabricated metric.
    """
    train = registry.load_split("train", registry.AGENT_SLUGS[0])
    test = registry.load_split("test")
    if not (train.real and test.real):
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Embedding caches missing. Run scripts/build_splits.py then "
                "scripts/extract_embeddings.py --split train and --split test."
            ),
        )

    results = experiments_service.run_all(train, test)
    experiments_service.persist(results)

    for res in results:
        metrics = res.metrics
        row = Experiment(
            name=res.name,
            model=registry.load_model_card()["model_id"],
            strategy=res.strategy,
            learning_rate=res.learning_rate,
            epochs=res.epochs,
            accuracy=metrics.get("accuracy"),
            precision=metrics.get("precision"),
            recall=metrics.get("recall"),
            specificity=metrics.get("specificity"),
            f1=metrics.get("macro_f1"),
            auc=metrics.get("malignant_auc"),
            trainable_layers=res.trainable_layers,
            trainable_params=res.trainable_params,
            total_params=res.total_params,
            train_seconds=res.train_seconds,
            status="skipped" if res.status == "skipped" else "completed",
            is_real_result=res.is_real_result,
            notes=res.notes,
        )
        db.add(row)

    db.commit()

    # `partial` rather than `complete` whenever an arm was skipped, so the UI can
    # label the comparison honestly instead of implying all three strategies ran.
    skipped = any(r.status == "skipped" for r in results)
    persisted = list(
        db.scalars(
            select(Experiment).order_by(Experiment.id.desc()).limit(len(results))
        )
    )
    return {
        "status": "partial" if skipped else "complete",
        "experiments": [ExperimentOut.model_validate(r).model_dump() for r in persisted],
    }