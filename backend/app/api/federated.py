"""Week 7: federated round control, history and topology."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.base import get_db
from app.models import ModelVersion, TrainingRound
from app.schemas import FederatedStatusOut, ModelVersionOut, TrainingRoundOut
from app.services import registry
from app.services import training as training_service

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/federated", tags=["federated"])


@router.get("/status", response_model=FederatedStatusOut)
def federated_status() -> dict:
    """What the federated engine can do right now.

    Mirrors `detection/status`: the dashboard shows the run panel only when the
    caches exist, and shows the extraction commands otherwise.
    """
    ready = registry.caches_ready()
    train_ready = all(ready.get(f"train_{a}") for a in registry.AGENT_SLUGS)

    instructions = None
    if not train_ready:
        instructions = (
            "Run .\\backend-venv\\Scripts\\python.exe scripts/build_splits.py, then "
            ".\\.venv-ov\\Scripts\\python.exe scripts/extract_embeddings.py --split train "
            "(~3.4 h on the iGPU at the measured 0.57 img/s)."
        )

    return {
        "ready": train_ready,
        "caches_ready": ready,
        "param_count": training_service.PARAM_COUNT,
        "param_kb": round(training_service.param_kb(), 3),
        "aggregation": "FedAvg (sample-count weighted mean)",
        "instructions": instructions,
    }


@router.post("/rounds", response_model=TrainingRoundOut)
def start_round(db: Session = Depends(get_db)) -> TrainingRound:
    """Run one federated round synchronously.

    A round here trains a 4,612-parameter head on cached embeddings, so it
    completes in seconds. Real multi-hour rounds would move to a background task;
    at this head size, doing it inline keeps the audit trail in one transaction.
    """
    try:
        outcome = training_service.run_round(db)
    except RuntimeError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

    training_service.publish_checkpoint(db, _engine())
    db.commit()

    row = db.scalar(
        select(TrainingRound).where(TrainingRound.round_number == outcome.round_result.round_number)
    )
    return row


@router.get("/rounds", response_model=list[TrainingRoundOut])
def list_rounds(db: Session = Depends(get_db)) -> list[TrainingRound]:
    return list(
        db.scalars(select(TrainingRound).order_by(TrainingRound.round_number.desc()).limit(50))
    )


@router.get("/rounds/{round_number}", response_model=TrainingRoundOut)
def get_round(round_number: int, db: Session = Depends(get_db)) -> TrainingRound:
    row = db.scalar(select(TrainingRound).where(TrainingRound.round_number == round_number))
    if row is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, detail=f"training round {round_number} not found"
        )
    return row


@router.get("/models", response_model=list[ModelVersionOut])
def list_models(db: Session = Depends(get_db)) -> list[ModelVersion]:
    return list(
        db.scalars(select(ModelVersion).order_by(ModelVersion.training_round.desc().nullslast()))
    )


def _engine():
    from app.federated.engine import get_engine

    return get_engine()