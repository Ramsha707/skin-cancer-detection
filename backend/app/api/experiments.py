"""Week 6: experiment endpoints for the Experiments page.

`POST /api/experiments/run` executes only strategies the host can honestly
run (see `app.services.experiments.feasibility`); everything else returns
409 with the reason, so the page's Run button and the backend gate always
agree. Metrics are never written here - only the trainer's real evaluation
produces them.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.base import get_db
from app.models import Experiment
from app.schemas import ExperimentOut, RunExperimentRequest
from app.services import experiments as experiments_service

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


def _payload(row: Experiment) -> ExperimentOut:
    return ExperimentOut(**experiments_service.experiment_payload(row))


def _get_row(db: Session, name: str) -> Experiment:
    row = db.scalar(select(Experiment).where(Experiment.name == name))
    if row is None:
        raise HTTPException(status_code=404, detail=f'Experiment "{name}" not found')
    return row


@router.get("", response_model=list[ExperimentOut])
def list_experiments(db: Session = Depends(get_db)) -> list[ExperimentOut]:
    """All registered experiments, seeded strategies first (id order)."""
    rows = db.scalars(select(Experiment).order_by(Experiment.id)).all()
    return [_payload(row) for row in rows]


@router.post("/run", response_model=ExperimentOut)
def run_experiment(
    body: RunExperimentRequest, db: Session = Depends(get_db)
) -> ExperimentOut:
    """Execute an experiment and return it with freshly written real metrics.

    404 unknown strategy name, 409 when the strategy is infeasible on this
    host or the embedding caches it needs are missing.
    """
    row = _get_row(db, body.name)
    try:
        row = experiments_service.run_experiment(db, row.name)
    except experiments_service.ExperimentNotRunnable as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except KeyError as exc:  # pragma: no cover - name re-checked above
        raise HTTPException(status_code=404, detail=f"Experiment {exc} not found") from exc
    return _payload(row)


@router.delete("/{experiment_id}")
def delete_experiment(experiment_id: int, db: Session = Depends(get_db)) -> dict:
    row = db.get(Experiment, experiment_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
    db.delete(row)
    db.flush()
    return {"ok": True}
