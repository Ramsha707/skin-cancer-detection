"""Week 9: model evaluation over the held-out validation split.

One endpoint drives the whole `/performance` page. It doubles as a readiness
check: `ready=False` plus `instructions` is the honest answer when no checkpoint
or no validation embeddings exist, and the UI shows that instead of a chart.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database.base import get_db
from app.schemas import EvaluationSummary
from app.services import evaluation as evaluation_service

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])


@router.get("/summary", response_model=EvaluationSummary)
def evaluation_summary(db: Session = Depends(get_db)) -> dict:
    """Metrics, per-class breakdown, confusion matrix, ROC and round history.

    Every number comes from running the aggregated head over the validation
    embeddings that exist on disk at request time. Missing artefacts produce an
    explicit not-ready payload, never a placeholder score.
    """
    return evaluation_service.build_summary(db)


__all__ = ["router"]
