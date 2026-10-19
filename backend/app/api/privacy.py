"""Week 10: privacy centre endpoint.

One call drives the whole `/privacy` page. Every figure is counted from the
append-only audit ledger at request time, so the page cannot drift from what
actually happened.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.database.base import get_db
from app.schemas import PrivacySummaryOut
from app.services import privacy as privacy_service

router = APIRouter(prefix="/api/privacy", tags=["privacy"])


@router.get("/summary", response_model=PrivacySummaryOut)
def privacy_summary(
    ledger_limit: int = Query(default=50, ge=1, le=500),
    db: Session = Depends(get_db),
) -> dict:
    """Counters, live invariants, per-agent traces and the transmission ledger."""
    return privacy_service.build_summary(db, ledger_limit=ledger_limit)


__all__ = ["router"]
