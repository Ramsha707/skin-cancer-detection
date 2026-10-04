"""Audit ledger endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.base import get_db
from app.models import AuditLog
from app.schemas import AuditLogOut

router = APIRouter(prefix="/api/audit", tags=["audit"])


@router.get("", response_model=list[AuditLogOut])
def list_logs(
    limit: int = Query(default=100, ge=1, le=1000),
    event_type: str | None = Query(default=None),
    round_number: int | None = Query(default=None),
    db: Session = Depends(get_db),
) -> list[AuditLog]:
    stmt = select(AuditLog).order_by(AuditLog.timestamp.desc(), AuditLog.id.desc()).limit(limit)
    if event_type:
        stmt = stmt.where(AuditLog.event_type == event_type)
    if round_number is not None:
        stmt = stmt.where(AuditLog.round_number == round_number)
    return list(db.scalars(stmt))
