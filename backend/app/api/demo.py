"""Week 12: the judge-demo checklist endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database.base import get_db
from app.schemas import DemoChecklist
from app.services import demo as demo_service

router = APIRouter(prefix="/api/demo", tags=["demo"])


@router.get("/checklist", response_model=DemoChecklist)
def checklist(db: Session = Depends(get_db)) -> dict:
    """Twelve steps, each with live evidence or the exact command that unblocks it."""
    return demo_service.build_checklist(db)


__all__ = ["router"]
