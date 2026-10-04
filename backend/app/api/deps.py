"""Shared FastAPI dependencies."""

from __future__ import annotations

from collections.abc import Generator

from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database.base import get_db

DbSession = Generator[Session, None, None]


def db_session() -> Generator[Session, None, None]:
    yield from get_db()


DbDep = Depends(db_session)


def get_or_404(db: Session, model: type, pk: int, label: str) -> object:
    """Fetch a row by primary key or raise a 404 with a useful message."""
    row = db.get(model, pk)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{label} {pk} not found",
        )
    return row