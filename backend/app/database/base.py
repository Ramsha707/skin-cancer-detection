"""SQLAlchemy engine, session factory and declarative base."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings

# SQLite needs check_same_thread disabled for FastAPI's threadpool,
# and WAL so the training thread can write while requests read.
connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}

engine = create_engine(
    settings.database_url,
    connect_args=connect_args,
    pool_pre_ping=True,
    future=True,
)


@event.listens_for(engine, "connect")
def _sqlite_pragmas(dbapi_connection, _record) -> None:  # pragma: no cover - driver hook
    if not settings.database_url.startswith("sqlite"):
        return
    cur = dbapi_connection.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA synchronous=NORMAL")
    cur.execute("PRAGMA foreign_keys=ON")
    cur.close()


SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    """Declarative base for every ORM model."""


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency. Commits on success, rolls back on error."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db() -> None:
    """Create tables if they do not exist.

    Imports model modules first so their metadata is registered on `Base`.
    """
    from app import models  # noqa: F401  (registration side effect)

    Path(engine.url.database).parent.mkdir(
        parents=True, exist_ok=True
    ) if engine.url.database else None
    Base.metadata.create_all(bind=engine)
