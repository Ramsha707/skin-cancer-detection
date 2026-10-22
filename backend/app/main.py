"""FastAPI application entry point."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import api_router
from app.config import settings
from app.database.base import SessionLocal, init_db
from app.database.seed import seed_all
from app.websocket import TYPE_SYSTEM, get_event_bus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s",
)
log = logging.getLogger("skinfl")


@asynccontextmanager
async def lifespan(app: FastAPI):
    bus = get_event_bus()
    bus.attach_loop(asyncio.get_running_loop())
    init_db()
    db = SessionLocal()
    try:
        seed_all(db)
        log.info("%s v%s ready", settings.app_name, settings.version)
    except Exception:
        log.exception("seeding failed; continuing with an empty database")
        db.rollback()
    finally:
        db.close()

    bus.publish(
        TYPE_SYSTEM,
        {"status": "online", "version": settings.version},
        source="api",
    )
    log.info("live event bus attached: %s", bus.stats())
    try:
        yield
    finally:
        bus.publish(
            TYPE_SYSTEM,
            {"status": "offline", "version": settings.version},
            source="api",
        )
        bus.detach_loop()


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description=(
        "Privacy-preserving multi-class skin cancer detection across federated "
        "hospital agents. Research prototype - not a medical device."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/", tags=["system"])
def root() -> dict:
    return {
        "name": settings.app_name,
        "version": settings.version,
        "docs": "/docs",
        "api": "/api",
        "disclaimer": (
            "Research prototype. Not a medical device. Must not be used for "
            "clinical decision making."
        ),
    }


@app.get("/health", tags=["system"])
def health() -> dict:
    return {"status": "ok"}
