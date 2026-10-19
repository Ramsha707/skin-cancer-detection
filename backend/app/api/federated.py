"""Week 7: federated round control, live state and history.

Rounds execute on a background worker (see `app.federated.runner`); these
endpoints queue work, read the audit-backed history and report live state.

There is no simulation mode: a start with no real embedding caches is refused
with a 503 carrying the command that would fix it, so every round row in the
database was produced by training that actually ran.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.database.base import get_db
from app.federated.engine import get_engine
from app.federated.runner import STATUS_RUNNING, get_runner
from app.models import HospitalAgent, TrainingRound
from app.schemas import (
    AgentRoundState,
    FederatedState,
    OkResponse,
    RoundDetail,
    RunRoundsRequest,
    RunRoundsResponse,
    StartResponse,
    TrainingRoundOut,
)
from app.services import training as training_service

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/federated", tags=["federated"])


def _require_ready() -> None:
    reason = training_service.readiness()
    if reason is not None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=reason)


def _next_round_number(db: Session) -> int:
    latest = db.scalar(select(TrainingRound).order_by(TrainingRound.round_number.desc()))
    return (latest.round_number + 1) if latest else 1


def _agent_states(db: Session, runner) -> list[AgentRoundState]:
    """Merge seeded agent rows, live round progress and the latest round's
    persisted metrics into one panel per hospital."""
    live = runner.agent_live()
    rows = db.scalars(select(HospitalAgent).order_by(HospitalAgent.id)).all()

    latest = db.scalar(select(TrainingRound).order_by(TrainingRound.round_number.desc()))
    persisted: dict[str, dict] = {}
    if latest is not None:
        for entry in latest.agent_results or []:
            if isinstance(entry, dict) and entry.get("agent_id"):
                persisted[entry["agent_id"]] = entry

    states = []
    for row in rows:
        info = live.get(row.agent_id, {})
        entry = persisted.get(row.agent_id, {})
        states.append(
            AgentRoundState(
                agent_id=row.agent_id,
                agent_name=row.name,
                status=info.get("status", row.status),
                samples=int(info.get("samples", row.dataset_size)),
                local_loss=info.get("local_loss", entry.get("loss")),
                local_accuracy=info.get("local_accuracy", entry.get("local_accuracy")),
                local_recall=info.get("local_recall", entry.get("local_recall")),
                local_f1=info.get("local_f1", entry.get("local_f1")),
                weight_delta_norm=info.get(
                    "weight_delta_norm", entry.get("weight_delta_norm")
                ),
                update_size_kb=info.get("update_size_kb", entry.get("update_size_kb")),
                phase=info.get("phase"),
            )
        )
    return states


@router.get("/rounds", response_model=list[TrainingRoundOut])
def list_rounds(db: Session = Depends(get_db)) -> list[TrainingRound]:
    return list(db.scalars(select(TrainingRound).order_by(TrainingRound.round_number)))


@router.get("/state", response_model=FederatedState)
def federated_state(db: Session = Depends(get_db)) -> FederatedState:
    runner = get_runner()
    snap = runner.snapshot()
    latest = db.scalar(select(TrainingRound).order_by(TrainingRound.round_number.desc()))
    version = get_engine().current_global_version
    if version is None and latest is not None:
        version = latest.global_model_version
    return FederatedState(
        status=snap["status"],
        current_round=latest.round_number if latest else 0,
        global_model_version=version,
        aggregation_status=snap["aggregation_status"],
        training_status=snap["status"],
        agents=_agent_states(db, runner),
    )


@router.post("/start", response_model=StartResponse)
def start_training(db: Session = Depends(get_db)) -> StartResponse:
    """Queue `settings.default_rounds` rounds. Idempotent while already
    running; a paused run is resumed by queueing a fresh batch."""
    _require_ready()
    runner = get_runner()
    next_round = _next_round_number(db)
    if runner.status != STATUS_RUNNING:
        runner.request_rounds(settings.default_rounds)
    return StartResponse(round=next_round)


@router.post("/pause", response_model=OkResponse)
def pause_training() -> OkResponse:
    """Drop queued rounds. The round in flight finishes at the next boundary."""
    get_runner().pause()
    return OkResponse()


@router.post("/next-round", response_model=StartResponse)
def next_round(db: Session = Depends(get_db)) -> StartResponse:
    """Queue exactly one more round."""
    _require_ready()
    runner = get_runner()
    next_number = _next_round_number(db)
    runner.request_rounds(1)
    return StartResponse(round=next_number)


@router.post("/run", response_model=RunRoundsResponse)
def run_rounds(body: RunRoundsRequest) -> RunRoundsResponse:
    """Queue an explicit number of rounds."""
    _require_ready()
    get_runner().request_rounds(body.rounds)
    return RunRoundsResponse(rounds=body.rounds)


@router.post("/reset", response_model=OkResponse)
def reset_federated() -> OkResponse:
    """Stop the worker and drop in-memory engine state.

    Round history and the audit ledger are kept: they describe what actually
    happened and are never rewritten.
    """
    get_runner().reset()
    return OkResponse()


@router.get("/rounds/{round_number}", response_model=RoundDetail)
def get_round(round_number: int, db: Session = Depends(get_db)) -> RoundDetail:
    row = db.scalar(select(TrainingRound).where(TrainingRound.round_number == round_number))
    if row is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, detail=f"training round {round_number} not found"
        )
    agents = [
        AgentRoundState(
            agent_id=entry.get("agent_id", ""),
            agent_name=entry.get("agent_name", entry.get("agent_id", "")),
            status="synced",
            samples=int(entry.get("samples", 0)),
            local_loss=entry.get("loss"),
            local_accuracy=entry.get("local_accuracy"),
            local_recall=entry.get("local_recall"),
            local_f1=entry.get("local_f1"),
            weight_delta_norm=entry.get("weight_delta_norm"),
            update_size_kb=entry.get("update_size_kb"),
            phase="completed",
        )
        for entry in row.agent_results or []
        if isinstance(entry, dict)
    ]
    return RoundDetail(round=row, agents=agents)


__all__ = ["router"]
