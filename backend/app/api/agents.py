"""Hospital agent endpoints.

In this build the four agents are simulated *within* one process, but the API
surface is the same one a real deployment would expose, where each agent would
sit behind its own host and speak only over the three tensor-only calls
documented in `app/federated/engine.py`.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.base import get_db
from app.models import HospitalAgent
from app.schemas import HospitalAgentOut
from app.services import audit as audit_service

router = APIRouter(prefix="/api/agents", tags=["agents"])


@router.get("", response_model=list[HospitalAgentOut])
def list_agents(db: Session = Depends(get_db)) -> list[HospitalAgent]:
    return list(db.scalars(select(HospitalAgent).order_by(HospitalAgent.id)))


@router.get("/{agent_id}", response_model=HospitalAgentOut)
def get_agent(agent_id: int, db: Session = Depends(get_db)) -> HospitalAgent:
    agent = db.get(HospitalAgent, agent_id)
    if agent is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, detail=f"Hospital agent {agent_id} not found"
        )
    return agent


@router.get("/{agent_id}/class-distribution")
def class_distribution(agent_id: int, db: Session = Depends(get_db)) -> dict[str, int]:
    agent = _require(db, agent_id)
    return agent.class_distribution or {}


@router.post("/{agent_id}/train", response_model=HospitalAgentOut)
def start_training(agent_id: int, db: Session = Depends(get_db)) -> HospitalAgent:
    agent = _require(db, agent_id)
    if agent.status == "training":
        return agent

    agent.status = "training"
    audit_service.log(
        db,
        event_type=audit_service.EVENT_LOCAL_TRAINING_STARTED,
        message=f"{agent.name} began local training on its private partition.",
        agent_id=agent.agent_id,
        round_number=agent.rounds_participated + 1,
        details={"samples": agent.dataset_size},
    )
    db.flush()
    return agent


@router.post("/{agent_id}/pause", response_model=HospitalAgentOut)
def pause_training(agent_id: int, db: Session = Depends(get_db)) -> HospitalAgent:
    agent = _require(db, agent_id)
    if agent.status == "training":
        agent.status = "paused"
    return agent


@router.post("/{agent_id}/sync", response_model=HospitalAgentOut)
def sync_agent(agent_id: int, db: Session = Depends(get_db)) -> HospitalAgent:
    """Pull the latest global weights.

    Logged as a *download* to the hospital, so the audit ledger shows both
    directions of every transmission.
    """
    from app.models import utcnow

    agent = _require(db, agent_id)
    # The engine is registered in Week 7; guard so Week 2 runs standalone.
    try:
        from app.federated.engine import get_engine

        version = get_engine().current_global_version or "none"
    except (ImportError, AttributeError):
        version = "unassigned"
    agent.model_version = version
    agent.last_sync = utcnow()
    agent.status = "synced"

    audit_service.log(
        db,
        event_type=audit_service.EVENT_AGENT_SYNCHRONIZED,
        message=f"{agent.name} received global model {version}.",
        agent_id=agent.agent_id,
        details={"direction": "aggregator_to_hospital", "model_version": version},
    )
    db.flush()
    return agent


def _require(db: Session, agent_id: int) -> HospitalAgent:
    agent = db.get(HospitalAgent, agent_id)
    if agent is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, detail=f"Hospital agent {agent_id} not found"
        )
    return agent
