"""Week 12: the judge-demo checklist.

Two states are tested, because the honesty contract matters more than the happy
path: a cold checkout must return twelve blocked-or-ready steps with instructions,
and a prepared environment must flip every required step to ready from live rows.
Nothing here stubs the checklist itself; only the artefact probes are pinned.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from app import models  # noqa: F401  (registration side effect)
from app.database.base import Base, get_db
from app.main import app
from app.models import AuditLog, HospitalAgent, ModelVersion, TrainingRound
from app.services import dataset as dataset_service
from app.services import explainability as explainability_service
from app.services import registry
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

STEP_KEYS = [
    "api",
    "database",
    "dataset",
    "agents",
    "embeddings",
    "federated_round",
    "model_registry",
    "evaluation",
    "detection",
    "explainability",
    "privacy",
    "audit",
]


@pytest.fixture()
def env():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autoflush=False, future=True)

    with Session() as db:
        for agent_id, name in [
            ("agent-a", "Hospital A"),
            ("agent-b", "Hospital B"),
            ("agent-c", "Hospital C"),
            ("agent-d", "Hospital D"),
        ]:
            db.add(
                HospitalAgent(
                    agent_id=agent_id,
                    name=name,
                    status="idle",
                    dataset_size=100,
                    privacy_status="protected",
                )
            )
        db.commit()

    def override():
        db = Session()
        try:
            yield db
            db.commit()
        finally:
            db.close()

    app.dependency_overrides[get_db] = override
    with TestClient(app) as client:
        yield client, Session
    app.dependency_overrides.clear()


def _blocked_artefacts(monkeypatch):
    monkeypatch.setattr(registry, "latest_checkpoint", lambda: None)
    monkeypatch.setattr(
        registry,
        "caches_ready",
        lambda: {
            "val": False,
            **{f"train_{agent}": False for agent in registry.AGENT_SLUGS},
        },
    )
    monkeypatch.setattr(
        dataset_service,
        "load_manifest",
        lambda: dataset_service.DatasetManifest(message="No dataset configured."),
    )


def test_cold_checkout_reports_twelve_blocked_steps_with_commands(env, monkeypatch):
    client, _ = env
    _blocked_artefacts(monkeypatch)

    res = client.get("/api/demo/checklist")
    assert res.status_code == 200
    body = res.json()

    assert body["total"] == 12
    assert [step["key"] for step in body["steps"]] == STEP_KEYS
    assert [step["number"] for step in body["steps"]] == list(range(1, 13))
    assert all(step["evidence"] for step in body["steps"])
    assert body["completed"] == sum(1 for step in body["steps"] if step["ready"])
    assert body["ready"] == all(
        step["ready"] for step in body["steps"] if step["required"]
    )
    assert body["ready"] is False

    by_key = {step["key"]: step for step in body["steps"]}
    assert by_key["agents"]["ready"] is True
    assert "4 of 4 agents" in by_key["agents"]["evidence"]
    assert by_key["privacy"]["ready"] is True
    assert by_key["dataset"]["command"]
    assert by_key["embeddings"]["command"]
    assert by_key["federated_round"]["ready"] is False
    assert by_key["federated_round"]["command"]
    assert by_key["model_registry"]["ready"] is False
    assert by_key["audit"]["ready"] is False
    assert body["disclaimer"].startswith("Research prototype")


def test_prepared_environment_flips_every_required_step_ready(env, monkeypatch):
    client, Session = env

    monkeypatch.setattr(
        registry,
        "caches_ready",
        lambda: {
            "val": True,
            **{f"train_{agent}": True for agent in registry.AGENT_SLUGS},
        },
    )
    monkeypatch.setattr(registry, "latest_checkpoint", lambda: Path("global_v3.pt"))
    monkeypatch.setattr(
        dataset_service,
        "load_manifest",
        lambda: dataset_service.DatasetManifest(
            loaded=True,
            patients=20,
            records=[object()] * 100,
            message="loaded",
        ),
    )
    monkeypatch.setattr(
        explainability_service,
        "availability",
        lambda: SimpleNamespace(available=True, reason="", device="CPU"),
    )

    with Session() as db:
        db.add(
            TrainingRound(
                round_number=1,
                status="completed",
                participating_agents=4,
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
            )
        )
        db.add(
            ModelVersion(
                version="global_v3",
                strategy="fedavg",
                training_round=1,
                status="active",
            )
        )
        db.add(
            AuditLog(
                event_type="round_completed",
                message="Round 1 completed.",
                status="success",
                round_number=1,
                details={},
            )
        )
        db.commit()

    res = client.get("/api/demo/checklist")
    assert res.status_code == 200
    body = res.json()

    assert body["ready"] is True
    assert body["completed"] == 12
    assert all(step["ready"] for step in body["steps"])
    assert all(step["command"] is None for step in body["steps"])

    by_key = {step["key"]: step for step in body["steps"]}
    assert "100 images" in by_key["dataset"]["evidence"]
    assert "1 completed" in by_key["federated_round"]["evidence"]
    assert "global_v3" in by_key["model_registry"]["evidence"]
    assert "0 raw images shared" in by_key["privacy"]["evidence"]
    assert "1 ledger row" in by_key["audit"]["evidence"]
