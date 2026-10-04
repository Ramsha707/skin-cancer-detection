from __future__ import annotations

import pytest
from app.database.base import Base, get_db
from app.main import app
from app.models import HospitalAgent
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


@pytest.fixture()
def client():
    """Agent API tests run against a throwaway in-memory database.

    The real `skinfl.db` is seeded from HAM10000 at import time, which is slow and
    would make these tests depend on the dataset being present.
    """
    # StaticPool pins every session to one connection; without it each checkout
    # of `sqlite://` opens its own empty database.
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    with Session() as db:
        for i, (agent_id, name) in enumerate(
            [("agent-a", "Hospital A"), ("agent-b", "Hospital B")], start=1
        ):
            db.add(
                HospitalAgent(
                    agent_id=agent_id,
                    name=name,
                    location=f"City {i}",
                    status="idle",
                    dataset_size=100 * i,
                    class_distribution={"melanoma": 10 * i, "benign_lesion": 90 * i},
                    privacy_status="protected",
                )
            )
        db.commit()

    def override():
        """Mirrors app.database.base.get_db, which commits on success."""
        db = Session()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    app.dependency_overrides[get_db] = override
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


class TestListAgents:
    def test_returns_seeded_agents_in_id_order(self, client):
        r = client.get("/api/agents")
        assert r.status_code == 200
        assert [a["agent_id"] for a in r.json()] == ["agent-a", "agent-b"]

    def test_exposes_class_distribution_on_the_roster(self, client):
        """The Agents page renders class mix straight off the list response."""
        assert client.get("/api/agents").json()[0]["class_distribution"]["melanoma"] == 10


class TestGetAgent:
    def test_returns_single_agent(self, client):
        assert client.get("/api/agents/1").json()["name"] == "Hospital A"

    def test_unknown_id_is_404(self, client):
        r = client.get("/api/agents/99")
        assert r.status_code == 404
        assert "not found" in r.json()["detail"].lower()

    def test_class_distribution_matches_roster(self, client):
        dist = client.get("/api/agents/1/class-distribution").json()
        assert dist == {"melanoma": 10, "benign_lesion": 90}


class TestAgentControls:
    def test_train_then_pause_round_trips_status(self, client):
        assert client.post("/api/agents/1/train").json()["status"] == "training"
        assert client.post("/api/agents/1/pause").json()["status"] == "paused"

    def test_pause_is_a_no_op_when_not_training(self, client):
        """Pausing an idle agent must not silently move it to 'paused'."""
        assert client.post("/api/agents/1/pause").json()["status"] == "idle"

    def test_train_is_idempotent(self, client):
        client.post("/api/agents/1/train")
        assert client.post("/api/agents/1/train").json()["status"] == "training"

    def test_sync_sets_status_and_timestamp(self, client):
        r = client.post("/api/agents/1/sync").json()
        assert r["status"] == "synced"
        assert r["last_sync"] is not None
        assert r["model_version"] == "unassigned"

    def test_controls_404_on_unknown_agent(self, client):
        for action in ("train", "pause", "sync"):
            assert client.post(f"/api/agents/99/{action}").status_code == 404


class TestAgentAuditTrail:
    def test_train_and_sync_are_audited(self, client):
        client.post("/api/agents/1/train")
        client.post("/api/agents/1/sync")
        events = [e["event_type"] for e in client.get("/api/audit").json()]
        assert "local_training_started" in events
        assert "agent_synchronized" in events

    def test_sync_logs_the_direction_of_transmission(self, client):
        """Privacy claims depend on the ledger recording which way data moved."""
        client.post("/api/agents/1/sync")
        entry = next(
            e for e in client.get("/api/audit").json() if e["event_type"] == "agent_synchronized"
        )
        assert entry["details"]["direction"] == "aggregator_to_hospital"