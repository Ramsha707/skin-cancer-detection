"""Week 10: privacy centre.

Every counter and invariant is derived from the audit ledger, so these tests drive
the ledger directly with the real `audit_service` helpers rather than inserting
synthetic rows -- if the derivation and the writer disagree, the test fails.
"""

from __future__ import annotations

import pytest
from app import models  # noqa: F401  (registration side effect)
from app.database.base import Base, get_db
from app.main import app
from app.models import HospitalAgent
from app.services import audit as audit_service
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


@pytest.fixture()
def client():
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
        # A weight update travelling hospital -> aggregator.
        audit_service.transmit(
            db,
            event_type=audit_service.EVENT_MODEL_UPDATE_TRANSFERRED,
            message="Hospital A sent a weight update.",
            agent_id="agent-a",
            payload_kb=18.0,
            round_number=1,
        )
        # A broadcast travelling aggregator -> every hospital.
        for agent_id in ("agent-a", "agent-b", "agent-c"):
            audit_service.transmit(
                db,
                event_type=audit_service.EVENT_MODEL_BROADCAST,
                message=f"Global model broadcast to {agent_id}.",
                agent_id=agent_id,
                payload_kb=18.0,
                round_number=1,
            )
        # A local detection: must carry no identifying detail.
        audit_service.log(
            db,
            event_type=audit_service.EVENT_DETECTION_PERFORMED,
            message="Classified lesion as benign_lesion.",
            details={"model_version": "v1", "is_real_inference": True},
        )
        db.commit()

    def override():
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


class TestCounters:
    def test_counts_transmissions_from_the_ledger(self, client):
        body = client.get("/api/privacy/summary").json()
        counters = body["counters"]
        assert counters["transmissions_logged"] == 4
        assert counters["broadcasts_to_hospitals"] == 3
        assert counters["model_updates_shared"] is True
        # One upload of 18 KB; broadcasts are counted as transmissions but their
        # payload size is tracked separately from the update total.
        assert counters["total_update_size_kb"] == pytest.approx(18.0)

    def test_raw_images_shared_is_structurally_zero(self, client):
        counters = client.get("/api/privacy/summary").json()["counters"]
        assert counters["raw_images_shared"] == 0

    def test_detections_are_counted_separately(self, client):
        counters = client.get("/api/privacy/summary").json()["counters"]
        assert counters["detections_performed"] == 1
        assert counters["explainability_generated"] == 0

    def test_agents_protected_matches_the_table(self, client):
        counters = client.get("/api/privacy/summary").json()["counters"]
        assert counters["total_agents"] == 3
        assert counters["agents_protected"] == 3


class TestInvariants:
    def test_all_invariants_hold_on_a_clean_ledger(self, client):
        invariants = client.get("/api/privacy/summary").json()["invariants"]
        assert invariants
        assert all(item["enforced"] for item in invariants)
        assert {item["key"] for item in invariants} >= {
            "no_image_payloads",
            "images_stay_local",
            "tensor_only_channel",
        }

    def test_every_transmission_is_a_weight_update(self, client):
        body = client.get("/api/privacy/summary").json()
        for entry in body["ledger"]:
            assert entry["payload_type"] == "weight_update"
            assert entry["payload_kb"] is not None


class TestAgentTrace:
    def test_per_agent_direction_split(self, client):
        agents = {a["agent_id"]: a for a in client.get("/api/privacy/summary").json()["agents"]}
        assert agents["agent-a"]["uploads"] == 1
        assert agents["agent-a"]["downloads"] == 1
        assert agents["agent-b"]["uploads"] == 0
        assert agents["agent-b"]["downloads"] == 1
        assert agents["agent-a"]["upload_kb"] == pytest.approx(18.0)

    def test_last_activity_is_recorded(self, client):
        agents = {a["agent_id"]: a for a in client.get("/api/privacy/summary").json()["agents"]}
        assert agents["agent-a"]["last_activity"] is not None
        assert agents["agent-c"]["last_activity"] is not None


class TestLedger:
    def test_ledger_only_contains_transmission_events(self, client):
        ledger = client.get("/api/privacy/summary").json()["ledger"]
        assert len(ledger) == 4
        assert {e["event_type"] for e in ledger} == {
            "model_update_transferred",
            "model_broadcast",
        }

    def test_ledger_entries_carry_direction_and_label(self, client):
        ledger = client.get("/api/privacy/summary").json()["ledger"]
        directions = {e["direction"] for e in ledger}
        assert directions == {"hospital_to_aggregator", "aggregator_to_hospital"}
        assert all(e["label"] for e in ledger)

    def test_ledger_limit_is_respected(self, client):
        ledger = client.get("/api/privacy/summary", params={"ledger_limit": 2}).json()["ledger"]
        assert len(ledger) == 2
