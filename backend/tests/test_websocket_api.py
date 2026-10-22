"""Week 11: live event bus, WebSocket stream and integration diagnostics.

The WebSocket test drives the real endpoint: connect, read the hello frame,
publish through the bus from the test thread, and assert the exact event arrives.
That is the same cross-thread hop the round worker uses in production, so a
regression in the loop hand-off fails here rather than in the browser.
"""

from __future__ import annotations

import time

import pytest
from app import models  # noqa: F401  (registration side effect)
from app.database.base import Base, get_db
from app.federated.runner import RoundRunner
from app.main import app
from app.models import HospitalAgent
from app.websocket import (
    TYPE_AGENT,
    TYPE_AUDIT,
    TYPE_TRAINING,
    EventBus,
    get_event_bus,
    set_event_bus,
)
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


@pytest.fixture()
def bus():
    """A fresh bus per test; restored afterwards so tests stay independent."""
    previous = get_event_bus()
    fresh = EventBus()
    set_event_bus(fresh)
    try:
        yield fresh
    finally:
        set_event_bus(previous)


@pytest.fixture()
def client(bus):
    """Depend on `bus` so the isolated event bus is installed before app import."""
    assert bus.subscriber_count == 0
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
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def wait_until(predicate, timeout: float = 10.0, interval: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def test_system_events_returns_recent_history(client, bus):
    bus.publish(TYPE_AUDIT, {"event_type": "detection_performed"}, source="test")

    res = client.get("/api/system/events", params={"limit": 10})
    assert res.status_code == 200
    body = res.json()
    assert body["stats"]["events_seen"] >= 1
    assert any(e["type"] == TYPE_AUDIT for e in body["events"])
    assert all("timestamp" in e and "payload" in e for e in body["events"])


def test_websocket_hello_seeds_history_and_streams_new_events(client, bus):
    bus.publish(TYPE_AUDIT, {"event_type": "detection_performed"}, source="seed")

    with client.websocket_connect("/ws/training") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello"
        assert hello["payload"]["subscribers"] == 1
        assert any(e["type"] == TYPE_AUDIT for e in hello["payload"]["history"])

        bus.publish(
            TYPE_AUDIT,
            {"event_type": "model_broadcast", "message": "to all hospitals"},
            source="test",
        )
        event = ws.receive_json()

    assert event["type"] == TYPE_AUDIT
    assert event["payload"]["event_type"] == "model_broadcast"
    assert event["source"] == "test"
    assert event["id"] > 0
    assert bus.subscriber_count == 0


def test_runner_publishes_training_and_agent_events(bus):
    runner = RoundRunner(execute_round=lambda: None)
    depth = runner.request_rounds(1)
    assert depth == 1

    assert wait_until(lambda: runner.status == "completed")
    runner.note_agent("agent-a", status="synced", phase="synced", samples=16)

    events = bus.recent(50)
    training_events = [e for e in events if e["type"] == TYPE_TRAINING]
    agent_events = [e for e in events if e["type"] == TYPE_AGENT]
    assert any(e["payload"].get("event") == "rounds_requested" for e in training_events)
    assert any(e["payload"].get("event") == "queue_finished" for e in training_events)
    assert any(e["payload"]["agent_id"] == "agent-a" for e in agent_events)


def test_integration_reports_every_subsystem(client):
    res = client.get("/api/system/integration")
    assert res.status_code == 200
    body = res.json()

    keys = {m["key"] for m in body["modules"]}
    assert {
        "api",
        "database",
        "dataset",
        "agents",
        "embeddings",
        "federated",
        "models",
        "evaluation",
        "explainability",
        "privacy",
        "live_events",
    } <= keys
    assert body["counts"]["agents"] == 4
    assert body["websocket"]["events_seen"] >= 1
    assert isinstance(body["ready"], bool)
    for module in body["modules"]:
        assert module["detail"]
