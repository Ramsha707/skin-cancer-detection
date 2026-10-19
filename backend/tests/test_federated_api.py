"""Week 7: federated API, background runner and round persistence.

Uses a file-based throwaway SQLite database (not StaticPool) because the round
worker opens its own sessions on its own thread. The embedding caches are
replaced with tiny synthetic separable ones, so a full round -- local training,
FedAvg, checkpoint, audit trail -- runs in well under a second and every metric
asserted is produced by code that actually executed.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
from app import models  # noqa: F401  (registration side effect)
from app.config import settings
from app.database.base import Base, get_db
from app.federated.engine import FederatedEngine, set_engine
from app.federated.runner import RoundRunner, get_runner, set_runner
from app.main import app
from app.models import AuditLog, HospitalAgent, ModelVersion
from app.services import registry
from app.services import training as training_service
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from ml.models import CANCER_CLASSES, EmbeddingCache

AGENTS = ["agent-a", "agent-b", "agent-c", "agent-d"]
DIM = 16


def wait_until(predicate, timeout: float = 30.0, interval: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def _write_cache(directory, name: str, embeddings: np.ndarray, labels: np.ndarray) -> None:
    ids = [f"{name}_{i}" for i in range(len(labels))]
    EmbeddingCache(directory / f"{name}.npy").save(embeddings, labels, ids)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """Throwaway DB + synthetic caches + fresh engine and runner."""
    embed_dir = tmp_path / "embeddings"
    ckpt_dir = tmp_path / "checkpoints"
    embed_dir.mkdir()
    monkeypatch.setattr(registry, "EMBEDDING_DIR", embed_dir)
    monkeypatch.setattr(registry, "CHECKPOINT_DIR", ckpt_dir)
    monkeypatch.setattr(settings, "federated_local_epochs", 25)
    monkeypatch.setattr(settings, "federated_local_patience", 25)

    rng = np.random.default_rng(7)
    centers = rng.normal(size=(len(CANCER_CLASSES), DIM))

    def make(n_per_class: int, tag: str) -> None:
        labels = np.repeat(np.arange(len(CANCER_CLASSES)), n_per_class)
        emb = centers[labels] + rng.normal(scale=0.2, size=(len(labels), DIM))
        _write_cache(embed_dir, tag, emb.astype(np.float32), labels)

    for slug in registry.AGENT_SLUGS:
        make(4, f"train_{slug}")
    make(4, "val")

    db_engine = create_engine(
        f"sqlite:///{tmp_path / 'skinfl-test.db'}",
        connect_args={"check_same_thread": False},
        future=True,
    )
    Base.metadata.create_all(bind=db_engine)
    Session = sessionmaker(bind=db_engine, autoflush=False, future=True)

    with Session() as db:
        for i, agent_id in enumerate(AGENTS, start=1):
            db.add(
                HospitalAgent(
                    agent_id=agent_id,
                    name=f"Hospital {chr(64 + i)}",
                    location=f"City {i}",
                    status="idle",
                    dataset_size=16 * i,
                    class_distribution={"melanoma": 4 * i, "benign_lesion": 12 * i},
                    privacy_status="protected",
                )
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

    # Fresh singleton engine + worker; the worker opens its own sessions on that
    # same throwaway database.
    get_runner().reset()
    set_runner(RoundRunner())
    runner = get_runner()
    runner.session_factory = Session
    set_engine(FederatedEngine(server_lr=settings.fedavg_learning_rate))

    app.dependency_overrides[get_db] = override
    with TestClient(app) as client:
        yield {
            "client": client,
            "Session": Session,
            "runner": runner,
            "embed_dir": embed_dir,
            "ckpt_dir": ckpt_dir,
        }
    app.dependency_overrides.clear()
    runner.reset()
    set_runner(None)
    set_engine(None)


def _rounds(client) -> list[dict]:
    return client.get("/api/federated/rounds").json()


def _wait_for_rounds(client, count: int, timeout: float = 30.0) -> list[dict]:
    """Wait until at least `count` rounds exist *and* all are completed.

    The round row is inserted as `running` when the worker picks it up, so
    counting rows alone would race the completion commit.
    """

    def settled() -> bool:
        rows = _rounds(client)
        return len(rows) >= count and all(r["status"] == "completed" for r in rows)

    ok = wait_until(settled, timeout=timeout)
    rows = _rounds(client)
    assert ok, f"expected {count} completed rounds, got {rows}"
    return rows


class TestFederatedState:
    def test_initial_state_lists_seeded_agents(self, env):
        state = env["client"].get("/api/federated/state").json()
        assert state["status"] == "idle"
        assert state["current_round"] == 0
        assert state["global_model_version"] is None
        assert [a["agent_id"] for a in state["agents"]] == AGENTS
        assert all(a["status"] == "idle" for a in state["agents"])
        assert all(a["phase"] is None for a in state["agents"])


class TestOneRealRound:
    def test_next_round_runs_and_persists_real_metrics(self, env):
        client = env["client"]
        r = client.post("/api/federated/next-round")
        assert r.status_code == 200, r.text
        assert r.json() == {"ok": True, "round": 1}

        rows = _wait_for_rounds(client, 1)
        row = rows[0]
        assert row["status"] == "completed"
        assert row["participating_agents"] == len(AGENTS)
        assert row["global_model_version"] == "v1"
        assert row["samples_processed"] == 16 * len(AGENTS)
        # Separable synthetic data: the aggregated head must actually separate it.
        assert row["global_accuracy"] > 0.8
        for field in ("global_precision", "global_recall", "global_f1"):
            assert row[field] is not None and 0.0 <= row[field] <= 1.0
        assert row["global_loss"] is not None and row["global_loss"] > 0.0
        assert row["duration_seconds"] is not None and row["duration_seconds"] > 0.0

    def test_agents_are_synchronized_after_a_round(self, env):
        client = env["client"]
        client.post("/api/federated/next-round")
        _wait_for_rounds(client, 1)

        state = client.get("/api/federated/state").json()
        assert state["status"] in ("completed", "running")
        assert state["global_model_version"] == "v1"
        assert state["current_round"] == 1
        for agent in state["agents"]:
            assert agent["status"] == "synced"
            assert agent["phase"] == "synced"
            assert agent["weight_delta_norm"] is not None
            assert agent["update_size_kb"] is not None and agent["update_size_kb"] > 0
            assert agent["local_accuracy"] is not None

        agents = client.get("/api/agents").json()
        for agent in agents:
            assert agent["model_version"] == "v1"
            assert agent["status"] == "synced"
            assert agent["rounds_participated"] == 1
            assert agent["last_sync"] is not None

    def test_audit_trail_covers_every_step_with_real_payload_sizes(self, env):
        client = env["client"]
        client.post("/api/federated/next-round")
        _wait_for_rounds(client, 1)

        with env["Session"]() as db:
            entries = list(db.scalars(select(AuditLog).order_by(AuditLog.id)))

        types = [e.event_type for e in entries]
        assert types[0] == "fedavg_started"
        assert types[-1] == "round_completed"
        for event in (
            "local_training_started",
            "local_training_completed",
            "model_update_generated",
            "model_update_transferred",
            "model_broadcast",
            "agent_synchronized",
        ):
            assert types.count(event) == len(AGENTS), event
        assert types.count("fedavg_completed") == 1
        assert types.count("global_model_generated") == 1

        assert all(e.round_number == 1 for e in entries)

        transfers = [e for e in entries if e.event_type == "model_update_transferred"]
        for entry in transfers:
            assert entry.details["payload_type"] == "weight_update"
            assert entry.details["direction"] == "hospital_to_aggregator"
            assert entry.details["payload_kb"] > 0

        broadcasts = [e for e in entries if e.event_type == "model_broadcast"]
        assert all(b.details["direction"] == "aggregator_to_hospital" for b in broadcasts)

        # Structural: nothing in the ledger may reference raw patient imagery.
        for entry in entries:
            blob = f"{entry.message} {entry.details}".lower()
            assert ".jpg" not in blob and ".png" not in blob
            assert "image_path" not in blob

    def test_model_version_and_checkpoint_are_written(self, env):
        client = env["client"]
        client.post("/api/federated/next-round")
        _wait_for_rounds(client, 1)

        with env["Session"]() as db:
            versions = list(db.scalars(select(ModelVersion)))
        assert len(versions) == 1
        assert versions[0].version == "v1"
        assert versions[0].strategy == "frozen"
        assert versions[0].status == "active"
        assert versions[0].is_recommended is True
        assert versions[0].trainable_params == training_service.PARAM_COUNT
        assert (env["ckpt_dir"] / "v1.npz").is_file()

    def test_round_detail_returns_per_agent_metrics(self, env):
        client = env["client"]
        client.post("/api/federated/next-round")
        _wait_for_rounds(client, 1)

        detail = client.get("/api/federated/rounds/1").json()
        assert detail["round"]["round_number"] == 1
        assert [a["agent_id"] for a in detail["agents"]] == AGENTS
        for agent in detail["agents"]:
            assert agent["phase"] == "completed"
            assert agent["samples"] > 0
            assert agent["local_f1"] is not None

    def test_unknown_round_is_404(self, env):
        r = env["client"].get("/api/federated/rounds/99")
        assert r.status_code == 404


class TestMultipleRounds:
    def test_run_queues_two_more_and_archives_previous_version(self, env):
        client = env["client"]
        client.post("/api/federated/next-round")
        _wait_for_rounds(client, 1)

        r = client.post("/api/federated/run", json={"rounds": 2})
        assert r.status_code == 200, r.text
        assert r.json() == {"ok": True, "rounds": 2}

        rows = _wait_for_rounds(client, 3)
        assert [row["round_number"] for row in rows] == [1, 2, 3]
        assert [row["global_model_version"] for row in rows] == ["v1", "v2", "v3"]
        assert all(row["status"] == "completed" for row in rows)

        with env["Session"]() as db:
            versions = {
                v.version: v for v in db.scalars(select(ModelVersion))
            }
        assert versions["v1"].status == "archived"
        assert versions["v2"].status == "archived"
        assert versions["v3"].status == "active"
        assert versions["v3"].is_recommended is True
        assert versions["v1"].is_recommended is False

    def test_resume_after_reset_reuses_the_checkpoint(self, env):
        client = env["client"]
        client.post("/api/federated/next-round")
        _wait_for_rounds(client, 1)

        assert client.post("/api/federated/reset").json() == {"ok": True}

        state = client.get("/api/federated/state").json()
        # Runtime engine state is gone, but history still reports the version.
        assert state["global_model_version"] == "v1"
        assert len(client.get("/api/federated/rounds").json()) == 1

        # A fresh engine bootstraps from the v1 checkpoint, not from zeros.
        set_engine(FederatedEngine())
        engine = training_service.bootstrap_engine()
        assert engine.current_global_version == "v1"
        assert len(engine.agents) == len(AGENTS)

        client.post("/api/federated/next-round")
        rows = _wait_for_rounds(client, 2)
        assert rows[1]["global_model_version"] == "v2"


class TestControlEndpoints:
    def test_pause_when_idle_is_ok(self, env):
        r = env["client"].post("/api/federated/pause")
        assert r.status_code == 200
        assert r.json() == {"ok": True}

    def test_start_is_idempotent_while_running(self, env, monkeypatch):
        client = env["client"]
        runner = env["runner"]
        executed = []
        started = threading.Event()

        def slow_round():
            executed.append(1)
            started.set()
            time.sleep(0.25)

        monkeypatch.setattr(runner, "_execute_round", slow_round)
        assert client.post("/api/federated/start").json() == {
            "ok": True,
            "round": 1,
        }
        assert started.wait(timeout=5)
        # Second start while the worker is running must not double the queue.
        assert client.post("/api/federated/start").json() == {
            "ok": True,
            "round": 1,
        }

        assert wait_until(lambda: runner.status == "completed", timeout=15)
        assert len(executed) == settings.default_rounds

    def test_start_without_caches_is_503_with_instructions(self, env, tmp_path, monkeypatch):
        monkeypatch.setattr(registry, "EMBEDDING_DIR", tmp_path / "empty")
        r = env["client"].post("/api/federated/start")
        assert r.status_code == 503
        assert "extract_embeddings" in r.json()["detail"]
        assert env["client"].get("/api/federated/rounds").json() == []


class TestRoundRunner:
    def test_queue_drains_and_completes(self):
        executed = []
        runner = RoundRunner(execute_round=lambda: executed.append(1))
        runner.request_rounds(3)
        assert wait_until(lambda: runner.status == "completed", timeout=10)
        assert executed == [1, 1, 1]
        assert runner.snapshot()["queued"] == 0

    def test_pause_drops_pending_rounds_at_the_boundary(self):
        executed = []
        gate = threading.Event()

        def slow():
            executed.append(1)
            gate.wait(timeout=5)

        runner = RoundRunner(execute_round=slow)
        runner.request_rounds(10)
        assert wait_until(lambda: len(executed) == 1, timeout=5)
        runner.pause()
        gate.set()
        assert wait_until(lambda: runner.status == "paused", timeout=10)
        assert executed == [1]
        assert runner.snapshot()["queued"] == 0

    def test_failure_sets_error_status_and_clears_the_queue(self):
        def boom():
            raise RuntimeError("caches vanished")

        runner = RoundRunner(execute_round=boom)
        runner.request_rounds(3)
        assert wait_until(lambda: runner.status == "error", timeout=10)
        snap = runner.snapshot()
        assert "caches vanished" in snap["error"]
        assert snap["queued"] == 0

    def test_request_rounds_rejects_zero(self):
        runner = RoundRunner(execute_round=lambda: None)
        with pytest.raises(ValueError, match="at least one"):
            runner.request_rounds(0)

    def test_reset_stops_the_worker_and_clears_live_state(self):
        executed = []
        runner = RoundRunner(execute_round=lambda: executed.append(1))
        runner.note_agent("agent-a", phase="training")
        runner.request_rounds(50)
        assert wait_until(lambda: len(executed) >= 1, timeout=5)
        runner.reset(timeout=10)
        assert runner.status == "idle"
        assert runner.agent_live() == {}
