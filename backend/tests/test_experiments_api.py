"""Week 6: experiments API and trainer tests.

Covers the three seeded strategies, the feasibility gate (409s), the
delete/list contract, the numpy head trainer, and a full synthetic end-to-end
run that proves metrics are only ever written from a real evaluation.
"""

from __future__ import annotations

import numpy as np
import pytest
from app import models  # noqa: F401  (registration side effect)
from app.database.base import Base, get_db
from app.database.seed import EXPERIMENT_DEFINITIONS, seed_experiments
from app.main import app
from app.services import experiments as experiments_service
from app.services import registry
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from ml.models import CANCER_CLASSES, MALIGNANT_CLASSES, EmbeddingCache

STRATEGIES = ["frozen", "selective", "full"]


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
        seed_experiments(db)
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


class TestListExperiments:
    def test_returns_seeded_strategies_in_order(self, client):
        rows = client.get("/api/experiments").json()
        assert [r["strategy"] for r in rows] == STRATEGIES

    def test_all_strategies_start_awaiting_with_no_metrics(self, client):
        for row in client.get("/api/experiments").json():
            assert row["status"] == "awaiting"
            assert row["is_real_result"] is False
            for field in ("accuracy", "precision", "recall", "specificity", "f1", "auc"):
                assert row[field] is None

    def test_feasibility_gate_is_exposed_per_row(self, client):
        rows = {r["strategy"]: r for r in client.get("/api/experiments").json()}
        assert rows["frozen"]["feasible"] is True
        assert rows["selective"]["feasible"] is False
        assert rows["full"]["feasible"] is False
        assert "CUDA" in rows["full"]["feasibility_note"]

    def test_seed_repairs_mojibake_names_to_ascii(self, client):
        names = [r["name"] for r in client.get("/api/experiments").json()]
        assert names == [d["name"] for d in EXPERIMENT_DEFINITIONS]
        assert all(name.isascii() for name in names)


class TestRunExperiment:
    def test_unknown_name_is_404(self, client):
        r = client.post("/api/experiments/run", json={"name": "D - Not registered"})
        assert r.status_code == 404
        assert "not found" in r.json()["detail"].lower()

    @pytest.mark.parametrize("strategy", ["selective", "full"])
    def test_infeasible_strategies_are_409_with_reason(self, client, strategy):
        name = next(d["name"] for d in EXPERIMENT_DEFINITIONS if d["strategy"] == strategy)
        r = client.post("/api/experiments/run", json={"name": name})
        assert r.status_code == 409
        detail = r.json()["detail"]
        assert (strategy == "full" and "CPU-days" in detail) or "backward pass" in detail
        assert detail

    def test_infeasible_run_leaves_the_row_awaiting(self, client):
        rows = client.get("/api/experiments").json()
        full = next(r for r in rows if r["strategy"] == "full")
        client.post("/api/experiments/run", json={"name": full["name"]})
        after = next(r for r in client.get("/api/experiments").json() if r["id"] == full["id"])
        assert after["status"] == "awaiting"
        assert after["accuracy"] is None

    def test_frozen_without_embedding_caches_is_409(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr(registry, "EMBEDDING_DIR", tmp_path)
        name = next(d["name"] for d in EXPERIMENT_DEFINITIONS if d["strategy"] == "frozen")
        r = client.post("/api/experiments/run", json={"name": name})
        assert r.status_code == 409
        assert "extract_embeddings" in r.json()["detail"]


class TestDeleteExperiment:
    def test_delete_then_get_404(self, client):
        row = client.get("/api/experiments").json()[0]
        assert client.delete(f"/api/experiments/{row['id']}").json() == {"ok": True}
        assert client.delete(f"/api/experiments/{row['id']}").status_code == 404
        assert len(client.get("/api/experiments").json()) == 2


def _write_cache(tmp_path, name, embeddings, labels):
    ids = [f"{name}_{i}" for i in range(len(labels))]
    EmbeddingCache(tmp_path / f"{name}.npy").save(embeddings, labels, ids)


@pytest.fixture()
def synthetic_caches(tmp_path, monkeypatch):
    """Tiny separable 4-class caches for every split the runner reads."""
    monkeypatch.setattr(registry, "EMBEDDING_DIR", tmp_path)
    rng = np.random.default_rng(7)
    centers = rng.normal(size=(len(CANCER_CLASSES), 16))

    def make(n_per_class, tag):
        labels = np.repeat(np.arange(len(CANCER_CLASSES)), n_per_class)
        emb = centers[labels] + rng.normal(scale=0.2, size=(len(labels), 16))
        _write_cache(tmp_path, tag, emb.astype(np.float32), labels)

    for agent in registry.AGENT_SLUGS:
        make(5, f"train_{agent}")
    make(4, "val")
    return tmp_path


class TestTrainer:
    def test_learns_separable_classes(self):
        rng = np.random.default_rng(3)
        centers = rng.normal(size=(len(CANCER_CLASSES), 8))
        labels = np.repeat(np.arange(len(CANCER_CLASSES)), 40)
        emb = centers[labels] + rng.normal(scale=0.15, size=(len(labels), 8))

        head, epochs_run, holdout_f1 = experiments_service.train_head(
            emb.astype(np.float32), labels, seed=1
        )
        preds = head.predict(emb)
        assert (preds == labels).mean() > 0.95
        assert 0 < holdout_f1 <= 1
        assert epochs_run >= 1

    def test_early_stopping_terminates_before_max_epochs(self):
        rng = np.random.default_rng(5)
        labels = np.tile(np.arange(len(CANCER_CLASSES)), 30)
        emb = rng.normal(size=(len(labels), 8)).astype(np.float32)
        head, epochs_run, _ = experiments_service.train_head(
            emb, labels, epochs=500, patience=5, seed=2
        )
        assert epochs_run < 500
        assert head.weights.shape == (8, len(CANCER_CLASSES))


class TestFrozenRunEndToEnd:
    def test_synthetic_run_writes_real_metrics(self, client, synthetic_caches):
        name = next(d["name"] for d in EXPERIMENT_DEFINITIONS if d["strategy"] == "frozen")
        r = client.post("/api/experiments/run", json={"name": name})
        assert r.status_code == 200, r.text
        row = r.json()
        assert row["status"] == "completed"
        assert row["is_real_result"] is True
        # Separable synthetic data: the head must actually separate the classes.
        assert row["accuracy"] > 0.8
        for field in ("precision", "recall", "specificity", "f1"):
            assert 0.0 <= row[field] <= 1.0
        assert row["auc"] is None or 0.0 <= row["auc"] <= 1.0
        assert row["trainable_params"] == (16 + 1) * len(CANCER_CLASSES)
        assert row["train_seconds"] > 0
        assert "real run" in row["notes"]

    def test_metrics_survive_a_listing(self, client, synthetic_caches):
        name = next(d["name"] for d in EXPERIMENT_DEFINITIONS if d["strategy"] == "frozen")
        client.post("/api/experiments/run", json={"name": name})
        frozen = next(r for r in client.get("/api/experiments").json() if r["strategy"] == "frozen")
        assert frozen["status"] == "completed"
        assert frozen["accuracy"] > 0.8


class TestMalignantIndices:
    def test_malignant_indices_align_with_cancer_classes(self):
        assert experiments_service.MALIGNANT_INDICES == [
            CANCER_CLASSES.index(c) for c in MALIGNANT_CLASSES
        ]
