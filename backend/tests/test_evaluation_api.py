"""Week 9: evaluation API over the held-out validation split.

The fixture builds tiny but genuinely separable embeddings and a matching
checkpoint, so every asserted metric is produced by running the head -- nothing
is stubbed. The not-ready paths are exercised with the same fixture by removing
one artefact at a time.
"""

from __future__ import annotations

import numpy as np
import pytest
from app import models  # noqa: F401  (registration side effect)
from app.database.base import Base, get_db
from app.main import app
from app.models import ModelVersion, TrainingRound
from app.services import registry
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from ml.models import CANCER_CLASSES, EMBEDDING_DIM, EmbeddingCache

N_PER_CLASS = 3


def _write_cache(directory, name: str, embeddings: np.ndarray, labels: np.ndarray) -> None:
    ids = [f"{name}_{i}" for i in range(len(labels))]
    EmbeddingCache(directory / f"{name}.npy").save(embeddings, labels, ids)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    embed_dir = tmp_path / "embeddings"
    ckpt_dir = tmp_path / "checkpoints"
    embed_dir.mkdir()
    ckpt_dir.mkdir()
    monkeypatch.setattr(registry, "EMBEDDING_DIR", embed_dir)
    monkeypatch.setattr(registry, "CHECKPOINT_DIR", ckpt_dir)

    rng = np.random.default_rng(19)
    centers = rng.normal(size=(len(CANCER_CLASSES), EMBEDDING_DIM)).astype(np.float32)
    labels = np.repeat(np.arange(len(CANCER_CLASSES)), N_PER_CLASS)
    embeddings = centers[labels] + rng.normal(
        scale=0.03, size=(len(labels), EMBEDDING_DIM)
    ).astype(np.float32)
    _write_cache(embed_dir, "val", embeddings.astype(np.float32), labels)

    head = centers.T.copy()
    np.savez(ckpt_dir / "v1.npz", weights=head, bias=np.zeros(len(CANCER_CLASSES)))

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autoflush=False, future=True)
    with Session() as db:
        db.add(
            TrainingRound(
                round_number=1,
                status="completed",
                global_model_version="v1",
                global_accuracy=0.75,
                global_f1=0.7,
                global_auc=0.8,
                global_loss=0.4,
            )
        )
        db.add(
            TrainingRound(round_number=2, status="running")
        )
        db.add(
            ModelVersion(
                version="v1",
                strategy="frozen",
                training_round=1,
                accuracy=0.75,
                precision=0.8,
                recall=0.7,
                specificity=0.9,
                f1=0.7,
                auc=0.8,
                loss=0.4,
                status="active",
                is_recommended=True,
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

    app.dependency_overrides[get_db] = override
    with TestClient(app) as client:
        yield {"client": client, "embed_dir": embed_dir, "ckpt_dir": ckpt_dir}
    app.dependency_overrides.clear()


class TestEvaluationSummary:
    def test_ready_report_reflects_a_real_run(self, env):
        body = env["client"].get("/api/evaluation/summary").json()
        assert body["ready"] is True
        assert body["is_real"] is True
        assert body["model_version"] == "v1"
        assert body["n_eval"] == N_PER_CLASS * len(CANCER_CLASSES)
        assert body["instructions"] is None

        metrics = body["metrics"]
        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["malignant_auc"] == pytest.approx(1.0)
        assert metrics["false_negatives"] == 0
        assert metrics["support"] == N_PER_CLASS * len(CANCER_CLASSES)

    def test_confusion_matrix_and_per_class_are_complete(self, env):
        body = env["client"].get("/api/evaluation/summary").json()
        matrix = body["confusion_matrix"]
        assert len(matrix) == len(CANCER_CLASSES)
        assert all(len(row) == len(CANCER_CLASSES) for row in matrix)
        assert sum(sum(row) for row in matrix) == body["n_eval"]
        # Perfect separation puts every validation sample on the diagonal.
        assert all(matrix[i][i] == N_PER_CLASS for i in range(len(CANCER_CLASSES)))

        assert set(body["per_class"]) == set(CANCER_CLASSES)
        for name, values in body["per_class"].items():
            assert values["support"] == N_PER_CLASS
            assert values["recall"] == pytest.approx(1.0)
            assert values["tp"] == N_PER_CLASS
            assert values["fp"] == 0
            assert values["fn"] == 0
            assert 0.0 <= values["specificity"] <= 1.0
            assert name in body["roc"]["per_class"]

    def test_roc_curves_cover_every_class_and_malignant_axis(self, env):
        roc = env["client"].get("/api/evaluation/summary").json()["roc"]
        assert set(roc["per_class"]) == set(CANCER_CLASSES)
        for points in roc["per_class"].values():
            assert len(points) >= 2
            assert points[0] == {"threshold": None, "fpr": 0.0, "tpr": 0.0}
            assert points[-1] == {"threshold": None, "fpr": 1.0, "tpr": 1.0}
        assert roc["malignant"] is not None
        assert roc["malignant"][-1]["tpr"] == pytest.approx(1.0)

    def test_learning_curve_and_model_history_come_from_the_database(self, env):
        body = env["client"].get("/api/evaluation/summary").json()
        curve = body["learning_curve"]
        assert [p["round_number"] for p in curve] == [1, 2]
        assert curve[0]["evaluated"] is True
        assert curve[0]["accuracy"] == pytest.approx(0.75)
        assert curve[1]["evaluated"] is False
        assert curve[1]["accuracy"] is None

        history = body["model_history"]
        assert len(history) == 1
        assert history[0]["version"] == "v1"
        assert history[0]["is_recommended"] is True
        assert history[0]["created_at"] is not None


class TestNotReady:
    def test_without_a_checkpoint_returns_instructions_not_numbers(self, env):
        for path in env["ckpt_dir"].glob("*.npz"):
            path.unlink()
        body = env["client"].get("/api/evaluation/summary").json()
        assert body["ready"] is False
        assert body["is_real"] is False
        assert body["metrics"] is None
        assert body["confusion_matrix"] == []
        assert body["roc"] == {"per_class": {}, "malignant": None}
        assert "federated round" in body["instructions"]
        # History still reports what the database knows.
        assert body["learning_curve"][0]["accuracy"] == pytest.approx(0.75)

    def test_without_validation_embeddings_points_at_extraction(self, env):
        (env["embed_dir"] / "val.npy").unlink()
        body = env["client"].get("/api/evaluation/summary").json()
        assert body["ready"] is False
        assert "extract_embeddings" in body["instructions"]
        assert body["n_eval"] == 0

    def test_dimension_mismatch_is_refused(self, env):
        rng = np.random.default_rng(5)
        labels = np.repeat(np.arange(len(CANCER_CLASSES)), N_PER_CLASS)
        _write_cache(
            env["embed_dir"],
            "val",
            rng.normal(size=(len(labels), 8)).astype(np.float32),
            labels,
        )
        body = env["client"].get("/api/evaluation/summary").json()
        assert body["ready"] is False
        assert "input width" in body["instructions"]
