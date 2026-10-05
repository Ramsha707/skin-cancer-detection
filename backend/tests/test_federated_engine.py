from __future__ import annotations

import numpy as np
import pytest
from app.database.base import Base, get_db
from app.federated.engine import FederatedEngine, HospitalAgentRunner, fedavg
from app.main import app
from app.models import HospitalAgent
from app.services.dataset import CANCER_CLASSES
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

D = 1152


def toy_embeddings(n: int, seed: int = 0) -> np.ndarray:
    """Separable embeddings: each class sits on its own axis.

    A linear head can fit this perfectly, which makes "did training work" a
    decidable assertion instead of a tolerance judgement.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(scale=0.01, size=(n, D)).astype(np.float32)
    for i in range(n):
        x[i, (i % len(CANCER_CLASSES)) * 4] += 3.0
    return x


@pytest.fixture()
def engine() -> FederatedEngine:
    val_x = toy_embeddings(40, seed=99)
    val_y = np.arange(40) % len(CANCER_CLASSES)
    eng = FederatedEngine(val_x, val_y)
    eng.initialise(np.zeros((D, len(CANCER_CLASSES)), dtype=np.float32))
    return eng


class TestFedavg:
    def test_is_the_plain_mean_when_weights_are_equal(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        out = fedavg([a, b], [10, 10])
        assert out.tolist() == [0.5, 0.5]

    def test_weights_by_sample_count(self):
        """A 2,049-image site should not vote the same as a 1,413-image one."""
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        out = fedavg([a, b], [3, 1])
        assert out.tolist() == [0.75, 0.25]

    def test_server_lr_scales_the_result(self):
        a = np.array([2.0], dtype=np.float32)
        assert fedavg([a], [1], server_lr=0.5)[0] == pytest.approx(1.0)

    def test_rejects_empty_client_list(self):
        with pytest.raises(ValueError):
            fedavg([], [])

    def test_rejects_all_zero_weights(self):
        with pytest.raises(ValueError):
            fedavg([np.zeros(2, np.float32)], [0, 0])

    def test_output_dtype_is_float32(self):
        assert fedavg([np.zeros(3, np.float64)], [1]).dtype == np.float32


class TestHospitalAgentPrivacyInvariant:
    def test_update_returns_only_a_flat_parameter_vector(self, engine):
        """The whole privacy claim rests on this: nothing but weights exits."""
        x, y = toy_embeddings(20), np.arange(20) % len(CANCER_CLASSES)
        runner = engine.register("a1", x, y)
        runner.fit()
        delta = runner.update()
        assert delta.shape == (D * len(CANCER_CLASSES) + len(CANCER_CLASSES),)
        assert delta.dtype == np.float32

    def test_agent_api_has_no_dataset_shaped_parameter(self):
        """A structural check: no public method accepts a path or PIL image."""
        import inspect

        for name, member in inspect.getmembers(HospitalAgentRunner, inspect.isfunction):
            if name.startswith("_"):
                continue
            params = inspect.signature(member).parameters
            for pname in params:
                assert "path" not in pname.lower(), f"{name}({pname}) accepts a path"
                assert "image" not in pname.lower(), f"{name}({pname}) accepts an image"

    def test_embeddings_are_not_exposed_as_public_attributes(self):
        x, y = toy_embeddings(8), np.arange(8) % len(CANCER_CLASSES)
        runner = HospitalAgentRunner("a1", x, y)
        assert not any(
            isinstance(v, np.ndarray) for k, v in vars(runner).items() if not k.startswith("_")
        )

    def test_flatten_unflatten_round_trips(self):
        head_weights = np.random.default_rng(0).normal(size=(D, 4)).astype(np.float32)
        bias = np.random.default_rng(1).normal(size=4).astype(np.float32)
        from ml.models import CancerHead

        head = CancerHead(weights=head_weights, bias=bias)
        flat = HospitalAgentRunner._flatten(head.state_dict())
        back = HospitalAgentRunner._unflatten(flat)
        assert np.allclose(back.weights, head_weights)
        assert np.allclose(back.bias, bias)


class TestRoundExecution:
    def test_round_updates_global_weights(self, engine):
        before = engine.global_state.copy()
        engine.register("a1", toy_embeddings(40), np.arange(40) % 4)
        engine.register("a2", toy_embeddings(30, seed=5), np.arange(30) % 4)
        engine.run_round(1)
        assert not np.allclose(engine.global_state, before)

    def test_round_learns_the_separable_structure(self, engine):
        engine.register("a1", toy_embeddings(80), np.arange(80) % 4)
        result = engine.run_round(1)
        assert result.accuracy > 0.9
        assert result.macro_f1 > 0.9

    def test_version_increments_with_the_round(self, engine):
        engine.register("a1", toy_embeddings(40), np.arange(40) % 4)
        assert engine.run_round(1).global_version == "v1"
        assert engine.run_round(2).global_version == "v2"

    def test_reports_all_participants_and_their_samples(self, engine):
        engine.register("a1", toy_embeddings(40), np.arange(40) % 4)
        engine.register("a2", toy_embeddings(20, seed=3), np.arange(20) % 4)
        result = engine.run_round(1)
        assert result.participating == 2
        assert result.samples == 60

    def test_local_metrics_are_per_agent(self, engine):
        engine.register("a1", toy_embeddings(40), np.arange(40) % 4)
        result = engine.run_round(1)
        assert [r["agent_id"] for r in result.agent_results] == ["a1"]
        assert 0.0 <= result.agent_results[0]["local_accuracy"] <= 1.0

    def test_agent_results_carry_no_per_sample_output(self, engine):
        """Metrics only. If a predictions array ever appears here, the audit
        payload is leaking."""
        engine.register("a1", toy_embeddings(40), np.arange(40) % 4)
        keys = set(engine.run_round(1).agent_results[0])
        assert keys == {
            "agent_id",
            "samples",
            "loss",
            "local_accuracy",
            "local_macro_f1",
            "steps",
        }

    def test_init_required_before_rounds(self, engine):
        engine.global_state = None
        with pytest.raises(RuntimeError, match="initialise"):
            engine.run_round(1)

    def test_no_agents_raises(self, engine):
        with pytest.raises(ValueError, match="no agents"):
            engine.run_round(1)

    def test_evaluate_scores_the_global_model(self, engine):
        engine.register("a1", toy_embeddings(80), np.arange(80) % 4)
        engine.run_round(1)
        metrics = engine.evaluate(engine._val_embeddings, engine._val_labels)
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert metrics["loss"] > 0.0


class TestEngineInitialisation:
    def test_zero_weights_round_trip_into_global_state(self):
        eng = FederatedEngine()
        w = np.zeros((D, 4), dtype=np.float32)
        version = eng.initialise(w)
        assert version == "v0-prompt-init"
        assert eng.global_state.shape == (D * 4 + 4,)

    def test_initialised_weights_are_actually_loaded(self):
        """Regression: a transposed seed used to run but score near-uniform."""
        eng = FederatedEngine()
        w = np.zeros((D, 4), dtype=np.float32)
        w[0, 0] = 5.0  # strongly favours class 0 along feature 0
        eng.initialise(w)
        head = eng.global_head()
        assert head.weights.shape == (D, 4)
        assert head.weights[0, 0] == pytest.approx(5.0)
        assert head.predict(np.array([[5.0] + [0.0] * (D - 1)], np.float32))[0] == 0

    def test_global_head_requires_initialisation(self):
        with pytest.raises(RuntimeError):
            FederatedEngine().global_head()


@pytest.fixture()
def client():
    """Throwaway in-memory database; the real `skinfl.db` seeds from HAM10000."""
    db_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(db_engine)
    Session = sessionmaker(bind=db_engine)

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