"""Week 7: pure federated engine tests.

No database and no FastAPI here -- these cover the tensor layer only:
FedAvg arithmetic, the round loop, and the structural privacy invariant that
nothing but a weight delta ever leaves a `HospitalAgentRunner`.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from app.federated.engine import (
    FederatedEngine,
    HospitalAgentRunner,
    RoundResult,
    weighted_average,
)

from ml.models import CANCER_CLASSES, N_CLASSES, CancerHead

D = 64


def toy_embeddings(n: int, dim: int = D, seed: int = 0) -> np.ndarray:
    """Separable embeddings: each class sits on its own block of features.

    A linear head can fit this perfectly, which makes "did training work" a
    decidable assertion instead of a tolerance judgement.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(scale=0.01, size=(n, dim)).astype(np.float32)
    width = max(1, dim // N_CLASSES)
    for i in range(n):
        x[i, (i % N_CLASSES) * width] += 3.0
    return x


def toy_labels(n: int) -> np.ndarray:
    return np.arange(n) % N_CLASSES


@pytest.fixture()
def engine() -> FederatedEngine:
    eng = FederatedEngine(
        toy_embeddings(40, seed=99), toy_labels(40), server_lr=1.0
    )
    eng.initialise(CancerHead(weights=np.zeros((D, N_CLASSES), dtype=np.float32)))
    return eng


def _runner_kwargs() -> dict:
    return {"learning_rate": 5e-2, "local_epochs": 80, "patience": 80}


class TestWeightedAverage:
    def test_is_the_plain_mean_when_weights_are_equal(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        out = weighted_average([a, b], [10, 10])
        assert out.tolist() == [0.5, 0.5]

    def test_weights_by_sample_count(self):
        """A 2,049-image site should not vote the same as a 1,413-image one."""
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        out = weighted_average([a, b], [3, 1])
        assert out.tolist() == [0.75, 0.25]

    def test_rejects_empty_client_list(self):
        with pytest.raises(ValueError, match="at least one"):
            weighted_average([], [])

    def test_rejects_all_zero_weights(self):
        with pytest.raises(ValueError, match="non-zero"):
            weighted_average(
                [np.zeros(2, np.float32), np.zeros(2, np.float32)], [0, 0]
            )

    def test_rejects_length_mismatch(self):
        with pytest.raises(ValueError, match="deltas and"):
            weighted_average([np.zeros(2, np.float32)], [1, 2])

    def test_output_dtype_is_float32(self):
        assert weighted_average([np.zeros(3, np.float64)], [1]).dtype == np.float32


class TestPrivacyInvariant:
    def test_update_returns_only_a_flat_parameter_vector(self, engine):
        """The whole privacy claim rests on this: nothing but weights exits."""
        runner = engine.register("agent-a", toy_embeddings(20), toy_labels(20), **_runner_kwargs())
        runner.load(engine.global_state)
        runner.fit()
        delta = runner.update()
        assert delta.shape == (D * N_CLASSES + N_CLASSES,)
        assert delta.dtype == np.float32

    def test_agent_api_has_no_dataset_shaped_parameter(self):
        """Structural check: no public method accepts a path or image."""
        for name, member in inspect.getmembers(HospitalAgentRunner, inspect.isfunction):
            if name.startswith("_"):
                continue
            for pname in inspect.signature(member).parameters:
                assert "path" not in pname.lower(), f"{name}({pname}) accepts a path"
                assert "image" not in pname.lower(), f"{name}({pname}) accepts an image"

    def test_embeddings_are_not_exposed_as_public_attributes(self):
        runner = HospitalAgentRunner("agent-a", toy_embeddings(8), toy_labels(8))
        assert not any(
            isinstance(v, np.ndarray) for k, v in vars(runner).items() if not k.startswith("_")
        )

    def test_agent_results_carry_no_per_sample_output(self, engine):
        """Metrics only. If a predictions array ever appears here, the audit
        payload is leaking."""
        engine.register("agent-a", toy_embeddings(40), toy_labels(40), **_runner_kwargs())
        result = engine.run_round(1)
        keys = set(result.agent_results[0])
        assert keys == {
            "agent_id",
            "samples",
            "loss",
            "epochs_run",
            "local_accuracy",
            "local_precision",
            "local_recall",
            "local_f1",
            "local_macro_f1",
            "weight_delta_norm",
            "update_size_kb",
        }


class TestRoundExecution:
    def test_round_updates_global_weights(self, engine):
        before = engine.global_state.copy()
        engine.register("agent-a", toy_embeddings(40), toy_labels(40), **_runner_kwargs())
        engine.register("agent-b", toy_embeddings(30, seed=5), toy_labels(30), **_runner_kwargs())
        engine.run_round(1)
        assert not np.allclose(engine.global_state, before)

    def test_round_learns_the_separable_structure(self, engine):
        engine.register("agent-a", toy_embeddings(80), toy_labels(80), **_runner_kwargs())
        result = engine.run_round(1)
        assert result.evaluated is True
        assert result.accuracy is not None and result.accuracy > 0.9
        assert result.macro_f1 is not None and result.macro_f1 > 0.9

    def test_version_increments_with_the_round(self, engine):
        engine.register("agent-a", toy_embeddings(40), toy_labels(40), **_runner_kwargs())
        assert engine.run_round(1).global_version == "v1"
        assert engine.run_round(2).global_version == "v2"

    def test_reports_all_participants_and_their_samples(self, engine):
        engine.register("agent-a", toy_embeddings(40), toy_labels(40), **_runner_kwargs())
        engine.register("agent-b", toy_embeddings(20, seed=3), toy_labels(20), **_runner_kwargs())
        result = engine.run_round(1)
        assert result.participating == 2
        assert result.samples == 60

    def test_local_metrics_are_per_agent(self, engine):
        engine.register("agent-a", toy_embeddings(40), toy_labels(40), **_runner_kwargs())
        result = engine.run_round(1)
        assert [r["agent_id"] for r in result.agent_results] == ["agent-a"]
        entry = result.agent_results[0]
        assert 0.0 <= entry["local_accuracy"] <= 1.0
        assert entry["weight_delta_norm"] > 0.0
        assert entry["update_size_kb"] > 0.0

    def test_aggregation_is_the_weighted_mean_of_deltas(self, engine):
        """Global-after must equal global-before plus the sample-weighted mean
        of the client deltas, which is the FedAvg definition verbatim."""
        engine.register("agent-a", toy_embeddings(40), toy_labels(40), **_runner_kwargs())
        engine.register("agent-b", toy_embeddings(20, seed=3), toy_labels(20), **_runner_kwargs())
        before = engine.global_state.copy()

        captured: list[tuple[np.ndarray, int]] = []

        def observe(runner, local, delta):
            captured.append((delta, local.samples))

        result = engine.run_round(1, on_agent_trained=observe)

        expected_step = weighted_average(
            [d for d, _ in captured], [n for _, n in captured]
        )
        assert np.allclose(engine.global_state, before + expected_step, atol=1e-6)
        assert result.samples == 60

    def test_warm_start_improves_on_zeros(self, engine):
        """`init` must actually be used: training a second round from round 1's
        weights must beat a fresh zero-init round on the same data."""
        engine.register("agent-a", toy_embeddings(80), toy_labels(80), **_runner_kwargs())
        first = engine.run_round(1)
        second = engine.run_round(2)
        assert second.macro_f1 >= first.macro_f1 - 1e-9

    def test_init_required_before_rounds(self, engine):
        engine.global_state = None
        with pytest.raises(RuntimeError, match="initialise"):
            engine.run_round(1)

    def test_no_agents_raises(self, engine):
        with pytest.raises(ValueError, match="no agents"):
            engine.run_round(1)

    def test_update_before_load_raises(self):
        runner = HospitalAgentRunner("agent-a", toy_embeddings(8), toy_labels(8))
        runner.fit()
        with pytest.raises(RuntimeError, match="broadcast"):
            runner.update()

    def test_observer_hook_sees_every_agent(self, engine):
        engine.register("agent-a", toy_embeddings(20), toy_labels(20), **_runner_kwargs())
        engine.register("agent-b", toy_embeddings(20, seed=2), toy_labels(20), **_runner_kwargs())
        seen = []

        def observe(runner, local, delta):
            seen.append((runner.agent_id, local.samples, delta.shape))

        engine.run_round(1, on_agent_trained=observe)
        assert [s[0] for s in seen] == ["agent-a", "agent-b"]
        assert all(shape == (D * N_CLASSES + N_CLASSES,) for _, _, shape in seen)


class TestValidationAndReset:
    def test_evaluate_scores_the_global_model(self, engine):
        engine.register("agent-a", toy_embeddings(80), toy_labels(80), **_runner_kwargs())
        engine.run_round(1)
        metrics = engine.evaluate(engine._val_embeddings, engine._val_labels)
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert metrics["loss"] > 0.0

    def test_without_validation_metrics_stay_none(self):
        eng = FederatedEngine(server_lr=1.0)
        eng.initialise(CancerHead(weights=np.zeros((D, N_CLASSES), dtype=np.float32)))
        eng.register("agent-a", toy_embeddings(40), toy_labels(40), **_runner_kwargs())
        result = eng.run_round(1)
        assert result.evaluated is False
        assert result.accuracy is None
        assert result.macro_f1 is None
        # Weights are still real and the version still advanced.
        assert result.global_version == "v1"
        assert eng.global_state is not None

    def test_reset_clears_everything(self, engine):
        engine.register("agent-a", toy_embeddings(40), toy_labels(40), **_runner_kwargs())
        engine.run_round(1)
        engine.reset()
        assert engine.global_state is None
        assert engine.current_global_version is None
        assert engine.agents == {}
        assert engine.rounds == []


class TestEngineInitialisation:
    def test_initialise_round_trips_a_head(self):
        eng = FederatedEngine()
        head = CancerHead(weights=np.zeros((D, N_CLASSES), dtype=np.float32))
        head.weights[0, 0] = 5.0
        version = eng.initialise(head, version="v0")
        assert version == "v0"
        assert eng.global_state.shape == (D * N_CLASSES + N_CLASSES,)
        back = eng.global_head()
        assert back.weights.shape == (D, N_CLASSES)
        assert back.weights[0, 0] == pytest.approx(5.0)

    def test_global_head_requires_initialisation(self):
        with pytest.raises(RuntimeError, match="initialise"):
            FederatedEngine().global_head()

    def test_round_result_defaults_to_unevaluated(self):
        result = RoundResult(
            round_number=1, global_version="v1", participating=0, samples=0, duration_seconds=0.0
        )
        assert result.evaluated is False
        assert result.accuracy is None

    def test_class_count_matches_the_taxonomy(self):
        assert N_CLASSES == len(CANCER_CLASSES) == 4
