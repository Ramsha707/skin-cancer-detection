from __future__ import annotations

import numpy as np
import pytest
from ml.models import EMBEDDING_DIM, N_CLASSES, CancerHead, softmax_cross_entropy
from ml.training.head import inverse_frequency_weights, train_head

D = EMBEDDING_DIM


def separable(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Each class gets its own high-variance axis, so a linear head can fit it."""
    rng = np.random.default_rng(seed)
    x = rng.normal(scale=0.01, size=(n, D)).astype(np.float32)
    y = np.arange(n) % N_CLASSES
    for i, c in enumerate(y):
        x[i, c * 4] += 3.0
    return x, y


class TestSoftmaxCrossEntropy:
    def test_loss_uses_log_of_the_target_probability(self):
        """Regression: the loss once read `logit - max_logit`, which is a
        negative margin, not a probability, so it was silently constant-ish."""
        logits = np.array([[10.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        loss, grad = softmax_cross_entropy(logits, np.array([0]))
        probs = np.exp(logits - logits.max()) / np.exp(logits - logits.max()).sum()
        assert loss == pytest.approx(-np.log(probs[0, 0]), rel=1e-5)
        assert loss < 1e-3

    def test_uniform_logits_give_log_four(self):
        logits = np.zeros((1, N_CLASSES), dtype=np.float32)
        loss, _ = softmax_cross_entropy(logits, np.array([0]))
        assert loss == pytest.approx(np.log(N_CLASSES), rel=1e-6)

    def test_loss_decreases_as_the_target_logit_grows(self):
        y = np.array([0])
        losses = [
            softmax_cross_entropy(np.full((1, 4), v, dtype=np.float32), y)[0]
            for v in (0.0, 2.0, 5.0, 10.0)
        ]
        assert losses == sorted(losses, reverse=True)

    def test_gradient_is_the_standard_residual(self):
        logits = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        _, grad = softmax_cross_entropy(logits, np.array([0]))
        probs = np.exp(logits) / np.exp(logits).sum()
        expected = probs - np.array([[1.0, 0.0, 0.0, 0.0]])
        assert grad == pytest.approx(expected, rel=1e-6)

    def test_gradient_rows_sum_to_zero(self):
        logits = np.random.default_rng(0).normal(size=(8, N_CLASSES)).astype(np.float32)
        y = np.arange(8) % N_CLASSES
        _, grad = softmax_cross_entropy(logits, y)
        assert np.allclose(grad.sum(axis=1), 0.0, atol=1e-6)

    def test_gradient_is_not_mutated_by_the_loss_computation(self):
        """The gradient must not alias `probs` in a way that corrupts callers
        that reuse the array."""
        logits = np.random.default_rng(1).normal(size=(4, N_CLASSES)).astype(np.float32)
        y = np.arange(4) % N_CLASSES
        logits_before = logits.copy()
        softmax_cross_entropy(logits, y)
        assert np.allclose(logits, logits_before)

    def test_is_numerically_stable_for_large_logits(self):
        """Max-shifting must keep the loss finite and accurate at large magnitudes."""
        logits = np.array([[1000.0, 999.0, 998.0, 997.0]], dtype=np.float32)
        loss, _ = softmax_cross_entropy(logits, np.array([0]))
        assert np.isfinite(loss)

        reference = float(
            -np.log(
                np.exp(logits.astype(np.float64) - 1000.0)
                / np.exp(logits.astype(np.float64) - 1000.0).sum()
            )[0, 0]
        )
        assert loss == pytest.approx(reference, rel=1e-5)

    def test_gradient_shape_matches_logits(self):
        logits = np.random.default_rng(2).normal(size=(5, N_CLASSES)).astype(np.float32)
        _, grad = softmax_cross_entropy(logits, np.arange(5) % N_CLASSES)
        assert grad.shape == logits.shape


class TestInverseFrequencyWeights:
    def test_rare_class_gets_the_larger_weight(self):
        y = np.array([0] * 90 + [1] * 8 + [2] + [3])
        w = inverse_frequency_weights(y, N_CLASSES)
        assert w[0] < w[1]

    def test_weights_are_inversely_proportional_to_frequency(self):
        y = np.array([0] * 75 + [1] * 25)
        w = inverse_frequency_weights(y, N_CLASSES)
        ratio = w[1] / w[0]
        assert ratio == pytest.approx(3.0, rel=1e-6)

    def test_balanced_classes_get_equal_weights(self):
        y = np.arange(40) % 4
        w = inverse_frequency_weights(y, N_CLASSES)
        assert max(w.values()) == pytest.approx(min(w.values()))


class TestTrainHead:
    def test_learns_separable_structure(self):
        x, y = separable(120)
        result = train_head(x, y, epochs=30, learning_rate=0.5)
        assert (result.head.predict(x) == y).mean() > 0.9

    def test_loss_is_finite_and_recorded_per_epoch(self):
        x, y = separable(40)
        result = train_head(x, y, epochs=5)
        assert len(result.history) == 5
        assert all(np.isfinite(h) for h in result.history)

    def test_init_head_is_copied_not_mutated(self):
        """A caller passing the shared prompt head must not find it overwritten."""
        x, y = separable(40)
        init = CancerHead()
        before = init.weights.copy()
        train_head(x, y, init=init, epochs=3)
        assert np.allclose(init.weights, before)

    def test_does_not_mutate_the_input_embeddings(self):
        x, y = separable(40)
        before = x.copy()
        train_head(x, y, epochs=3)
        assert np.allclose(x, before)

    def test_class_weighting_lifts_minority_recall(self):
        """The reason class weights exist: unweighted training collapses onto
        the majority class, so malignant recall stays near zero."""
        rng = np.random.default_rng(7)
        n = 400
        x = rng.normal(scale=0.01, size=(n, D)).astype(np.float32)
        y = np.array([0] * 340 + [1] * 40 + [2] * 12 + [3] * 8)
        for i, c in enumerate(y):
            x[i, c * 4] += 2.0
        weights = inverse_frequency_weights(y, N_CLASSES)

        plain = train_head(x, y, epochs=25, learning_rate=0.5)
        weighted = train_head(x, y, epochs=25, learning_rate=0.5, class_weight=weights)

        def minority_recall(head: CancerHead) -> float:
            pred = head.predict(x)
            mask = y != 0
            return float((pred[mask] == y[mask]).mean())

        assert minority_recall(weighted.head) > minority_recall(plain.head)

    def test_reports_steps_and_samples(self):
        x, y = separable(50)
        result = train_head(x, y, epochs=2)
        assert result.steps == 2  # full batch, one step per epoch
        assert result.samples == 50

    def test_minibatch_increases_step_count(self):
        x, y = separable(50)
        result = train_head(x, y, epochs=2, minibatch=10)
        assert result.steps == 10  # 5 minibatches x 2 epochs

    def test_zero_embeddings_do_not_divide_by_zero(self):
        x = np.zeros((8, D), dtype=np.float32)
        y = np.arange(8) % N_CLASSES
        result = train_head(x, y, epochs=2)
        assert np.isfinite(result.head.weights).all()

    @pytest.mark.parametrize("epochs", [0])
    def test_zero_epochs_returns_the_init_head_unchanged(self, epochs):
        x, y = separable(20)
        init = CancerHead(weights=np.full((D, N_CLASSES), 0.25, dtype=np.float32))
        before = init.weights.copy()
        result = train_head(x, y, init=init, epochs=epochs)
        assert np.allclose(result.head.weights, before)