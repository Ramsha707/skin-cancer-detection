from __future__ import annotations

import numpy as np
import pytest
from app.services.metrics import (
    BinaryMetrics,
    confusion,
    evaluate,
    one_vs_rest,
    roc_auc_binary,
    weighted_class_weights,
)

CLASS_NAMES = ["melanoma", "basal_cell_carcinoma", "actinic_keratosis", "benign_lesion"]
MALIGNANT = [0, 1, 2]


def _onehot_scores(y_true, n_classes=4, confident=True):
    scores = np.full((len(y_true), n_classes), 0.05)
    for i, label in enumerate(y_true):
        scores[i, label] = 0.9 if confident else 0.4
    return scores


class TestBinaryMetrics:
    def test_perfect_prediction(self):
        m = BinaryMetrics(tp=10, fp=0, fn=0, tn=90)
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.f1 == 1.0
        assert m.support == 10

    def test_zero_division_is_not_an_error(self):
        m = BinaryMetrics()
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1 == 0.0
        assert m.specificity == 0.0


class TestConfusion:
    def test_counts_land_in_correct_cells(self):
        # 2 true class 0 both predicted 0; 1 true class 1 predicted as 0.
        matrix = confusion([0, 0, 1], [0, 0, 0], n_classes=2)
        assert matrix[0][0] == 2
        assert matrix[1][0] == 1
        assert matrix.sum() == 3

    def test_one_vs_rest_totals_are_consistent(self):
        y_true = [0, 1, 2, 2, 3, 3]
        y_pred = [0, 0, 2, 3, 3, 3]
        per_class = one_vs_rest(y_true, y_pred, n_classes=4)
        for c in range(4):
            m = per_class[c]
            assert m.tp + m.fn == y_true.count(c)
            assert m.tp + m.fp == y_pred.count(c)
            assert m.tp + m.fp + m.fn + m.tn == len(y_true)


class TestRocAuc:
    def test_perfect_ranking(self):
        assert roc_auc_binary([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)

    def test_inverted_ranking(self):
        assert roc_auc_binary([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == pytest.approx(0.0)

    def test_single_class_returns_none(self):
        assert roc_auc_binary([1, 1, 1], [0.1, 0.5, 0.9]) is None


class TestEvaluate:
    def test_mismatched_lengths_raise(self):
        # Silent broadcasting here would corrupt every reported metric.
        with pytest.raises(ValueError, match="same length"):
            evaluate([0, 1, 2], [0, 1], n_classes=3)

    def test_macro_f1_is_the_primary_metric(self):
        # Majority class predicted perfectly; the rarest class never detected.
        y_true = [3] * 200 + [2] * 20
        y_pred = [3] * 200 + [3] * 20
        report = evaluate(y_true, y_pred, n_classes=4, class_names=CLASS_NAMES)

        assert report.accuracy == pytest.approx(200 / 220)
        assert report.per_class_recall["benign_lesion"] == pytest.approx(1.0)
        assert report.per_class_recall["actinic_keratosis"] == pytest.approx(0.0)
        # Every one of the 20 missed keratoses was also called benign, so benign
        # precision falls to 200/220 and its F1 to 0.952. Macro-F1 is therefore
        # ~0.476 while accuracy is ~0.909, which is exactly why accuracy is not
        # the headline metric on this dataset.
        assert report.per_class["benign_lesion"]["precision"] == pytest.approx(200 / 220)
        assert report.macro_f1 == pytest.approx(0.9523809523809524 / 2)
        assert report.macro_f1 < 0.5 * report.accuracy + 0.25
        assert report.worst_class_recall == "actinic_keratosis"

    def test_malignant_auc_uses_summed_malignant_probability(self):
        y_true = [0, 1, 2, 3, 3]
        y_pred = [0, 1, 3, 3, 3]
        report = evaluate(
            y_true,
            y_pred,
            n_classes=4,
            class_names=CLASS_NAMES,
            malignant_indices=MALIGNANT,
            y_score=_onehot_scores(y_true),
        )
        assert report.malignant_auc == pytest.approx(1.0)
        # The hard-label AUC would be lower, since one actinic keratosis was
        # called benign; the probability-based AUC must win.
        assert report.auc < report.malignant_auc
        assert report.per_class_recall["actinic_keratosis"] == pytest.approx(0.0)
        assert report.false_negatives == 1

    def test_perfect_predictions_beat_majority_baseline(self):
        y_true = [3] * 200 + [0] * 20
        y_pred = list(y_true)
        report = evaluate(y_true, y_pred, n_classes=4, class_names=CLASS_NAMES)
        assert report.macro_f1 == pytest.approx(1.0)
        # Accuracy is high here, but the point of macro-F1 is that it would drop
        # if the model collapsed onto the majority class.
        assert report.accuracy > 0.9

    def test_confusion_matrix_is_square_and_sums_to_support(self):
        y_true = [0, 1, 2, 3]
        y_pred = [0, 0, 3, 3]
        report = evaluate(y_true, y_pred, n_classes=4, class_names=CLASS_NAMES)
        assert len(report.confusion_matrix) == 4
        assert all(len(row) == 4 for row in report.confusion_matrix)
        assert sum(sum(row) for row in report.confusion_matrix) == report.support

    def test_absent_classes_are_excluded_from_macro_average(self):
        # Only classes 0 and 3 appear; classes 1 and 2 must not drag the macro
        # average to zero.
        y_true = [0, 3, 0, 3]
        y_pred = [0, 3, 0, 3]
        report = evaluate(y_true, y_pred, n_classes=4, class_names=CLASS_NAMES)
        assert set(report.per_class_recall) == {"melanoma", "benign_lesion"}
        assert report.macro_f1 == pytest.approx(1.0)

    def test_as_dict_is_serialisable(self):
        report = evaluate([0, 1, 3], [0, 1, 3], n_classes=4, class_names=CLASS_NAMES)
        payload = report.as_dict()
        assert isinstance(payload, dict)
        assert "macro_f1" in payload
        assert "per_class_recall" in payload


class TestWeightedClassWeights:
    def test_rare_classes_get_more_weight(self):
        weights = weighted_class_weights([1113, 514, 327, 7919])
        assert weights.argmax() == 2  # actinic keratosis is rarest -> heaviest
        assert weights.argmin() == 3  # benign dominates -> lightest
        assert weights[2] > weights[1] > weights[0] > weights[3]
        assert weights.mean() == pytest.approx(1.0)

    def test_all_zero_counts_is_safe(self):
        assert np.allclose(weighted_class_weights([0, 0, 0, 0]), 1.0)
