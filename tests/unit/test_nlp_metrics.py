"""Unit tests for src.nlp.metrics — task dispatch, exact values, guards."""

import json
import math

import numpy as np
import pytest

from src.nlp.labels import LabelSpace
from src.nlp.metrics import compute_metrics, summarize_for_ranking

pytestmark = pytest.mark.unit

MC_CLASSES = ["人事", "採購", "預算"]


@pytest.fixture
def mc_space():
    return LabelSpace(classes=list(MC_CLASSES), is_multilabel=False)


@pytest.fixture
def ml_space():
    return LabelSpace(classes=list(MC_CLASSES), is_multilabel=True)


# --------------------------------------------------------------------------- #
# multiclass — hand-computed values
# --------------------------------------------------------------------------- #
def test_multiclass_hand_computed_toy(mc_space):
    # true: 人事, 採購, 預算, 人事 / pred: 人事, 預算, 預算, 人事
    result = compute_metrics([0, 1, 2, 0], [0, 2, 2, 0], mc_space)

    assert result["accuracy"] == pytest.approx(0.75)
    assert result["balanced_accuracy"] == pytest.approx(2 / 3)
    # per-class f1: 1.0, 0.0 (never predicted), 2/3 (P=0.5, R=1)
    assert result["f1_macro"] == pytest.approx((1.0 + 0.0 + 2 / 3) / 3)
    assert result["f1_micro"] == pytest.approx(0.75)
    assert result["f1_weighted"] == pytest.approx((2 * 1.0 + 1 * 0.0 + 1 * (2 / 3)) / 4)
    assert result["per_class_f1"] == {
        "人事": pytest.approx(1.0),
        "採購": pytest.approx(0.0),
        "預算": pytest.approx(2 / 3),
    }
    assert result["confusion_matrix"] == [[2, 0, 0], [0, 0, 1], [0, 0, 1]]
    assert result["n_samples"] == 4
    assert "pr_auc_macro" not in result


def test_multiclass_perfect_prediction(mc_space):
    result = compute_metrics([0, 1, 2], [0, 1, 2], mc_space)
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["balanced_accuracy"] == pytest.approx(1.0)
    assert result["f1_macro"] == pytest.approx(1.0)
    assert result["f1_micro"] == pytest.approx(1.0)
    assert result["f1_weighted"] == pytest.approx(1.0)
    assert result["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def test_multiclass_balanced_accuracy_on_imbalanced_toy():
    space = LabelSpace(classes=["多數", "少數"], is_multilabel=False)
    result = compute_metrics([0, 0, 0, 0, 1], [0, 0, 0, 0, 0], space)
    assert result["accuracy"] == pytest.approx(0.8)
    assert result["balanced_accuracy"] == pytest.approx(0.5)


def test_multiclass_per_class_f1_keys_are_class_names(mc_space):
    result = compute_metrics([0, 1, 2], [0, 1, 2], mc_space)
    assert list(result["per_class_f1"]) == MC_CLASSES


# --------------------------------------------------------------------------- #
# multiclass — PR-AUC
# --------------------------------------------------------------------------- #
def test_multiclass_pr_auc_present_only_with_proba(mc_space):
    proba = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.05, 0.05, 0.9],
        [0.9, 0.05, 0.05],
    ])
    result = compute_metrics([0, 1, 2, 0], [0, 1, 2, 0], mc_space, y_proba=proba)
    assert result["pr_auc_macro"] == pytest.approx(1.0)
    assert 0.0 <= result["pr_auc_macro"] <= 1.0


def test_multiclass_pr_auc_two_class_binarize_quirk():
    space = LabelSpace(classes=["否", "是"], is_multilabel=False)
    proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]])
    result = compute_metrics([0, 1, 0, 1], [0, 1, 0, 1], space, y_proba=proba)
    assert result["pr_auc_macro"] == pytest.approx(1.0)


def test_multiclass_pr_auc_guards_classes_absent_from_y_true(mc_space):
    # Class index 2 never appears in y_true; its all-zero AP column is
    # skipped instead of poisoning the macro average.
    proba = np.array([
        [0.7, 0.2, 0.1],
        [0.2, 0.7, 0.1],
        [0.6, 0.3, 0.1],
        [0.3, 0.6, 0.1],
    ])
    result = compute_metrics([0, 1, 0, 1], [0, 1, 0, 0], mc_space, y_proba=proba)
    value = result["pr_auc_macro"]
    assert not math.isnan(value)
    assert 0.0 <= value <= 1.0
    assert value == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# multilabel — hand-computed values
# --------------------------------------------------------------------------- #
def test_multilabel_hand_computed_toy(ml_space):
    y_true = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 0]])
    y_pred = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 0]])
    result = compute_metrics(y_true, y_pred, ml_space)

    assert result["subset_accuracy"] == pytest.approx(1 / 3)
    assert result["hamming_loss"] == pytest.approx(2 / 9)
    assert result["f1_micro"] == pytest.approx(0.75)
    assert result["f1_macro"] == pytest.approx((2 / 3 + 1.0 + 0.0) / 3)
    assert result["per_label_f1"] == {
        "人事": pytest.approx(2 / 3),
        "採購": pytest.approx(1.0),
        "預算": pytest.approx(0.0),
    }
    assert result["n_samples"] == 3
    assert "pr_auc_macro" not in result


def test_multilabel_perfect_prediction(ml_space):
    y = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
    result = compute_metrics(y, y.copy(), ml_space)
    assert result["subset_accuracy"] == pytest.approx(1.0)
    assert result["hamming_loss"] == pytest.approx(0.0)
    assert result["f1_micro"] == pytest.approx(1.0)


def test_multilabel_pr_auc_guards_absent_label_column(ml_space):
    y_true = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0]])  # column 2 all zero
    y_pred = y_true.copy()
    proba = np.array([[0.9, 0.1, 0.5], [0.8, 0.2, 0.5], [0.1, 0.9, 0.5]])
    result = compute_metrics(y_true, y_pred, ml_space, y_proba=proba)
    value = result["pr_auc_macro"]
    assert not math.isnan(value)
    assert value == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# validation errors
# --------------------------------------------------------------------------- #
def test_multiclass_length_mismatch_raises(mc_space):
    with pytest.raises(ValueError, match="samples"):
        compute_metrics([0, 1, 2], [0, 1], mc_space)


def test_multiclass_2d_y_true_raises(mc_space):
    with pytest.raises(ValueError, match="1-D"):
        compute_metrics([[0, 1], [1, 0]], [0, 1], mc_space)


def test_multiclass_out_of_range_index_raises(mc_space):
    with pytest.raises(ValueError, match="outside"):
        compute_metrics([0, 5], [0, 1], mc_space)


def test_multiclass_proba_wrong_width_raises(mc_space):
    proba = np.full((3, 4), 0.25)
    with pytest.raises(ValueError, match="y_proba"):
        compute_metrics([0, 1, 2], [0, 1, 2], mc_space, y_proba=proba)


def test_multiclass_zero_samples_raises(mc_space):
    with pytest.raises(ValueError, match="zero samples"):
        compute_metrics([], [], mc_space)


def test_multilabel_given_1d_arrays_raises(ml_space):
    with pytest.raises(ValueError, match=r"\(n, 3\)"):
        compute_metrics([0, 1, 2], [0, 1, 2], ml_space)


def test_multilabel_wrong_width_raises(ml_space):
    y = np.zeros((2, 2), dtype=int)
    with pytest.raises(ValueError, match=r"\(n, 3\)"):
        compute_metrics(y, y, ml_space)


def test_multilabel_proba_wrong_shape_raises(ml_space):
    y = np.zeros((2, 3), dtype=int)
    y[0, 0] = 1
    with pytest.raises(ValueError, match="y_proba"):
        compute_metrics(y, y, ml_space, y_proba=np.zeros((2, 2)))


def test_label_space_type_checked():
    with pytest.raises(TypeError, match="LabelSpace"):
        compute_metrics([0, 1], [0, 1], {"classes": ["a", "b"]})


# --------------------------------------------------------------------------- #
# JSON serializability
# --------------------------------------------------------------------------- #
def test_multiclass_result_is_json_serializable(mc_space):
    proba = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    result = compute_metrics([0, 1, 2], [0, 1, 2], mc_space, y_proba=proba)
    parsed = json.loads(json.dumps(result))
    assert parsed["n_samples"] == 3
    assert parsed["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def test_multilabel_result_is_json_serializable(ml_space):
    y = np.array([[1, 0, 0], [0, 1, 1]])
    proba = np.array([[0.9, 0.1, 0.2], [0.1, 0.8, 0.7]])
    result = compute_metrics(y, y.copy(), ml_space, y_proba=proba)
    parsed = json.loads(json.dumps(result))
    assert parsed["subset_accuracy"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# summarize_for_ranking
# --------------------------------------------------------------------------- #
def test_summarize_for_ranking_returns_f1_macro_for_both_tasks(mc_space, ml_space):
    mc = compute_metrics([0, 1, 2, 0], [0, 2, 2, 0], mc_space)
    assert summarize_for_ranking(mc, is_multilabel=False) == pytest.approx(mc["f1_macro"])

    y = np.array([[1, 0, 0], [0, 1, 1]])
    ml = compute_metrics(y, y.copy(), ml_space)
    assert summarize_for_ranking(ml, is_multilabel=True) == pytest.approx(ml["f1_macro"])


def test_summarize_for_ranking_missing_key_raises():
    with pytest.raises(ValueError, match="f1_macro"):
        summarize_for_ranking({"accuracy": 1.0}, is_multilabel=False)
