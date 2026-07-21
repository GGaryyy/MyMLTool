"""Task-dispatched evaluation metrics for the Chinese-text benchmark harness.

:func:`compute_metrics` inspects ``label_space.is_multilabel`` and computes
the matching metric set with ``zero_division=0`` everywhere. Every returned
value is JSON-serializable (plain Python floats / ints / lists / dicts).

PR-AUC guard: average precision is undefined for a class whose ``y_true``
column contains no positive sample, so ``pr_auc_macro`` computes per-class
AP only for the classes actually present in ``y_true`` and averages those.
Multiclass f1 / confusion metrics pass ``labels=range(n_classes)`` so
result shapes stay stable across splits; a class absent from both
``y_true`` and ``y_pred`` scores 0.0 under ``zero_division=0``.
"""

from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
)
from sklearn.preprocessing import label_binarize

from src.nlp.labels import LabelSpace

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
RANKING_METRIC = "f1_macro"


def compute_metrics(y_true, y_pred, label_space: LabelSpace, y_proba=None) -> dict:
    """Compute the evaluation metric dict for one model run.

    Multiclass expects 1-D integer class indices for ``y_true`` / ``y_pred``;
    multilabel expects ``(n, n_classes)`` 0/1 indicator matrices. Optional
    ``y_proba`` is an ``(n, n_classes)`` score matrix and adds
    ``pr_auc_macro``. Shape mismatches raise ``ValueError``.
    """
    if not isinstance(label_space, LabelSpace):
        raise TypeError(f"label_space must be a LabelSpace, got {type(label_space).__name__}")
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_proba is not None:
        y_proba = np.asarray(y_proba, dtype=float)
    if label_space.is_multilabel:
        return _multilabel_metrics(y_true, y_pred, label_space, y_proba)
    return _multiclass_metrics(y_true, y_pred, label_space, y_proba)


def summarize_for_ranking(metrics: dict, is_multilabel: bool) -> float:
    """Headline score used to rank model runs: macro-F1 for both tasks.

    Macro-F1 weights every class equally, so it is robust to the long-tail
    class imbalance typical of Chinese text categories and stays comparable across
    multiclass and multilabel runs; ``is_multilabel`` is accepted so call
    sites stay explicit should the headline metric ever diverge by task.
    """
    if RANKING_METRIC not in metrics:
        raise ValueError(f"metrics dict is missing '{RANKING_METRIC}' (is_multilabel={is_multilabel})")
    return float(metrics[RANKING_METRIC])


def _check_common(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(f"y_true has {y_true.shape[0]} samples but y_pred has {y_pred.shape[0]}")
    if y_true.shape[0] == 0:
        raise ValueError("Cannot compute metrics on zero samples")


def _check_proba(y_proba: Optional[np.ndarray], n_samples: int, n_classes: int) -> None:
    if y_proba is not None and y_proba.shape != (n_samples, n_classes):
        raise ValueError(
            f"y_proba must have shape ({n_samples}, {n_classes}), got {y_proba.shape}"
        )


def _multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                        label_space: LabelSpace, y_proba: Optional[np.ndarray]) -> dict:
    n_classes = label_space.n_classes
    if y_true.ndim != 1:
        raise ValueError(f"Multiclass y_true must be 1-D class indices, got shape {y_true.shape}")
    if y_pred.ndim != 1:
        raise ValueError(f"Multiclass y_pred must be 1-D class indices, got shape {y_pred.shape}")
    _check_common(y_true, y_pred)
    for name, values in (("y_true", y_true), ("y_pred", y_pred)):
        if values.min() < 0 or values.max() >= n_classes:
            raise ValueError(f"{name} contains class indices outside [0, {n_classes})")
    _check_proba(y_proba, y_true.shape[0], n_classes)

    labels = list(range(n_classes))
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, y_pred, labels=labels, average="micro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_class_f1": {c: float(per_class[i]) for i, c in enumerate(label_space.classes)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "n_samples": int(y_true.shape[0]),
    }
    if y_proba is not None:
        metrics["pr_auc_macro"] = _pr_auc_macro_multiclass(y_true, y_proba, n_classes)
    return metrics


def _multilabel_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                        label_space: LabelSpace, y_proba: Optional[np.ndarray]) -> dict:
    n_classes = label_space.n_classes
    if y_true.ndim != 2 or y_true.shape[1] != n_classes:
        raise ValueError(f"Multilabel y_true must have shape (n, {n_classes}), got {y_true.shape}")
    if y_pred.ndim != 2 or y_pred.shape[1] != n_classes:
        raise ValueError(f"Multilabel y_pred must have shape (n, {n_classes}), got {y_pred.shape}")
    _check_common(y_true, y_pred)
    _check_proba(y_proba, y_true.shape[0], n_classes)

    per_label = f1_score(y_true, y_pred, average=None, zero_division=0)
    metrics = {
        "subset_accuracy": float(np.mean(np.all(y_true == y_pred, axis=1))),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_label_f1": {c: float(per_label[i]) for i, c in enumerate(label_space.classes)},
        "n_samples": int(y_true.shape[0]),
    }
    if y_proba is not None:
        metrics["pr_auc_macro"] = _pr_auc_macro_multilabel(y_true, y_proba)
    return metrics


def _pr_auc_macro_multiclass(y_true: np.ndarray, y_proba: np.ndarray,
                             n_classes: int) -> float:
    binarized = label_binarize(y_true, classes=list(range(n_classes)))
    if binarized.shape[1] == 1:
        # 2-class quirk: label_binarize returns the positive column only.
        binarized = np.hstack([1 - binarized, binarized])
    present = np.unique(y_true)
    scores = [
        float(average_precision_score(binarized[:, int(c)], y_proba[:, int(c)]))
        for c in present
    ]
    return float(np.mean(scores))


def _pr_auc_macro_multilabel(y_true: np.ndarray, y_proba: np.ndarray) -> Optional[float]:
    present = [c for c in range(y_true.shape[1]) if y_true[:, c].sum() > 0]
    if not present:
        return None  # undefined: no class has a positive sample
    scores = [
        float(average_precision_score(y_true[:, c], y_proba[:, c]))
        for c in present
    ]
    return float(np.mean(scores))
