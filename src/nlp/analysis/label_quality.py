"""Label-quality analysis via self-implemented confident learning.

COMPLIANCE NOTE: the reference confident-learning implementation,
``cleanlab``, is licensed AGPL-3.0 and is therefore PROHIBITED in this
project. This module deliberately re-implements the confident-learning
threshold rule from scratch on top of scikit-learn only.

Multiclass rule: out-of-fold predicted probabilities from a char TF-IDF +
LogisticRegression pipeline give each document a self-confidence
``p(given label)``. The per-class threshold ``t_c`` is the mean
self-confidence of documents whose given label is ``c``. A document is a
suspect when its self-confidence falls below its class threshold AND the
model's argmax disagrees with the given label.

Multilabel simplification: each label column is scored as an independent
binary problem (out-of-fold positive probability per label). A given
positive is suspect when its probability is below both the positive-class
mean and 0.5; a given negative is suspect when its probability exceeds
both the negative-class mean and 0.5. Suggestions are encoded as
``"+label"`` / ``"-label"`` strings. Joint label interactions are ignored
on purpose — this is a triage list, not an automatic relabeler.
"""

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from src.nlp.config import VALID_TASK_TYPES
from src.nlp.labels import LabelSpace, build_label_space

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
MAX_SUSPECTS = 100
MIN_DOCS_PER_FOLD = 3
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MAX_FEATURES = 30000


@dataclass
class SuspectLabel:
    """One potentially mislabelled document."""

    row: int
    given: str
    suggested: str
    self_confidence: float


@dataclass
class LabelQualityReport:
    """Confident-learning triage summary over one labelled corpus."""

    n_docs: int
    n_suspects: int
    suspect_ratio: float
    suspects: list
    per_class_thresholds: dict
    notes: list

    def to_dict(self) -> dict:
        return asdict(self)


def _oof_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=TFIDF_NGRAM_RANGE,
                                  max_features=TFIDF_MAX_FEATURES)),
        ("logreg", LogisticRegression(max_iter=1000)),
    ])


def _empty_report(n_docs: int, notes: list) -> LabelQualityReport:
    return LabelQualityReport(n_docs=n_docs, n_suspects=0, suspect_ratio=0.0,
                              suspects=[], per_class_thresholds={}, notes=notes)


def find_label_issues(texts: Sequence[str], raw_labels: Sequence[str], task_type: str,
                      label_separator: str = "|", n_folds: int = 3,
                      seed: int = 0) -> LabelQualityReport:
    """Rank documents whose given label the out-of-fold model distrusts.

    Suspects are sorted by ascending self-confidence (worst first) and the
    stored list is capped at ``MAX_SUSPECTS``. Corpora too small for
    cross-validation or with a single class return an empty report with an
    explanatory note instead of crashing.
    """
    if task_type not in VALID_TASK_TYPES:
        raise ValueError(f"task_type must be one of {VALID_TASK_TYPES}, got '{task_type}'")
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    texts = [str(t) for t in texts]
    n_docs = len(texts)
    if len(raw_labels) != n_docs:
        raise ValueError("texts and raw_labels must have the same length")

    if n_docs < MIN_DOCS_PER_FOLD * n_folds:
        return _empty_report(n_docs, [
            f"too few documents for {n_folds}-fold confident learning "
            f"(need >= {MIN_DOCS_PER_FOLD * n_folds}, got {n_docs})"])

    label_space, y = build_label_space(raw_labels, task_type, separator=label_separator)
    if label_space.n_classes < 2:
        return _empty_report(n_docs, ["confident learning needs >= 2 distinct classes"])

    if label_space.is_multilabel:
        suspects, thresholds, notes = _multilabel_issues(
            texts, y, label_space, raw_labels, n_folds, seed)
    else:
        suspects, thresholds, notes = _multiclass_issues(
            texts, y, label_space, n_folds, seed)

    suspects.sort(key=lambda s: s.self_confidence)
    n_suspects = len(suspects)
    return LabelQualityReport(
        n_docs=n_docs,
        n_suspects=n_suspects,
        suspect_ratio=float(n_suspects / n_docs),
        suspects=suspects[:MAX_SUSPECTS],
        per_class_thresholds=thresholds,
        notes=notes,
    )


def _multiclass_issues(texts: list, y: np.ndarray, label_space: LabelSpace,
                       n_folds: int, seed: int):
    """Confident-learning pass over single-label classes."""
    notes: list = []
    counts = np.bincount(y, minlength=label_space.n_classes)
    min_count = int(counts[counts > 0].min())
    n_splits = min(n_folds, min_count)
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    else:
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        notes.append("a class has a single example; stratified CV infeasible, using KFold")

    proba = cross_val_predict(_oof_pipeline(), texts, y, cv=cv, method="predict_proba")
    self_conf = proba[np.arange(len(y)), y]
    predicted = proba.argmax(axis=1)

    thresholds = {}
    for c, name in enumerate(label_space.classes):
        mask = y == c
        thresholds[name] = float(self_conf[mask].mean()) if mask.any() else 0.0

    suspects = []
    for row in range(len(y)):
        given_idx = int(y[row])
        pred_idx = int(predicted[row])
        confidence = float(self_conf[row])
        if pred_idx != given_idx and confidence < thresholds[label_space.classes[given_idx]]:
            suspects.append(SuspectLabel(
                row=row,
                given=label_space.classes[given_idx],
                suggested=label_space.classes[pred_idx],
                self_confidence=confidence,
            ))
    return suspects, thresholds, notes


def _multilabel_issues(texts: list, Y: np.ndarray, label_space: LabelSpace,
                       raw_labels: Sequence[str], n_folds: int, seed: int):
    """Per-label binary confident-learning pass (see module docstring)."""
    notes: list = ["multilabel simplification: each label column scored as an "
                   "independent binary problem"]
    raw = [str(r) for r in raw_labels]
    suspects: list = []
    thresholds: dict = {}

    for col, name in enumerate(label_space.classes):
        target = Y[:, col]
        n_pos = int(target.sum())
        n_neg = len(target) - n_pos
        if n_pos == 0 or n_neg == 0:
            notes.append(f"label '{name}' skipped: needs both positive and negative examples")
            continue
        n_splits = min(n_folds, n_pos, n_neg)
        if n_splits < 2:
            notes.append(f"label '{name}' skipped: not enough examples on both sides for CV")
            continue

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        proba = cross_val_predict(_oof_pipeline(), texts, target, cv=cv,
                                  method="predict_proba")[:, 1]
        pos_mean = float(proba[target == 1].mean())
        neg_mean = float(proba[target == 0].mean())
        thresholds[name] = {"positive_mean": pos_mean, "negative_mean": neg_mean}

        for row in range(len(target)):
            p = float(proba[row])
            if target[row] == 1 and p < pos_mean and p < 0.5:
                suspects.append(SuspectLabel(row=row, given=raw[row],
                                             suggested=f"-{name}", self_confidence=p))
            elif target[row] == 0 and p > neg_mean and p > 0.5:
                suspects.append(SuspectLabel(row=row, given=raw[row],
                                             suggested=f"+{name}", self_confidence=1.0 - p))
    return suspects, thresholds, notes
