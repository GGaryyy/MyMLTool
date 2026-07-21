"""TF-IDF + classic sklearn baselines for Chinese text classification.

One class covers four estimator variants (logistic regression, linear SVM,
multinomial naive Bayes, decision tree) behind the shared
:class:`~src.nlp.models.base.TextClassifier` interface. Features are
CHARACTER 1-2 grams on purpose: char n-grams are a strong baseline for
Chinese text and need no word segmenter, so this family carries zero
segmentation dependencies.

LinearSVC exposes no ``predict_proba``; its ``decision_function`` margins
are mapped through a softmax (multiclass) or per-column sigmoid
(multilabel OvR scores). That mapping is an UNCALIBRATED approximation —
fine for ranking and thresholding, not for reading off true probabilities.

Persistence uses :mod:`joblib` (already a scikit-learn dependency): the
fitted sklearn pipeline plus the variant string are dumped as one payload.
"""

import time
import warnings
from typing import Optional, Sequence

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from src.nlp.config import DeviceConfig, ModelConfig
from src.nlp.labels import LabelSpace
from src.nlp.metrics import compute_metrics
from src.nlp.models.base import FAMILY_BASELINE, FitReport, TextClassifier

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
VARIANTS = ("logreg", "linearsvm", "nb", "tree")
TFIDF_ANALYZER = "char"  # char n-grams: strong for Chinese, no segmenter needed
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MAX_FEATURES = 50000
TFIDF_SUBLINEAR_TF = True
LOGREG_MAX_ITER = 1000
DEFAULT_C = 1.0
DEFAULT_NB_ALPHA = 1.0
NB_CLASS_WEIGHT_NOTE = "unsupported for nb; ignored"


class TfidfLinearClassifier(TextClassifier):
    """Char-TFIDF + {logreg, linearsvm, nb, tree} baseline classifier."""

    family = FAMILY_BASELINE

    def __init__(self, variant: str = "logreg"):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got '{variant}'")
        self.variant = variant
        self.name = f"tfidf_{variant}"
        self.family = FAMILY_BASELINE
        self.pipeline: Optional[Pipeline] = None
        self._notes: dict = {}

    def build(self, label_space: LabelSpace, model_config: ModelConfig,
              device_config: Optional[DeviceConfig] = None) -> None:
        """Construct the TF-IDF + estimator pipeline for this variant.

        Multilabel wraps the estimator in ``OneVsRestClassifier`` so every
        variant accepts indicator-matrix targets.
        """
        super().build(label_space, model_config, device_config)
        self._notes = {}
        estimator = self._make_estimator()
        if self.label_space.is_multilabel:
            estimator = OneVsRestClassifier(estimator)
        self.pipeline = Pipeline([
            ("tfidf", _make_vectorizer()),
            ("clf", estimator),
        ])

    def fit(self, texts: Sequence[str], y: np.ndarray,
            val_texts: Optional[Sequence[str]] = None,
            val_y: Optional[np.ndarray] = None) -> FitReport:
        """Fit the pipeline once (n_epochs=1) and report train/val macro-F1."""
        self._require_built()
        texts = list(texts)
        y = np.asarray(y)

        start = time.perf_counter()
        self.pipeline.fit(texts, y)
        train_seconds = time.perf_counter() - start

        entry = {"epoch": 1, "train_f1_macro": self._f1_macro(texts, y)}
        if val_texts is not None and val_y is not None and len(val_texts) > 0:
            entry["val_f1_macro"] = self._f1_macro(list(val_texts), np.asarray(val_y))
        return FitReport(
            model_name=self.name,
            family=self.family,
            n_epochs=1,
            train_seconds=train_seconds,
            history=[entry],
            device="cpu",  # sklearn baselines always run on CPU
            precision="fp32",
            notes=dict(self._notes),
        )

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Return ``(n, n_classes)`` scores in [0, 1].

        logreg / nb / tree use native ``predict_proba``. linearsvm maps
        ``decision_function`` margins through softmax (multiclass) or
        per-column sigmoid (multilabel) — uncalibrated, ranking/threshold
        use only. Multiclass output is renormalized defensively (OvR-style
        score matrices need it); all-zero rows fall back to uniform.
        Multilabel scores are NEVER renormalized — per-class probabilities
        are independent.
        """
        self._require_built()
        if self.pipeline is None:
            raise RuntimeError(f"{type(self).__name__}: call build() before predict")
        texts = list(texts)
        if self.variant == "linearsvm":
            return self._svm_proba(texts)
        proba = np.asarray(self.pipeline.predict_proba(texts), dtype=float)
        if self.label_space.is_multilabel:
            return proba  # independent per-class probabilities from OvR
        proba = _expand_to_n_classes(proba, self._fitted_classes(), self.label_space.n_classes)
        return _normalize_rows(proba)

    def save(self, path: str) -> None:
        """joblib-dump ``{"variant", "pipeline"}`` (see module docstring)."""
        self._require_built()
        if self.pipeline is None:
            raise RuntimeError(f"{type(self).__name__}: nothing to save; call build() first")
        joblib.dump({"variant": self.variant, "pipeline": self.pipeline}, path)

    def load(self, path: str) -> None:
        """Restore a payload written by :meth:`save` into this instance.

        Call ``build()`` first so label space / config are attached — the
        payload only carries the fitted pipeline and the variant string.
        """
        payload = joblib.load(path)
        if not isinstance(payload, dict) or "pipeline" not in payload or "variant" not in payload:
            raise ValueError(f"Not a TfidfLinearClassifier checkpoint: {path}")
        if payload["variant"] != self.variant:
            raise ValueError(
                f"Checkpoint variant '{payload['variant']}' does not match "
                f"this instance ('{self.variant}')"
            )
        self.pipeline = payload["pipeline"]

    def _make_estimator(self):
        """Instantiate the sklearn estimator for ``self.variant``."""
        params = self.model_config.params
        balanced = self.model_config.class_weight == "balanced"
        class_weight = "balanced" if balanced else None
        if self.variant == "logreg":
            return LogisticRegression(
                max_iter=LOGREG_MAX_ITER,
                C=params.get("C", DEFAULT_C),
                class_weight=class_weight,
            )
        if self.variant == "linearsvm":
            return LinearSVC(C=params.get("C", DEFAULT_C), class_weight=class_weight)
        if self.variant == "nb":
            if balanced:
                warnings.warn(
                    "MultinomialNB has no class_weight support; 'balanced' is ignored",
                    UserWarning,
                )
                self._notes["class_weight"] = NB_CLASS_WEIGHT_NOTE
            return MultinomialNB(alpha=params.get("alpha", DEFAULT_NB_ALPHA))
        return DecisionTreeClassifier(
            max_depth=params.get("max_depth", None),
            class_weight=class_weight,
        )

    def _svm_proba(self, texts: list) -> np.ndarray:
        """Map LinearSVC margins to pseudo-probabilities (uncalibrated)."""
        scores = np.asarray(self.pipeline.decision_function(texts), dtype=float)
        if self.label_space.is_multilabel:
            if scores.ndim == 1:  # single-class label space edge case
                scores = scores.reshape(-1, 1)
            return _sigmoid(scores)
        if scores.ndim == 1:
            # Binary LinearSVC yields one margin column; mirror it so the
            # softmax sees both classes.
            scores = np.column_stack([-scores, scores])
        proba = _softmax(scores)
        proba = _expand_to_n_classes(proba, self._fitted_classes(), self.label_space.n_classes)
        return _normalize_rows(proba)

    def _fitted_classes(self) -> np.ndarray:
        return np.asarray(self.pipeline.named_steps["clf"].classes_)

    def _f1_macro(self, texts: list, y: np.ndarray) -> float:
        metrics = compute_metrics(y, self.predict(texts), self.label_space)
        return float(metrics["f1_macro"])


def _make_vectorizer() -> TfidfVectorizer:
    """Char 1-2 gram TF-IDF: strong for Chinese, zero segmenter dependency."""
    return TfidfVectorizer(
        analyzer=TFIDF_ANALYZER,
        ngram_range=TFIDF_NGRAM_RANGE,
        max_features=TFIDF_MAX_FEATURES,
        sublinear_tf=TFIDF_SUBLINEAR_TF,
    )


def _softmax(scores: np.ndarray) -> np.ndarray:
    """Row-wise numerically stable softmax."""
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _sigmoid(scores: np.ndarray) -> np.ndarray:
    """Numerically stable element-wise logistic function (numpy only)."""
    out = np.empty_like(scores, dtype=float)
    positive = scores >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-scores[positive]))
    exp_scores = np.exp(scores[~positive])
    out[~positive] = exp_scores / (1.0 + exp_scores)
    return out


def _expand_to_n_classes(proba: np.ndarray, fitted_classes: np.ndarray,
                         n_classes: int) -> np.ndarray:
    """Map estimator proba columns onto the full label space.

    A training split can miss a class entirely; sklearn then emits fewer
    columns than the label space owns. Missing classes get 0.0 so the
    ``(n, n_classes)`` contract always holds.
    """
    if proba.shape[1] == n_classes:
        return proba
    full = np.zeros((proba.shape[0], n_classes), dtype=float)
    for column, cls in enumerate(fitted_classes):
        full[:, int(cls)] = proba[:, column]
    return full


def _normalize_rows(proba: np.ndarray) -> np.ndarray:
    """Renormalize multiclass rows to sum to 1; all-zero rows become uniform."""
    n_classes = proba.shape[1]
    sums = proba.sum(axis=1, keepdims=True)
    zero_rows = sums.ravel() <= 0.0
    if zero_rows.any():
        proba = proba.copy()
        proba[zero_rows, :] = 1.0 / n_classes
        sums = proba.sum(axis=1, keepdims=True)
    return proba / sums
