"""TF-IDF -> TruncatedSVD -> LightGBM baseline for 公文 classification.

Gradient boosting wants dense, reasonably low-dimensional inputs, so the
char TF-IDF matrix (same vectorizer choice as the linear baselines: char
1-2 grams, no segmenter dependency) is compressed with TruncatedSVD before
boosting. lightgbm itself is imported LAZILY inside :meth:`build` so this
module imports fine when the optional dependency is missing.

Persistence mirrors :mod:`src.nlp.models.tfidf_linear`: joblib dump of the
fitted pipeline plus the model name as one payload.
"""

import time
from typing import Optional, Sequence

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from src.nlp.config import DeviceConfig, ModelConfig
from src.nlp.labels import LabelSpace
from src.nlp.metrics import compute_metrics
from src.nlp.models.base import FAMILY_BASELINE, FitReport, TextClassifier

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
TFIDF_ANALYZER = "char"  # char n-grams: strong for Chinese, no segmenter needed
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MAX_FEATURES = 50000
TFIDF_SUBLINEAR_TF = True
DEFAULT_SVD_COMPONENTS = 128
MIN_SVD_COMPONENTS = 2
DEFAULT_N_ESTIMATORS = 200
DEFAULT_GBM_LEARNING_RATE = 0.1
DEFAULT_GBM_SEED = 0
LIGHTGBM_HINT = (
    "tfidf_lightgbm needs the optional dependency 'lightgbm' "
    "(install it via requirements-nlp.txt)"
)


def _import_lightgbm():
    """Import lightgbm lazily; missing/broken installs raise ImportError.

    A missing native OpenMP runtime (``libgomp.so.1``) surfaces as OSError
    from inside lightgbm's shared-library loader, so both failure modes are
    translated into the same actionable ImportError.
    """
    try:
        import lightgbm
    except (ImportError, OSError) as exc:
        raise ImportError(f"{LIGHTGBM_HINT}: {exc}") from exc
    return lightgbm


class TfidfGbmClassifier(TextClassifier):
    """Char-TFIDF + TruncatedSVD + LightGBM baseline."""

    name = "tfidf_lightgbm"
    family = FAMILY_BASELINE

    def __init__(self):
        super().__init__()
        self.pipeline: Optional[Pipeline] = None
        self._notes: dict = {}
        self._requested_svd_components: int = DEFAULT_SVD_COMPONENTS

    def build(self, label_space: LabelSpace, model_config: ModelConfig,
              device_config: Optional[DeviceConfig] = None) -> None:
        """Assemble tfidf -> svd -> LGBMClassifier (OvR for multilabel)."""
        super().build(label_space, model_config, device_config)
        self._notes = {}
        lightgbm = _import_lightgbm()
        params = self.model_config.params
        seed = params.get("seed", DEFAULT_GBM_SEED)
        self._requested_svd_components = int(
            params.get("svd_components", DEFAULT_SVD_COMPONENTS)
        )
        estimator = lightgbm.LGBMClassifier(
            n_estimators=params.get("n_estimators", DEFAULT_N_ESTIMATORS),
            learning_rate=params.get("gbm_learning_rate", DEFAULT_GBM_LEARNING_RATE),
            class_weight="balanced" if self.model_config.class_weight == "balanced" else None,
            verbosity=-1,
            random_state=seed,
        )
        if self.label_space.is_multilabel:
            estimator = OneVsRestClassifier(estimator)
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                analyzer=TFIDF_ANALYZER,
                ngram_range=TFIDF_NGRAM_RANGE,
                max_features=TFIDF_MAX_FEATURES,
                sublinear_tf=TFIDF_SUBLINEAR_TF,
            )),
            ("svd", TruncatedSVD(
                n_components=self._requested_svd_components,
                random_state=seed,
            )),
            ("clf", estimator),
        ])

    def fit(self, texts: Sequence[str], y: np.ndarray,
            val_texts: Optional[Sequence[str]] = None,
            val_y: Optional[np.ndarray] = None) -> FitReport:
        """Fit once; SVD components are clamped to the sample count first.

        TruncatedSVD cannot extract more components than
        ``min(n_features, n_samples)``, so the requested value is clamped to
        ``max(MIN_SVD_COMPONENTS, min(requested, n_samples - 1))`` and any
        adjustment is recorded in ``FitReport.notes["svd_components"]``.
        """
        self._require_built()
        texts = list(texts)
        y = np.asarray(y)

        requested = self._requested_svd_components
        clamped = max(MIN_SVD_COMPONENTS, min(requested, len(texts) - 1))
        self.pipeline.set_params(svd__n_components=clamped)
        self._notes.pop("svd_components", None)
        if clamped != requested:
            self._notes["svd_components"] = (
                f"clamped from {requested} to {clamped} "
                f"(must stay below n_samples={len(texts)})"
            )

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
            device="cpu",  # LightGBM baseline runs on CPU
            precision="fp32",
            notes=dict(self._notes),
        )

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Return ``(n, n_classes)`` scores in [0, 1].

        LGBMClassifier has native ``predict_proba``. Multiclass rows are
        renormalized defensively (all-zero rows become uniform); multilabel
        OvR probabilities stay independent and are never renormalized.
        """
        self._require_built()
        if self.pipeline is None:
            raise RuntimeError(f"{type(self).__name__}: call build() before predict")
        proba = np.asarray(self.pipeline.predict_proba(list(texts)), dtype=float)
        if self.label_space.is_multilabel:
            return proba
        proba = _expand_to_n_classes(
            proba,
            np.asarray(self.pipeline.named_steps["clf"].classes_),
            self.label_space.n_classes,
        )
        return _normalize_rows(proba)

    def save(self, path: str) -> None:
        """joblib-dump ``{"name", "pipeline"}`` (see module docstring)."""
        self._require_built()
        if self.pipeline is None:
            raise RuntimeError(f"{type(self).__name__}: nothing to save; call build() first")
        joblib.dump({"name": self.name, "pipeline": self.pipeline}, path)

    def load(self, path: str) -> None:
        """Restore a payload written by :meth:`save` into this instance.

        Call ``build()`` first so label space / config are attached — the
        payload only carries the fitted pipeline and the model name.
        """
        payload = joblib.load(path)
        if not isinstance(payload, dict) or "pipeline" not in payload or "name" not in payload:
            raise ValueError(f"Not a TfidfGbmClassifier checkpoint: {path}")
        if payload["name"] != self.name:
            raise ValueError(
                f"Checkpoint model '{payload['name']}' does not match this instance ('{self.name}')"
            )
        self.pipeline = payload["pipeline"]

    def _f1_macro(self, texts: list, y: np.ndarray) -> float:
        metrics = compute_metrics(y, self.predict(texts), self.label_space)
        return float(metrics["f1_macro"])


# Kept local instead of importing tfidf_linear's private helpers: the two
# baseline modules stay independently importable and the helpers are small.
def _expand_to_n_classes(proba: np.ndarray, fitted_classes: np.ndarray,
                         n_classes: int) -> np.ndarray:
    """Map estimator proba columns onto the full label space (0.0 for absent)."""
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
