"""Frozen sentence-embedding + linear classifier for Chinese text.

Encodes documents once with a frozen multilingual sentence-transformer, then
fits a cheap linear head (LogisticRegression, one-vs-rest for multilabel).
This is often the strongest option when labelled data is scarce, because the
embedding backbone already carries language knowledge and only the small head
is trained.

Default backbone: ``paraphrase-multilingual-MiniLM-L12-v2`` (Apache-2.0,
UKP-Lab). Pass a local dir via ``ModelConfig.pretrained_path`` for offline
use. sentence-transformers is imported lazily so the module itself imports
without it; tests inject a fake encoder.
"""

import time
from typing import Optional, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from src.nlp.config import DeviceConfig, ModelConfig
from src.nlp.device import detect_device
from src.nlp.labels import LabelSpace
from src.nlp.metrics import compute_metrics
from src.nlp.models.base import FAMILY_PRETRAINED, FitReport, TextClassifier

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
DEFAULT_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LOGREG_MAX_ITER = 1000


class SentEmbedClassifier(TextClassifier):
    """Frozen sentence embeddings + logistic-regression head."""

    name = "sent_embed"
    family = FAMILY_PRETRAINED

    def __init__(self):
        super().__init__()
        self.encoder = None
        self.head = None
        self._device_str = "cpu"

    def build(self, label_space: LabelSpace, model_config: ModelConfig,
              device_config: Optional[DeviceConfig] = None) -> None:
        super().build(label_space, model_config, device_config)
        self.encoder = None
        self.head = None

    def set_encoder(self, encoder) -> None:
        """Inject an encoder exposing ``encode(list[str]) -> np.ndarray``.

        Used by tests to avoid a model download; production builds create the
        real SentenceTransformer lazily in :meth:`fit`.
        """
        self.encoder = encoder

    def _ensure_encoder(self):
        if self.encoder is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sent_embed needs sentence-transformers (see requirements-nlp.txt)"
            ) from exc
        info = detect_device(self.device_config)
        self._device_str = info.resolved_device
        source = self.model_config.pretrained_path or self.model_config.params.get(
            "embed_model", DEFAULT_EMBED_MODEL
        )
        self.encoder = SentenceTransformer(source, device=info.resolved_device)

    def fit(self, texts: Sequence[str], y: np.ndarray,
            val_texts: Optional[Sequence[str]] = None,
            val_y: Optional[np.ndarray] = None) -> FitReport:
        self._require_built()
        texts = list(texts)
        y = np.asarray(y)
        self._ensure_encoder()

        start = time.perf_counter()
        embeddings = self._encode(texts)
        estimator = LogisticRegression(
            max_iter=LOGREG_MAX_ITER,
            class_weight="balanced" if self.model_config.class_weight == "balanced" else None,
        )
        self.head = OneVsRestClassifier(estimator) if self.label_space.is_multilabel else estimator
        self.head.fit(embeddings, y)
        train_seconds = time.perf_counter() - start

        entry = {"epoch": 1, "train_f1_macro": self._f1_macro(texts, y)}
        if val_texts is not None and val_y is not None and len(val_texts) > 0:
            entry["val_f1_macro"] = self._f1_macro(list(val_texts), np.asarray(val_y))
        return FitReport(
            model_name=self.name, family=self.family, n_epochs=1,
            train_seconds=train_seconds, history=[entry],
            device=self._device_str, precision="fp32", notes={},
        )

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        self._require_built()
        if self.head is None:
            raise RuntimeError(f"{type(self).__name__}: call fit() before predict")
        embeddings = self._encode(list(texts))
        if self.label_space.is_multilabel:
            return np.asarray(self.head.predict_proba(embeddings), dtype=float)
        proba = np.asarray(self.head.predict_proba(embeddings), dtype=float)
        return _expand_to_n_classes(proba, np.asarray(self.head.classes_), self.label_space.n_classes)

    def _encode(self, texts: list) -> np.ndarray:
        vectors = self.encoder.encode(texts)
        return np.asarray(vectors, dtype=float)

    def _f1_macro(self, texts, y) -> float:
        metrics = compute_metrics(y, self.predict(texts), self.label_space)
        return float(metrics["f1_macro"])


def _expand_to_n_classes(proba: np.ndarray, fitted_classes: np.ndarray, n_classes: int) -> np.ndarray:
    """Pad proba columns to the full label space (missing classes -> 0)."""
    if proba.shape[1] == n_classes:
        return proba
    full = np.zeros((proba.shape[0], n_classes), dtype=float)
    for column, cls in enumerate(fitted_classes):
        full[:, int(cls)] = proba[:, column]
    return full
