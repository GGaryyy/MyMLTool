"""Common interface every benchmark model family implements.

The harness only ever talks to :class:`TextClassifier`; multiclass vs
multilabel behaviour is decided by the attached
:class:`~src.nlp.labels.LabelSpace`, never by the model family.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from src.nlp.config import DeviceConfig, ModelConfig
from src.nlp.labels import LabelSpace

FAMILY_BASELINE = "baseline"
FAMILY_LIGHTWEIGHT_DL = "lightweight_dl"
FAMILY_PRETRAINED = "pretrained"


@dataclass
class FitReport:
    """Training summary returned by every ``fit`` call."""

    model_name: str
    family: str
    n_epochs: int = 0
    train_seconds: float = 0.0
    history: list = field(default_factory=list)
    device: str = "cpu"
    precision: str = "fp32"
    notes: dict = field(default_factory=dict)


class TextClassifier(ABC):
    """Abstract base for all pluggable text classifiers."""

    name: str = "base"
    family: str = "base"

    def __init__(self):
        self.label_space: Optional[LabelSpace] = None
        self.model_config: Optional[ModelConfig] = None
        self.device_config: Optional[DeviceConfig] = None

    def build(self, label_space: LabelSpace, model_config: ModelConfig,
              device_config: Optional[DeviceConfig] = None) -> None:
        """Attach label space and configuration before training.

        Subclasses extend this to construct their underlying model, calling
        ``super().build(...)`` first.
        """
        if not isinstance(label_space, LabelSpace):
            raise TypeError(f"label_space must be a LabelSpace, got {type(label_space).__name__}")
        if not isinstance(model_config, ModelConfig):
            raise TypeError(f"model_config must be a ModelConfig, got {type(model_config).__name__}")
        self.label_space = label_space
        self.model_config = model_config
        self.device_config = device_config or DeviceConfig()

    @abstractmethod
    def fit(self, texts: Sequence[str], y: np.ndarray,
            val_texts: Optional[Sequence[str]] = None,
            val_y: Optional[np.ndarray] = None) -> FitReport:
        """Train on encoded labels ``y`` (indices or indicator matrix)."""

    @abstractmethod
    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Return ``(n, n_classes)`` scores in [0, 1].

        Multiclass rows sum to 1; multilabel entries are independent
        per-class probabilities.
        """

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        """Shared decision rule: argmax (multiclass) or threshold (multilabel)."""
        self._require_built()
        proba = self.predict_proba(texts)
        if self.label_space.is_multilabel:
            return (proba >= self.model_config.threshold).astype(np.int64)
        return np.argmax(proba, axis=1).astype(np.int64)

    def save(self, path: str) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not implement save()")

    def load(self, path: str) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not implement load()")

    def _require_built(self) -> None:
        if self.label_space is None or self.model_config is None:
            raise RuntimeError(f"{type(self).__name__}: call build() before fit/predict")
