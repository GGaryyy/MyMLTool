"""SetFit few-shot classifier for Chinese text classification.

SetFit contrastively fine-tunes a sentence-transformer on label pairs, then
fits a classification head — strong when only a handful of examples per class
are labelled (NOT an LLM / prompt method). Backbone defaults to
``paraphrase-multilingual-MiniLM-L12-v2`` (Apache-2.0); a local dir via
``ModelConfig.pretrained_path`` supports offline use.

setfit is imported lazily so the module imports without it.
"""

import time
from typing import Optional, Sequence

import numpy as np

from src.nlp.config import DeviceConfig, ModelConfig
from src.nlp.labels import LabelSpace
from src.nlp.metrics import compute_metrics
from src.nlp.models.base import FAMILY_PRETRAINED, FitReport, TextClassifier

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
DEFAULT_SETFIT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_NUM_ITERATIONS = 20


class SetFitClassifier(TextClassifier):
    """SetFit contrastive few-shot classifier."""

    name = "setfit"
    family = FAMILY_PRETRAINED

    def __init__(self):
        super().__init__()
        self.model = None
        self._device_str = "cpu"

    def build(self, label_space: LabelSpace, model_config: ModelConfig,
              device_config: Optional[DeviceConfig] = None) -> None:
        super().build(label_space, model_config, device_config)
        self.model = None

    def _load_model(self):
        try:
            from setfit import SetFitModel
        except ImportError as exc:
            raise ImportError(
                "setfit is not installed (see requirements-nlp.txt)"
            ) from exc
        from src.nlp.device import detect_device

        info = detect_device(self.device_config)
        self._device_str = info.resolved_device
        source = self.model_config.pretrained_path or self.model_config.params.get(
            "embed_model", DEFAULT_SETFIT_MODEL
        )
        kwargs = {}
        if self.label_space.is_multilabel:
            kwargs["multi_target_strategy"] = "one-vs-rest"
        return SetFitModel.from_pretrained(source, **kwargs)

    def fit(self, texts: Sequence[str], y: np.ndarray,
            val_texts: Optional[Sequence[str]] = None,
            val_y: Optional[np.ndarray] = None) -> FitReport:
        self._require_built()
        from setfit import Trainer, TrainingArguments

        texts = list(texts)
        y = np.asarray(y)
        self.model = self._load_model()

        train_labels = _labels_for_setfit(y, self.label_space)
        from datasets import Dataset as HFDataset

        train_ds = HFDataset.from_dict({"text": texts, "label": train_labels})
        args = TrainingArguments(
            batch_size=self.model_config.batch_size,
            num_epochs=self.model_config.epochs,
            num_iterations=int(self.model_config.params.get("num_iterations", DEFAULT_NUM_ITERATIONS)),
        )
        trainer = Trainer(model=self.model, args=args, train_dataset=train_ds)

        start = time.perf_counter()
        trainer.train()
        train_seconds = time.perf_counter() - start

        entry = {"epoch": self.model_config.epochs, "train_f1_macro": self._f1_macro(texts, y)}
        if val_texts is not None and val_y is not None and len(val_texts) > 0:
            entry["val_f1_macro"] = self._f1_macro(list(val_texts), np.asarray(val_y))
        return FitReport(
            model_name=self.name, family=self.family, n_epochs=self.model_config.epochs,
            train_seconds=train_seconds, history=[entry],
            device=self._device_str, precision="fp32", notes={},
        )

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        self._require_built()
        if self.model is None:
            raise RuntimeError(f"{type(self).__name__}: call fit() before predict")
        proba = np.asarray(self.model.predict_proba(list(texts)), dtype=float)
        return proba

    def save(self, path: str) -> None:
        self._require_built()
        if self.model is None:
            raise RuntimeError(f"{type(self).__name__}: call fit() before save")
        self.model.save_pretrained(path)

    def load(self, path: str) -> None:
        self._require_built()
        from setfit import SetFitModel

        self.model = SetFitModel.from_pretrained(path)

    def _f1_macro(self, texts, y) -> float:
        metrics = compute_metrics(y, self.predict(texts), self.label_space)
        return float(metrics["f1_macro"])


def _labels_for_setfit(y: np.ndarray, label_space: LabelSpace) -> list:
    """Convert encoded targets to the label form SetFit's Trainer expects."""
    if label_space.is_multilabel:
        return [row.astype(int).tolist() for row in y]
    return y.astype(int).tolist()
