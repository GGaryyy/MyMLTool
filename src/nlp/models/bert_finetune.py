"""Chinese-BERT fine-tuning classifier for 公文 text classification.

Wraps ``transformers.AutoModelForSequenceClassification`` behind the shared
:class:`~src.nlp.models.base.TextClassifier` interface. The default backbone
is ``google-bert/bert-base-chinese`` (Apache-2.0, Google) — a
license-compliant, non-China-origin model (see docs/nlp/LICENSES.md). A local
directory can be passed via ``ModelConfig.pretrained_path`` for offline /
air-gapped deployment (set ``HF_HUB_OFFLINE=1``).

Documents longer than ``max_length`` tokens are TRUNCATED. The EDA report's
">512 token ratio" tells you whether that loses signal; sliding-window
chunking is a documented future extension.

transformers and torch are imported at module top because this module is only
imported lazily (via the registry) when a BERT model is requested.
"""

import time
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.nlp.config import DeviceConfig, ModelConfig
from src.nlp.device import detect_device, seed_everything
from src.nlp.labels import LabelSpace
from src.nlp.metrics import compute_metrics
from src.nlp.models.base import FAMILY_PRETRAINED, FitReport, TextClassifier

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
DEFAULT_BERT_MODEL = "google-bert/bert-base-chinese"
_AUTOCAST_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


class _EncodedDataset(Dataset):
    def __init__(self, encodings: dict, targets: np.ndarray):
        self.encodings = encodings
        self.targets = targets

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        return item, self.targets[idx]


class BertFinetuneClassifier(TextClassifier):
    """Fine-tune a Chinese BERT sequence classifier."""

    name = "bert"
    family = FAMILY_PRETRAINED

    def __init__(self):
        super().__init__()
        self.tokenizer = None
        self.module = None
        self._device = None
        self._precision = "fp32"

    def build(self, label_space: LabelSpace, model_config: ModelConfig,
              device_config: Optional[DeviceConfig] = None) -> None:
        super().build(label_space, model_config, device_config)
        self.tokenizer = None
        self.module = None

    def _model_source(self) -> str:
        return self.model_config.pretrained_path or DEFAULT_BERT_MODEL

    def fit(self, texts: Sequence[str], y: np.ndarray,
            val_texts: Optional[Sequence[str]] = None,
            val_y: Optional[np.ndarray] = None) -> FitReport:
        self._require_built()
        texts = list(texts)
        y = np.asarray(y)
        seed_everything(int(self.model_config.params.get("seed", 0)))

        info = detect_device(self.device_config)
        device = torch.device(info.resolved_device)
        self._device = device
        self._precision = info.precision

        source = self._model_source()
        problem_type = "multi_label_classification" if self.label_space.is_multilabel else "single_label_classification"
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        # ignore_mismatched_sizes: a base backbone has no classification head, so
        # from_pretrained builds a fresh num_labels head; if the checkpoint DID
        # carry a differently-sized head, reinitialise it rather than error.
        self.module = AutoModelForSequenceClassification.from_pretrained(
            source, num_labels=self.label_space.n_classes, problem_type=problem_type,
            ignore_mismatched_sizes=True,
        ).to(device)

        loader = self._make_loader(texts, y, shuffle=True)
        optimizer = torch.optim.AdamW(self.module.parameters(), lr=self.model_config.learning_rate)
        autocast_dtype = _AUTOCAST_DTYPES.get(self._precision) if device.type == "cuda" else None

        history = []
        start = time.perf_counter()
        for epoch in range(self.model_config.epochs):
            epoch_loss = self._train_epoch(loader, optimizer, device, autocast_dtype)
            entry = {"epoch": epoch + 1, "train_loss": epoch_loss}
            if val_texts is not None and val_y is not None and len(val_texts) > 0:
                entry["val_f1_macro"] = self._f1_macro(list(val_texts), np.asarray(val_y))
            history.append(entry)
        train_seconds = time.perf_counter() - start

        notes = {}
        if int(self.model_config.max_length) < 512:
            notes["max_length"] = f"truncating documents to {self.model_config.max_length} tokens"
        return FitReport(
            model_name=self.name,
            family=self.family,
            n_epochs=self.model_config.epochs,
            train_seconds=train_seconds,
            history=history,
            device=info.resolved_device,
            precision=self._precision,
            notes=notes,
        )

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        self._require_built()
        if self.module is None:
            raise RuntimeError(f"{type(self).__name__}: call fit() before predict")
        texts = list(texts)
        encodings = self._encode(texts)
        targets = np.zeros(len(texts), dtype=np.int64)
        loader = DataLoader(
            _EncodedDataset(encodings, targets),
            batch_size=self.model_config.batch_size, shuffle=False, collate_fn=_collate_encodings,
        )
        self.module.eval()
        outputs = []
        with torch.no_grad():
            for batch, _ in loader:
                batch = {k: v.to(self._device) for k, v in batch.items()}
                logits = self.module(**batch).logits.float()
                if self.label_space.is_multilabel:
                    outputs.append(torch.sigmoid(logits).cpu().numpy())
                else:
                    outputs.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(outputs, axis=0)

    def save(self, path: str) -> None:
        self._require_built()
        if self.module is None:
            raise RuntimeError(f"{type(self).__name__}: call fit() before save")
        self.module.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path: str) -> None:
        self._require_built()
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.module = AutoModelForSequenceClassification.from_pretrained(path)
        self._device = torch.device("cpu")

    # ---- internals ---- #
    def _encode(self, texts: list) -> dict:
        enc = self.tokenizer(
            texts, truncation=True, padding=True,
            max_length=int(self.model_config.max_length), return_tensors="pt",
        )
        return dict(enc)

    def _make_loader(self, texts, y, shuffle):
        encodings = self._encode(texts)
        return DataLoader(
            _EncodedDataset(encodings, y),
            batch_size=self.model_config.batch_size, shuffle=shuffle, collate_fn=_collate_encodings,
        )

    def _train_epoch(self, loader, optimizer, device, autocast_dtype):
        self.module.train()
        total = 0.0
        n_batches = 0
        for batch, targets in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = self._prepare_labels(targets, device)
            optimizer.zero_grad()
            if autocast_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    loss = self.module(**batch, labels=labels).loss
            else:
                loss = self.module(**batch, labels=labels).loss
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu())
            n_batches += 1
        return total / max(n_batches, 1)

    def _prepare_labels(self, targets, device):
        if self.label_space.is_multilabel:
            return targets.to(device).float()
        return targets.to(device).long()

    def _f1_macro(self, texts, y) -> float:
        metrics = compute_metrics(y, self.predict(texts), self.label_space)
        return float(metrics["f1_macro"])


def _collate_encodings(batch):
    items, targets = zip(*batch)
    keys = items[0].keys()
    collated = {k: torch.stack([torch.as_tensor(item[k]) for item in items]) for k in keys}
    targets = np.stack(targets) if isinstance(targets[0], np.ndarray) else np.asarray(targets)
    return collated, torch.from_numpy(targets)
