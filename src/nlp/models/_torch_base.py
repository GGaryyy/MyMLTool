"""Shared PyTorch scaffolding for the lightweight deep text classifiers.

TextCNN and BiLSTM+Attention differ only in their ``nn.Module``; everything
else — char vocabulary, padding/batching, the training loop, mixed-precision
autocast, class-weight handling and persistence — lives here in
:class:`TorchTextClassifier`. torch is imported at module top because this
module is only ever imported lazily (via the registry) when a torch model is
actually requested, so the pure-Python core stays torch-free.

Tokenisation is CHARACTER level: no word segmenter, hence no jieba/pkuseg
(China-origin, prohibited) and no segmentation dependency at all.
"""

import time
from abc import abstractmethod
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.nlp.config import DeviceConfig, ModelConfig
from src.nlp.device import detect_device, seed_everything
from src.nlp.labels import LabelSpace
from src.nlp.metrics import compute_metrics
from src.nlp.models.base import FAMILY_LIGHTWEIGHT_DL, FitReport, TextClassifier

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
PAD_ID = 0
UNK_ID = 1
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
DEFAULT_VOCAB_SIZE = 30000
DEFAULT_MIN_FREQ = 1
DEFAULT_EMBED_DIM = 128
_AUTOCAST_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


class _TextIdDataset(Dataset):
    """Wraps encoded id-lists + targets for a DataLoader."""

    def __init__(self, encoded: list, targets: np.ndarray):
        self.encoded = encoded
        self.targets = targets

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, idx):
        return self.encoded[idx], self.targets[idx]


def build_char_vocab(texts: Sequence[str], max_size: int = DEFAULT_VOCAB_SIZE,
                     min_freq: int = DEFAULT_MIN_FREQ) -> dict:
    """Build a ``char -> id`` map (0=pad, 1=unk) from training texts.

    Characters are ranked by descending frequency and truncated to
    ``max_size`` (including the two special tokens).
    """
    counts: dict = {}
    for text in texts:
        for ch in str(text):
            counts[ch] = counts.get(ch, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    vocab = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
    for ch, freq in ordered:
        if freq < min_freq:
            break
        if len(vocab) >= max_size:
            break
        vocab[ch] = len(vocab)
    return vocab


def encode_texts(texts: Sequence[str], vocab: dict, max_length: int) -> list:
    """Encode texts to truncated id-lists; empty texts become a single pad."""
    encoded = []
    for text in texts:
        ids = [vocab.get(ch, UNK_ID) for ch in str(text)[:max_length]]
        if not ids:
            ids = [PAD_ID]
        encoded.append(ids)
    return encoded


def _collate(batch):
    """Pad a batch of (ids, target) to the batch's max length."""
    seqs, targets = zip(*batch)
    max_len = max(len(s) for s in seqs)
    padded = torch.full((len(seqs), max_len), PAD_ID, dtype=torch.long)
    lengths = torch.empty(len(seqs), dtype=torch.long)
    for i, seq in enumerate(seqs):
        padded[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        lengths[i] = len(seq)
    targets = np.stack(targets) if isinstance(targets[0], np.ndarray) else np.asarray(targets)
    return padded, lengths, torch.from_numpy(targets)


class TorchTextClassifier(TextClassifier):
    """Base class: char-embedding + a subclass ``nn.Module`` head.

    Subclasses implement :meth:`_build_module`. Multiclass uses cross-entropy
    over class logits; multilabel uses ``BCEWithLogitsLoss`` over per-label
    logits, selected purely from ``label_space.is_multilabel``.
    """

    family = FAMILY_LIGHTWEIGHT_DL

    def __init__(self):
        super().__init__()
        self.vocab: Optional[dict] = None
        self.module: Optional[nn.Module] = None
        self._device = None
        self._precision = "fp32"

    @abstractmethod
    def _build_module(self, vocab_size: int, n_classes: int, embed_dim: int) -> nn.Module:
        """Return an nn.Module mapping (batch, seq, lengths) -> (batch, n_classes) logits."""

    def build(self, label_space: LabelSpace, model_config: ModelConfig,
              device_config: Optional[DeviceConfig] = None) -> None:
        super().build(label_space, model_config, device_config)
        self.vocab = None
        self.module = None

    def fit(self, texts: Sequence[str], y: np.ndarray,
            val_texts: Optional[Sequence[str]] = None,
            val_y: Optional[np.ndarray] = None) -> FitReport:
        self._require_built()
        texts = list(texts)
        y = np.asarray(y)
        seed_everything(self._seed())

        info = detect_device(self.device_config)
        device = torch.device(info.resolved_device)
        self._device = device
        self._precision = info.precision
        embed_dim = int(self.model_config.params.get("embed_dim", DEFAULT_EMBED_DIM))
        max_vocab = int(self.model_config.params.get("vocab_size", DEFAULT_VOCAB_SIZE))

        self.vocab = build_char_vocab(texts, max_size=max_vocab)
        self.module = self._build_module(len(self.vocab), self.label_space.n_classes, embed_dim).to(device)

        loader = self._make_loader(texts, y, shuffle=True)
        criterion = self._make_criterion(y, device)
        optimizer = torch.optim.AdamW(self.module.parameters(), lr=self.model_config.learning_rate)
        autocast_dtype = _AUTOCAST_DTYPES.get(self._precision) if device.type == "cuda" else None

        history = []
        start = time.perf_counter()
        for epoch in range(self.model_config.epochs):
            epoch_loss = self._train_epoch(loader, criterion, optimizer, device, autocast_dtype)
            entry = {"epoch": epoch + 1, "train_loss": epoch_loss}
            if val_texts is not None and val_y is not None and len(val_texts) > 0:
                entry["val_f1_macro"] = self._f1_macro(list(val_texts), np.asarray(val_y))
            history.append(entry)
        train_seconds = time.perf_counter() - start

        return FitReport(
            model_name=self.name,
            family=self.family,
            n_epochs=self.model_config.epochs,
            train_seconds=train_seconds,
            history=history,
            device=info.resolved_device,
            precision=self._precision,
            notes={},
        )

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        self._require_built()
        if self.module is None:
            raise RuntimeError(f"{type(self).__name__}: call fit() before predict")
        texts = list(texts)
        encoded = encode_texts(texts, self.vocab, self.model_config.max_length)
        targets = np.zeros(len(texts), dtype=np.int64)  # unused placeholder
        loader = DataLoader(
            _TextIdDataset(encoded, targets),
            batch_size=self.model_config.batch_size,
            shuffle=False,
            collate_fn=_collate,
        )
        self.module.eval()
        outputs = []
        with torch.no_grad():
            for padded, lengths, _ in loader:
                padded = padded.to(self._device)
                logits = self.module(padded, lengths.to(self._device)).float()
                if self.label_space.is_multilabel:
                    outputs.append(torch.sigmoid(logits).cpu().numpy())
                else:
                    outputs.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(outputs, axis=0)

    def save(self, path: str) -> None:
        self._require_built()
        if self.module is None:
            raise RuntimeError(f"{type(self).__name__}: call fit() before save")
        torch.save(
            {"state_dict": self.module.state_dict(), "vocab": self.vocab,
             "name": self.name},
            path,
        )

    def load(self, path: str) -> None:
        """Restore weights + vocab; call build() first to attach config.

        ``weights_only=True`` refuses to unpickle arbitrary objects — the
        payload is only tensors plus a ``str -> int`` vocab dict, so safe
        deserialization is sufficient and avoids the pickle RCE surface.
        """
        self._require_built()
        payload = torch.load(path, map_location="cpu", weights_only=True)
        self.vocab = payload["vocab"]
        embed_dim = int(self.model_config.params.get("embed_dim", DEFAULT_EMBED_DIM))
        self.module = self._build_module(len(self.vocab), self.label_space.n_classes, embed_dim)
        self.module.load_state_dict(payload["state_dict"])
        self._device = torch.device("cpu")

    # ---- internals ---- #
    def _make_loader(self, texts, y, shuffle):
        encoded = encode_texts(texts, self.vocab, self.model_config.max_length)
        return DataLoader(
            _TextIdDataset(encoded, y),
            batch_size=self.model_config.batch_size,
            shuffle=shuffle,
            collate_fn=_collate,
        )

    def _make_criterion(self, y, device):
        balanced = self.model_config.class_weight == "balanced"
        if self.label_space.is_multilabel:
            pos_weight = None
            if balanced:
                pos = y.sum(axis=0).astype(float)
                neg = y.shape[0] - pos
                pos_weight = torch.tensor(
                    np.divide(neg, np.maximum(pos, 1.0)), dtype=torch.float, device=device
                )
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        weight = None
        if balanced:
            weight = torch.tensor(_class_weights(y, self.label_space.n_classes),
                                  dtype=torch.float, device=device)
        return nn.CrossEntropyLoss(weight=weight)

    def _train_epoch(self, loader, criterion, optimizer, device, autocast_dtype):
        self.module.train()
        total = 0.0
        n_batches = 0
        for padded, lengths, targets in loader:
            padded = padded.to(device)
            lengths = lengths.to(device)
            targets = self._prepare_targets(targets, device)
            optimizer.zero_grad()
            if autocast_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    logits = self.module(padded, lengths)
                    loss = criterion(logits.float(), targets)
            else:
                logits = self.module(padded, lengths)
                loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu())
            n_batches += 1
        return total / max(n_batches, 1)

    def _prepare_targets(self, targets, device):
        if self.label_space.is_multilabel:
            return targets.to(device).float()
        return targets.to(device).long()

    def _f1_macro(self, texts, y) -> float:
        metrics = compute_metrics(y, self.predict(texts), self.label_space)
        return float(metrics["f1_macro"])

    def _seed(self) -> int:
        return int(self.model_config.params.get("seed", 0))


def _class_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Inverse-frequency class weights (balanced), robust to absent classes."""
    counts = np.bincount(y.astype(int), minlength=n_classes).astype(float)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (n_classes * counts)
    return weights
