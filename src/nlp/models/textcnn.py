"""TextCNN classifier for Chinese text classification (Kim, 2014).

Character-embedding + parallel 1-D convolutions over several kernel widths,
max-over-time pooling, dropout and a linear head. Lightweight and fast to
train on CPU or a single GPU. Shares all training/persistence machinery with
:class:`src.nlp.models._torch_base.TorchTextClassifier`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.nlp.models._torch_base import PAD_ID, TorchTextClassifier

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
DEFAULT_KERNEL_SIZES = (2, 3, 4)
DEFAULT_N_FILTERS = 100
DEFAULT_DROPOUT = 0.5


class _TextCnnModule(nn.Module):
    """Embedding -> multi-width Conv1d -> max-pool -> dropout -> linear."""

    def __init__(self, vocab_size: int, n_classes: int, embed_dim: int,
                 kernel_sizes, n_filters: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, n_filters, kernel_size=k) for k in kernel_sizes]
        )
        self.kernel_sizes = tuple(kernel_sizes)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(kernel_sizes), n_classes)

    def forward(self, x, lengths=None):
        # x: (batch, seq) -> embed (batch, embed_dim, seq) for Conv1d
        embedded = self.embedding(x).transpose(1, 2)
        # Pad so sequences shorter than the widest kernel still convolve.
        min_len = max(self.kernel_sizes)
        if embedded.size(2) < min_len:
            embedded = F.pad(embedded, (0, min_len - embedded.size(2)))
        pooled = [F.relu(conv(embedded)).max(dim=2).values for conv in self.convs]
        features = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(features)


class TextCnnClassifier(TorchTextClassifier):
    """TextCNN over character embeddings."""

    name = "textcnn"

    def _build_module(self, vocab_size: int, n_classes: int, embed_dim: int) -> nn.Module:
        params = self.model_config.params
        return _TextCnnModule(
            vocab_size=vocab_size,
            n_classes=n_classes,
            embed_dim=embed_dim,
            kernel_sizes=params.get("kernel_sizes", DEFAULT_KERNEL_SIZES),
            n_filters=params.get("n_filters", DEFAULT_N_FILTERS),
            dropout=params.get("dropout", DEFAULT_DROPOUT),
        )
