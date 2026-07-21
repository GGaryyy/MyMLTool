"""BiLSTM + additive attention classifier for 公文 text classification.

Character-embedding -> bidirectional LSTM -> additive (Bahdanau-style)
attention pooling over time (padding masked out) -> dropout -> linear head.
Shares training/persistence machinery with
:class:`src.nlp.models._torch_base.TorchTextClassifier`.
"""

import torch
import torch.nn as nn

from src.nlp.models._torch_base import PAD_ID, TorchTextClassifier

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
DEFAULT_HIDDEN_DIM = 128
DEFAULT_N_LAYERS = 1
DEFAULT_DROPOUT = 0.3


class _BiLstmAttnModule(nn.Module):
    """Embedding -> BiLSTM -> masked additive attention -> linear."""

    def __init__(self, vocab_size: int, n_classes: int, embed_dim: int,
                 hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=n_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, n_classes)

    def forward(self, x, lengths=None):
        mask = x != PAD_ID  # (batch, seq)
        outputs, _ = self.lstm(self.embedding(x))  # (batch, seq, 2*hidden)
        scores = self.attention(outputs).squeeze(-1)  # (batch, seq)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (batch, seq, 1)
        context = (outputs * weights).sum(dim=1)  # (batch, 2*hidden)
        return self.fc(self.dropout(context))


class BiLstmAttnClassifier(TorchTextClassifier):
    """BiLSTM with additive attention over character embeddings."""

    name = "bilstm_attn"

    def _build_module(self, vocab_size: int, n_classes: int, embed_dim: int) -> nn.Module:
        params = self.model_config.params
        return _BiLstmAttnModule(
            vocab_size=vocab_size,
            n_classes=n_classes,
            embed_dim=embed_dim,
            hidden_dim=params.get("hidden_dim", DEFAULT_HIDDEN_DIM),
            n_layers=params.get("n_layers", DEFAULT_N_LAYERS),
            dropout=params.get("dropout", DEFAULT_DROPOUT),
        )
