"""Dataset loading and splitting for the Chinese-text NLP benchmark.

Bridges a labelled CSV (or an in-memory text/label pair) into the
:class:`TextDataset` bundle every model family consumes. The label space is
built on ALL labels BEFORE splitting, so validation/test rows can never
contain classes the encoder has not seen. CSV loading mirrors the error
style of :func:`src.data_prep.load_dataset`; column resolution mirrors
:func:`src.data_prep.split_features_target` (int = positional with the
repo's text-first / label-last convention, str = column name).
"""

import warnings
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.nlp.config import (
    DEFAULT_LABEL_SEPARATOR,
    DEFAULT_SEED,
    DEFAULT_TEST_SIZE,
    DEFAULT_VAL_SIZE,
    RunConfig,
)
from src.nlp.labels import LabelSpace, build_label_space

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
MAX_REPORTED_NAN_ROWS = 10  # how many offending row numbers an error lists


@dataclass
class TextDataset:
    """Train/val/test text splits plus encoded labels and their LabelSpace."""

    texts_train: list
    texts_val: list
    texts_test: list
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    label_space: LabelSpace
    raw_label_col: str
    text_col: str


def load_text_dataset(config: RunConfig) -> TextDataset:
    """Load ``config.data.csv_path`` and split it into a :class:`TextDataset`.

    Raises ``FileNotFoundError`` when the CSV is missing (passthrough, like
    :func:`src.data_prep.load_dataset`) and ``ValueError`` for empty files,
    unknown columns, text/label resolving to the same column, and rows
    whose label is NaN.
    """
    if not isinstance(config, RunConfig):
        raise TypeError(f"config must be a RunConfig, got {type(config).__name__}")
    data = config.data
    df = _load_csv(data.csv_path)
    text_name = _resolve_column(df, data.text_col, "text")
    label_name = _resolve_column(df, data.label_col, "label")
    if text_name == label_name:
        raise ValueError(f"text_col and label_col resolve to the same column '{text_name}'")

    _check_nan_labels(df, label_name)
    texts = [str(t).strip() for t in df[text_name]]
    raw_labels = list(df[label_name])
    return split_text_dataset(
        texts,
        raw_labels,
        task_type=data.task_type,
        label_separator=data.label_separator,
        test_size=data.test_size,
        val_size=data.val_size,
        seed=config.seed,
        text_col=text_name,
        raw_label_col=label_name,
    )


def split_text_dataset(
    texts: Sequence,
    raw_labels: Sequence,
    task_type: str,
    label_separator: str = DEFAULT_LABEL_SEPARATOR,
    test_size: float = DEFAULT_TEST_SIZE,
    val_size: float = DEFAULT_VAL_SIZE,
    seed: int = DEFAULT_SEED,
    text_col: str = "text",
    raw_label_col: str = "label",
) -> TextDataset:
    """Encode labels and split texts into a train/val/test bundle.

    The :class:`LabelSpace` is built on ALL labels first, so every class is
    known to the encoder no matter where its rows land. ``val_size`` is a
    fraction of the WHOLE dataset (rescaled internally for the second
    split); ``val_size=0`` yields empty validation lists/arrays with the
    correct shapes. Multiclass splits are stratified whenever sklearn can
    honour it, otherwise a ``UserWarning`` is emitted and the split falls
    back to non-stratified. Multilabel splits are never stratified (see
    :func:`_stratify_or_none`). Same seed -> identical splits.
    """
    texts = [str(t) for t in texts]
    raw_labels = list(raw_labels)
    if len(texts) != len(raw_labels):
        raise ValueError(
            f"texts and raw_labels differ in length: {len(texts)} vs {len(raw_labels)}"
        )
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")
    if not 0.0 <= val_size < 1.0:
        raise ValueError(f"val_size must be in [0, 1), got {val_size}")
    if test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size must leave room for a training split")

    label_space, y_all = build_label_space(raw_labels, task_type, separator=label_separator)

    indices = np.arange(len(texts))
    stratify_stage1 = _stratify_or_none(y_all, test_size, label_space, "train/test")
    idx_rest, idx_test = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=stratify_stage1
    )

    if val_size > 0.0:
        val_fraction = val_size / (1.0 - test_size)  # rescale to the remainder
        y_rest = y_all[idx_rest]
        stratify_stage2 = _stratify_or_none(y_rest, val_fraction, label_space, "train/val")
        idx_train, idx_val = train_test_split(
            idx_rest, test_size=val_fraction, random_state=seed, stratify=stratify_stage2
        )
    else:
        idx_train = idx_rest
        idx_val = np.array([], dtype=np.int64)

    return TextDataset(
        texts_train=[texts[i] for i in idx_train],
        texts_val=[texts[i] for i in idx_val],
        texts_test=[texts[i] for i in idx_test],
        y_train=y_all[idx_train],
        y_val=y_all[idx_val],
        y_test=y_all[idx_test],
        label_space=label_space,
        raw_label_col=raw_label_col,
        text_col=text_col,
    )


def _load_csv(csv_path: str) -> pd.DataFrame:
    """Read the labelled CSV, mirroring :func:`src.data_prep.load_dataset` errors."""
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Dataset is empty: {csv_path}") from exc

    if df.empty:
        raise ValueError(f"Dataset has no rows: {csv_path}")
    return df


def _resolve_column(df: pd.DataFrame, col: Union[int, str], role: str) -> str:
    """Resolve an int position / str name into an existing column name."""
    if isinstance(col, str):
        if col not in df.columns:
            raise ValueError(f"{role} column not found: {col} (columns: {list(df.columns)})")
        return str(col)
    try:
        return str(df.columns[col])
    except IndexError as exc:
        raise ValueError(
            f"{role} column index {col} out of range for {df.shape[1]} column(s)"
        ) from exc


def _check_nan_labels(df: pd.DataFrame, label_name: str) -> None:
    """Raise ``ValueError`` listing the first NaN-label row numbers (0-based)."""
    nan_mask = df[label_name].isna().to_numpy()
    if not nan_mask.any():
        return
    rows = [int(i) for i in np.flatnonzero(nan_mask)]
    shown = rows[:MAX_REPORTED_NAN_ROWS]
    suffix = ", ..." if len(rows) > len(shown) else ""
    raise ValueError(
        f"Label column '{label_name}' has {len(rows)} NaN value(s) "
        f"at 0-based row(s) {shown}{suffix}"
    )


def _stratify_or_none(
    y: np.ndarray, holdout_fraction: float, label_space: LabelSpace, stage: str
) -> Optional[np.ndarray]:
    """Return ``y`` when sklearn can stratify this stage, else warn and return None.

    Multilabel is never stratified: ``train_test_split`` treats a 2-D
    ``stratify`` argument as one class per unique row, which aborts as soon
    as any label combination has a single member. Iterative stratification
    (e.g. scikit-multilearn) would be the proper upgrade and is left as
    future work. Multiclass stratification needs >= 2 members per class
    and at least ``n_classes`` slots in both folds.
    """
    if label_space.is_multilabel:
        return None
    _, counts = np.unique(y, return_counts=True)
    n = int(y.shape[0])
    n_holdout = int(np.ceil(n * holdout_fraction))
    n_kept = n - n_holdout
    n_present = int(counts.size)
    if int(counts.min()) >= 2 and n_holdout >= n_present and n_kept >= n_present:
        return y
    warnings.warn(
        f"{stage} split: class counts too sparse to stratify "
        f"(min class count {int(counts.min())}, {n_present} classes, "
        f"{n_holdout} holdout rows); falling back to non-stratified split",
        UserWarning,
    )
    return None
