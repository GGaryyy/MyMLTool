"""Label-space handling for multiclass and multilabel Chinese text classification.

A :class:`LabelSpace` owns the ordered class list and converts between raw
string labels and the encoded arrays every model family consumes:
multiclass -> int64 class indices, multilabel -> {0,1} indicator matrix.
"""

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from src.nlp.config import DEFAULT_LABEL_SEPARATOR, TASK_MULTICLASS, TASK_MULTILABEL, VALID_TASK_TYPES


@dataclass
class LabelSpace:
    """Ordered class vocabulary plus the single/multi-label switch."""

    classes: list
    is_multilabel: bool
    _index: dict = field(init=False, repr=False)

    def __post_init__(self):
        if not self.classes:
            raise ValueError("LabelSpace needs at least one class")
        self.classes = [str(c) for c in self.classes]
        if len(set(self.classes)) != len(self.classes):
            raise ValueError("LabelSpace classes must be unique")
        self._index = {c: i for i, c in enumerate(self.classes)}

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    def encode(self, labels: Sequence) -> np.ndarray:
        """Encode raw labels.

        Multiclass expects a sequence of scalars and returns int64 indices.
        Multilabel expects a sequence of iterables of scalars and returns an
        ``(n, n_classes)`` int64 indicator matrix. Unknown classes raise
        ``ValueError``.
        """
        if self.is_multilabel:
            matrix = np.zeros((len(labels), self.n_classes), dtype=np.int64)
            for row, doc_labels in enumerate(labels):
                if isinstance(doc_labels, str) or not isinstance(doc_labels, Iterable):
                    raise ValueError(
                        "Multilabel encode expects a list of label-lists; "
                        f"row {row} is {type(doc_labels).__name__}"
                    )
                for lab in doc_labels:
                    matrix[row, self._lookup(lab)] = 1
            return matrix

        return np.array([self._lookup(lab) for lab in labels], dtype=np.int64)

    def decode(self, y: np.ndarray) -> list:
        """Inverse of :meth:`encode`: indices -> names, matrix -> name lists."""
        y = np.asarray(y)
        if self.is_multilabel:
            if y.ndim != 2 or y.shape[1] != self.n_classes:
                raise ValueError(f"Expected (n, {self.n_classes}) indicator matrix, got shape {y.shape}")
            return [[self.classes[j] for j in np.flatnonzero(row)] for row in y]

        if y.ndim != 1:
            raise ValueError(f"Expected 1-D class indices, got shape {y.shape}")
        out_of_range = (y < 0) | (y >= self.n_classes)
        if out_of_range.any():
            raise ValueError("Class index out of range in decode()")
        return [self.classes[int(i)] for i in y]

    def _lookup(self, label) -> int:
        key = str(label)
        if key not in self._index:
            raise ValueError(f"Unknown class '{key}'; known classes: {self.classes}")
        return self._index[key]


def parse_multilabel(raw_labels: Sequence, separator: str = DEFAULT_LABEL_SEPARATOR) -> list:
    """Split separator-joined label strings (e.g. ``"人事|預算"``) into lists."""
    if not separator:
        raise ValueError("separator must be a non-empty string")
    parsed = []
    for row, raw in enumerate(raw_labels):
        parts = [p.strip() for p in str(raw).split(separator)]
        parts = [p for p in parts if p]
        if not parts:
            raise ValueError(f"Row {row} has no labels after parsing '{raw}'")
        parsed.append(parts)
    return parsed


def build_label_space(raw_labels: Sequence, task_type: str,
                      separator: str = DEFAULT_LABEL_SEPARATOR):
    """Derive a :class:`LabelSpace` from raw labels and encode them.

    Returns ``(label_space, encoded_y)``. Classes are sorted for a
    deterministic ordering regardless of row order.
    """
    if task_type not in VALID_TASK_TYPES:
        raise ValueError(f"task_type must be one of {VALID_TASK_TYPES}, got '{task_type}'")
    if len(raw_labels) == 0:
        raise ValueError("Cannot build a LabelSpace from zero labels")

    if task_type == TASK_MULTILABEL:
        parsed = parse_multilabel(raw_labels, separator=separator)
        classes = sorted({lab for doc in parsed for lab in doc})
        space = LabelSpace(classes=classes, is_multilabel=True)
        return space, space.encode(parsed)

    classes = sorted({str(lab) for lab in raw_labels})
    space = LabelSpace(classes=classes, is_multilabel=False)
    return space, space.encode(raw_labels)


def class_distribution(label_space: LabelSpace, y: np.ndarray) -> dict:
    """Per-class sample counts from encoded labels, keyed by class name."""
    y = np.asarray(y)
    if label_space.is_multilabel:
        counts = y.sum(axis=0)
        return {c: int(counts[i]) for i, c in enumerate(label_space.classes)}
    return {c: int(np.sum(y == i)) for i, c in enumerate(label_space.classes)}
