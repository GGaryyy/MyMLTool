"""Per-class keyword extraction for Chinese text corpora.

Counts char 2-3 grams — in Chinese these read like words (資安, 採購,
預算編列), so no segmenter is needed — and scores each gram against a
one-vs-rest binary target per class:

- ``chi2``: sklearn chi-squared score, keeping only grams whose mean term
  frequency inside the class exceeds the overall mean (positively
  associated grams).
- ``mi``: ``mutual_info_classif`` with discrete features; symmetric, so
  negatively associated grams may also rank.

Classes with fewer than two positive documents keep their report entry but
get an empty keyword list.
"""

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2, mutual_info_classif

from src.nlp.labels import build_label_space

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
KEYWORD_METHODS = ("chi2", "mi")
COUNT_NGRAM_RANGE = (2, 3)
DEFAULT_TOP_K = 20
DEFAULT_MIN_DF = 2
MIN_POSITIVE_DOCS = 2
MI_RANDOM_STATE = 0


@dataclass
class ClassKeywords:
    """Top scored character n-grams for one class."""

    class_name: str
    keywords: list

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class KeywordReport:
    """Keyword lists for every class in the label space."""

    method: str
    per_class: list

    def to_dict(self) -> dict:
        return asdict(self)


def class_keywords(texts: Sequence[str], raw_labels: Sequence[str], task_type: str,
                   label_separator: str = "|", top_k: int = DEFAULT_TOP_K,
                   method: str = "chi2", min_df: int = DEFAULT_MIN_DF) -> KeywordReport:
    """Score char 2-3 grams per class, one-vs-rest, sorted by descending score."""
    if method not in KEYWORD_METHODS:
        raise ValueError(f"method must be one of {KEYWORD_METHODS}, got '{method}'")
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")

    texts = [str(t) for t in texts]
    label_space, y = build_label_space(raw_labels, task_type, separator=label_separator)

    vectorizer = CountVectorizer(analyzer="char", ngram_range=COUNT_NGRAM_RANGE,
                                 min_df=min_df)
    X = vectorizer.fit_transform(texts)
    features = vectorizer.get_feature_names_out()
    overall_mean = np.asarray(X.mean(axis=0)).ravel()

    per_class = []
    for col, name in enumerate(label_space.classes):
        if label_space.is_multilabel:
            target = np.asarray(y[:, col])
        else:
            target = (y == col).astype(np.int64)
        keywords = _score_one_class(X, target, features, overall_mean, method, top_k)
        per_class.append(ClassKeywords(class_name=name, keywords=keywords))

    return KeywordReport(method=method, per_class=per_class)


def _score_one_class(X, target: np.ndarray, features: np.ndarray,
                     overall_mean: np.ndarray, method: str, top_k: int) -> list:
    """Top-k ``(gram, score)`` pairs for one binary one-vs-rest target."""
    n_pos = int(target.sum())
    if n_pos < MIN_POSITIVE_DOCS or n_pos == len(target):
        return []

    if method == "chi2":
        scores, _ = chi2(X, target)
        class_mean = np.asarray(X[target == 1].mean(axis=0)).ravel()
        eligible = np.flatnonzero((class_mean > overall_mean) & np.isfinite(scores))
    else:
        scores = mutual_info_classif(X, target, discrete_features=True,
                                     random_state=MI_RANDOM_STATE)
        eligible = np.flatnonzero(np.isfinite(scores))

    if eligible.size == 0:
        return []
    top = eligible[np.argsort(scores[eligible])[::-1]][:top_k]
    return [(str(features[i]), float(scores[i])) for i in top]
