"""Task-difficulty analysis for Chinese text classification.

Answers "how hard is this classification task and how much data do we
need" with three cheap, CPU-only views:

- :func:`vectorize_texts` turns raw documents into dense vectors
  (char TF-IDF + TruncatedSVD by default; optional sentence embeddings).
- :func:`separability_report` scores class separation (silhouette) plus a
  cross-validated linear-probe f1_macro.
- :func:`learning_curve_report` retrains a TF-IDF + LogisticRegression
  baseline on growing training fractions against one fixed holdout; a flat
  tail (``saturating``) suggests more labelled data will not help much.

:func:`run_difficulty_analysis` bundles everything and optionally saves a
2-D projection scatter and the learning curve as PNGs. matplotlib is
imported lazily with the Agg backend; sentence-transformers is imported
lazily and only for the optional ``sent_embed`` vector method. No torch
import happens in this module.
"""

import os
from dataclasses import asdict, dataclass
from typing import Optional, Sequence

import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, silhouette_score
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from src.nlp.config import RunConfig
from src.nlp.labels import LabelSpace, build_label_space

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
VECTOR_METHODS = ("tfidf_svd", "sent_embed")
PROJECTION_METHODS = ("pca", "tsne")
DEFAULT_SVD_COMPONENTS = 64
DEFAULT_FRACTIONS = (0.1, 0.25, 0.5, 0.75, 1.0)
DEFAULT_SENT_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MAX_FEATURES = 30000
N_PROBE_FOLDS = 3
HOLDOUT_FRACTION = 0.25
SATURATION_DELTA = 0.02
TSNE_MAX_PERPLEXITY = 30


def _char_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(analyzer="char", ngram_range=TFIDF_NGRAM_RANGE,
                           max_features=TFIDF_MAX_FEATURES)


# --------------------------------------------------------------------------- #
# Vectorization and 2-D projection
# --------------------------------------------------------------------------- #
def vectorize_texts(texts: Sequence[str], method: str = "tfidf_svd",
                    n_components: int = DEFAULT_SVD_COMPONENTS, seed: int = 0,
                    pretrained_path: Optional[str] = None) -> np.ndarray:
    """Turn raw documents into a dense float32 matrix ``(n_docs, dim)``.

    ``tfidf_svd`` (default): char 1-2 gram TF-IDF followed by TruncatedSVD
    with ``n_components`` clamped to what the corpus supports.
    ``sent_embed``: multilingual sentence embeddings, lazily imported.
    """
    if method not in VECTOR_METHODS:
        raise ValueError(f"method must be one of {VECTOR_METHODS}, got '{method}'")
    texts = [str(t) for t in texts]
    if not texts:
        raise ValueError("texts must be non-empty")

    if method == "sent_embed":
        return _embed_sentences(texts, pretrained_path)

    matrix = _char_tfidf().fit_transform(texts)
    n_samples, n_features = matrix.shape
    k = max(2, min(int(n_components), n_samples - 1, n_features - 1))
    svd = TruncatedSVD(n_components=k, random_state=seed)
    return svd.fit_transform(matrix).astype(np.float32)


def _embed_sentences(texts: list, pretrained_path: Optional[str]) -> np.ndarray:
    """Optional sentence-embedding path; heavy deps imported lazily."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "vector method 'sent_embed' needs the optional sentence-transformers "
            "dependency; install it via requirements-nlp.txt"
        ) from exc
    model = SentenceTransformer(pretrained_path or DEFAULT_SENT_EMBED_MODEL)
    return np.asarray(model.encode(texts), dtype=np.float32)


def project_2d(X: np.ndarray, method: str = "pca", seed: int = 0) -> np.ndarray:
    """Project vectors to 2-D for plotting; ``pca`` or ``tsne``."""
    if method not in PROJECTION_METHODS:
        raise ValueError(f"method must be one of {PROJECTION_METHODS}, got '{method}'")
    X = np.asarray(X)
    if method == "pca":
        return PCA(n_components=2, random_state=seed).fit_transform(X)

    n_samples = X.shape[0]
    perplexity = float(min(TSNE_MAX_PERPLEXITY, max(1, n_samples // 4)))
    tsne = TSNE(n_components=2, random_state=seed, perplexity=perplexity, init="pca")
    return tsne.fit_transform(X)


# --------------------------------------------------------------------------- #
# Separability
# --------------------------------------------------------------------------- #
@dataclass
class SeparabilityReport:
    """Cluster- and probe-based view of how separable the classes are."""

    silhouette: Optional[float]
    linear_probe_f1_macro: float
    n_samples: int
    n_classes: int
    notes: list

    def to_dict(self) -> dict:
        return asdict(self)


def separability_report(X: np.ndarray, y: np.ndarray, label_space: LabelSpace,
                        seed: int = 0) -> SeparabilityReport:
    """Silhouette score plus a 3-fold logistic-regression probe f1_macro."""
    X = np.asarray(X)
    y = np.asarray(y)
    notes: list = []

    if label_space.is_multilabel:
        silhouette = None
        notes.append("silhouette undefined for multilabel")
        probe_f1 = _multilabel_probe_f1(X, y, seed, notes)
    else:
        silhouette = _multiclass_silhouette(X, y, notes)
        probe_f1 = _multiclass_probe_f1(X, y, seed, notes)

    return SeparabilityReport(
        silhouette=silhouette,
        linear_probe_f1_macro=probe_f1,
        n_samples=int(X.shape[0]),
        n_classes=label_space.n_classes,
        notes=notes,
    )


def _multiclass_silhouette(X: np.ndarray, y: np.ndarray, notes: list) -> Optional[float]:
    counts = np.bincount(y)
    present = counts[counts > 0]
    if len(present) < 2 or int(present.min()) < 2:
        notes.append("silhouette skipped: needs >= 2 classes with >= 2 members each")
        return None
    return float(silhouette_score(X, y))


def _multiclass_probe_f1(X: np.ndarray, y: np.ndarray, seed: int, notes: list) -> float:
    counts = np.bincount(y)
    present = counts[counts > 0]
    if len(present) < 2:
        notes.append("linear probe skipped: fewer than 2 classes")
        return 0.0
    if len(y) < N_PROBE_FOLDS:
        notes.append(f"linear probe skipped: fewer than {N_PROBE_FOLDS} samples")
        return 0.0

    if int(present.min()) >= N_PROBE_FOLDS:
        cv = StratifiedKFold(n_splits=N_PROBE_FOLDS, shuffle=True, random_state=seed)
    else:
        cv = KFold(n_splits=N_PROBE_FOLDS, shuffle=True, random_state=seed)
        notes.append("stratified CV infeasible (a class has fewer members than folds); using KFold")

    pred = cross_val_predict(LogisticRegression(max_iter=1000), X, y, cv=cv)
    return float(f1_score(y, pred, average="macro", zero_division=0))


def _multilabel_probe_f1(X: np.ndarray, Y: np.ndarray, seed: int, notes: list) -> float:
    if len(Y) < N_PROBE_FOLDS:
        notes.append(f"linear probe skipped: fewer than {N_PROBE_FOLDS} samples")
        return 0.0
    cv = KFold(n_splits=N_PROBE_FOLDS, shuffle=True, random_state=seed)
    probe = OneVsRestClassifier(LogisticRegression(max_iter=1000))
    pred = cross_val_predict(probe, X, Y, cv=cv)
    return float(f1_score(Y, pred, average="macro", zero_division=0))


# --------------------------------------------------------------------------- #
# Learning curve
# --------------------------------------------------------------------------- #
@dataclass
class LearningCurvePoint:
    """One retrain: training fraction, resulting size and holdout f1_macro."""

    fraction: float
    n_train: int
    f1_macro: float


@dataclass
class LearningCurveReport:
    """Learning-curve points plus a saturation flag for the last segment."""

    points: list
    saturating: bool

    def to_dict(self) -> dict:
        return asdict(self)


def learning_curve_report(texts: Sequence[str], y: np.ndarray, label_space: LabelSpace,
                          fractions: Sequence[float] = DEFAULT_FRACTIONS,
                          seed: int = 0) -> LearningCurveReport:
    """f1_macro on one fixed 25% holdout while the train fraction grows.

    Subsampling is stratified for multiclass (per-class proportional pick,
    at least one document per class present in the training split); for
    multilabel a random pick is topped up with one positive document per
    otherwise-missing label.
    """
    fractions = _validate_fractions(fractions)
    texts = [str(t) for t in texts]
    y = np.asarray(y)

    idx = np.arange(len(texts))
    stratify = None
    if not label_space.is_multilabel:
        _, counts = np.unique(y, return_counts=True)
        if len(counts) >= 2 and int(counts.min()) >= 2:
            stratify = y
    train_idx, test_idx = train_test_split(
        idx, test_size=HOLDOUT_FRACTION, random_state=seed, stratify=stratify
    )

    test_texts = [texts[i] for i in test_idx]
    y_test = y[test_idx]

    rng = np.random.default_rng(seed)
    if label_space.is_multilabel:
        perm = train_idx[rng.permutation(len(train_idx))]
        class_perms = []
    else:
        perm = np.array([], dtype=np.int64)
        y_train = y[train_idx]
        class_perms = []
        for c in np.unique(y_train):
            members = train_idx[y_train == c]
            class_perms.append(members[rng.permutation(len(members))])

    points = []
    for fraction in fractions:
        if label_space.is_multilabel:
            subset = _multilabel_subset(perm, y, fraction)
        else:
            subset = _multiclass_subset(class_perms, fraction)
        sub_texts = [texts[i] for i in subset]
        f1 = _baseline_f1(sub_texts, y[subset], test_texts, y_test,
                          label_space.is_multilabel)
        points.append(LearningCurvePoint(fraction=float(fraction),
                                         n_train=int(len(subset)), f1_macro=f1))

    saturating = (len(points) >= 2
                  and abs(points[-1].f1_macro - points[-2].f1_macro) < SATURATION_DELTA)
    return LearningCurveReport(points=points, saturating=bool(saturating))


def _validate_fractions(fractions: Sequence[float]) -> list:
    fractions = [float(f) for f in fractions]
    if not fractions:
        raise ValueError("fractions must be non-empty")
    for f in fractions:
        if not 0.0 < f <= 1.0:
            raise ValueError(f"fractions must lie in (0, 1], got {f}")
    return sorted(fractions)


def _multiclass_subset(class_perms: list, fraction: float) -> np.ndarray:
    """Proportional per-class prefix pick, at least one doc per class."""
    parts = []
    for members in class_perms:
        k = min(len(members), max(1, int(round(fraction * len(members)))))
        parts.append(members[:k])
    return np.concatenate(parts)


def _multilabel_subset(perm: np.ndarray, Y: np.ndarray, fraction: float) -> np.ndarray:
    """Random prefix pick, topped up so every train-present label keeps >= 1 positive."""
    n_take = min(len(perm), max(1, int(round(fraction * len(perm)))))
    chosen = {int(i) for i in perm[:n_take]}
    present = Y[perm].sum(axis=0) > 0
    have = Y[perm[:n_take]].sum(axis=0) > 0
    for col in np.flatnonzero(present & ~have):
        positions = np.flatnonzero(Y[perm, col] == 1)
        chosen.add(int(perm[positions[0]]))
    return np.array(sorted(chosen), dtype=np.int64)


def _baseline_f1(train_texts: list, y_train: np.ndarray, test_texts: list,
                 y_test: np.ndarray, is_multilabel: bool) -> float:
    clf = LogisticRegression(max_iter=1000)
    estimator = OneVsRestClassifier(clf) if is_multilabel else clf
    pipe = Pipeline([("tfidf", _char_tfidf()), ("clf", estimator)])
    pipe.fit(train_texts, y_train)
    pred = pipe.predict(test_texts)
    return float(f1_score(y_test, pred, average="macro", zero_division=0))


# --------------------------------------------------------------------------- #
# Bundled analysis
# --------------------------------------------------------------------------- #
@dataclass
class DifficultyReport:
    """Everything the model-selection step needs about task difficulty."""

    separability: SeparabilityReport
    learning_curve: LearningCurveReport
    projection_plot: Optional[str]
    learning_curve_plot: Optional[str]
    vector_method: str

    def to_dict(self) -> dict:
        return asdict(self)


def run_difficulty_analysis(texts: Sequence[str], raw_labels: Sequence[str],
                            config: RunConfig, out_dir: Optional[str] = None,
                            vector_method: str = "tfidf_svd") -> DifficultyReport:
    """Full difficulty analysis; PNG plots are saved only when ``out_dir`` is given."""
    texts = [str(t) for t in texts]
    label_space, y = build_label_space(raw_labels, config.data.task_type,
                                       separator=config.data.label_separator)
    X = vectorize_texts(texts, method=vector_method, seed=config.seed)
    separability = separability_report(X, y, label_space, seed=config.seed)
    curve = learning_curve_report(texts, y, label_space, seed=config.seed)

    projection_plot = None
    learning_curve_plot = None
    if out_dir is not None:
        plots_dir = os.path.join(os.fspath(out_dir), "difficulty", "plots")
        os.makedirs(plots_dir, exist_ok=True)
        coords = project_2d(X, method="pca", seed=config.seed)
        color = y.sum(axis=1) if label_space.is_multilabel else y
        projection_plot = _save_projection_plot(
            coords, color, label_space.is_multilabel,
            os.path.join(plots_dir, "projection_pca.png"))
        learning_curve_plot = _save_learning_curve_plot(
            curve, os.path.join(plots_dir, "learning_curve.png"))

    return DifficultyReport(
        separability=separability,
        learning_curve=curve,
        projection_plot=projection_plot,
        learning_curve_plot=learning_curve_plot,
        vector_method=vector_method,
    )


def _save_projection_plot(coords: np.ndarray, color: np.ndarray,
                          is_multilabel: bool, path: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = "viridis" if is_multilabel else "tab10"
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=np.asarray(color), cmap=cmap, s=14)
    bar_label = "labels per doc" if is_multilabel else "class index"
    fig.colorbar(scatter, ax=ax, label=bar_label)
    ax.set_title("PCA projection (2D)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _save_learning_curve_plot(curve: LearningCurveReport, path: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [p.n_train for p in curve.points]
    ys = [p.f1_macro for p in curve.points]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel("training documents")
    ax.set_ylabel("f1_macro (holdout)")
    ax.set_title("Learning curve")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
