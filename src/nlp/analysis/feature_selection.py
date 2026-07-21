"""Feature-selection *recommendation* for 公文 classification.

This module produces an ANALYSIS & RECOMMENDATION report only — it never
trains the shipped models nor touches the benchmark. It answers two
questions a modeller asks before tuning a TF-IDF baseline:

- Term features: how many char n-gram features are worth keeping, which
  n-gram setting cross-validates best, which terms are near-duplicates, and
  which terms each ranking method (chi², mutual-info, ANOVA, L1-logreg,
  tree-importance) surfaces.
- Structured metadata (機關/文別/速別/密級/年度…): how relevant each column
  is to the label, and which columns are redundant with one another
  (Cramér's V, Pearson, VIF) — so low-value or collinear columns can be
  dropped.

Only sklearn / scipy / numpy / pandas are used. VIF and Cramér's V are
computed by hand. matplotlib is imported lazily with the Agg backend and
only inside the plot helpers; no torch import happens here. Term-level
feature selection mainly benefits the TF-IDF baseline; BERT / embedding
models learn their own features and are unaffected — this is documented in
the report notes.
"""

import os
from dataclasses import asdict, dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.multiclass import OneVsRestClassifier

from src.nlp.config import RunConfig
from src.nlp.labels import LabelSpace, build_label_space

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
TFIDF_ANALYZER = "char"
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MAX_FEATURES = 30000
DEFAULT_K_GRID = (200, 500, 1000, 2000, 5000, 10000)
# (report key, sklearn analyzer, ngram_range) candidates compared by cross-val.
NGRAM_CANDIDATES = (
    ("char_1_2", "char", (1, 2)),
    ("char_1_3", "char", (1, 3)),
    ("word_1_1", "word", (1, 1)),
)
# "word" analyzer for un-segmented Chinese: keep every CJK/word char as its
# own token (no real word segmentation — a documented limitation).
WORD_TOKEN_PATTERN = r"(?u)\w"
TERM_METHODS = ("chi2", "mutual_info", "anova", "l1_logreg", "tree_importance")
HIGH_CORRELATION = 0.8
HIGH_VIF = 10.0
REDUNDANCY_COSINE = 0.9
TOP_TERMS = 30
REDUNDANCY_TOP_N = 200
MAX_REDUNDANT_PAIRS = 50
MAX_CORR_PAIRS = 200
ELBOW_TOLERANCE = 0.01
CV_FOLDS = 3
TREE_N_ESTIMATORS = 100
TREE_MAX_DEPTH = 12
L1_C = 1.0
L1_MAX_ITER = 1000
LOGREG_MAX_ITER = 1000
VIF_CAP = 1_000_000.0
RELEVANCE_LOW_QUANTILE = 0.25
MI_NEAR_ZERO = 0.01
TREE_NEAR_ZERO = 0.01
_EPS = 1e-9

TERM_NOTE = ("詞彙層級特徵篩選主要提升 TF-IDF 基線；BERT / 嵌入模型自學特徵，"
             "不受此篩選影響。")
WORD_NGRAM_NOTE = ("word 1-gram 以單一 CJK 字元為 token（未做真正斷詞），"
                   "僅供相對比較參考。")


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class FeatureScore:
    """A single ranked feature and its method score."""

    name: str
    score: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TermSelectionReport:
    """Term-level feature-selection recommendation for the TF-IDF baseline."""

    methods: dict            # method -> list[FeatureScore] (top TOP_TERMS)
    feature_count_curve: list  # list[{"k": int, "f1_macro": float}]
    recommended_max_features: int
    ngram_scores: dict       # ngram key -> cross-val f1_macro
    recommended_ngram: str
    redundant_pairs: list    # list[{"term_a","term_b","cosine"}]
    notes: list

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MetadataRelevance:
    """Relevance of one structured metadata column to the label."""

    column: str
    dtype: str  # "categorical" | "numeric"
    chi2: Optional[float]
    mutual_info: float
    anova_f: Optional[float]
    cramers_v: Optional[float]
    tree_importance: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CorrelationReport:
    """Redundancy among metadata features."""

    cramers_v_pairs: list          # categorical-categorical
    numeric_correlation_pairs: list  # |pearson| >= HIGH_CORRELATION
    vif: dict                      # numeric column -> VIF
    high_vif_columns: list

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MetadataFeatureReport:
    """Metadata relevance + correlation + keep/drop recommendation."""

    relevance: list
    correlation: CorrelationReport
    recommended_keep: list
    recommended_drop: list
    notes: list

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FeatureSelectionReport:
    """Top-level feature-selection analysis & recommendation."""

    n_docs: int
    task_type: str
    term: TermSelectionReport
    metadata: Optional[MetadataFeatureReport]
    recommendations: list
    plots: dict

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Term-level scoring helpers
# --------------------------------------------------------------------------- #
def _build_vectorizer(analyzer: str, ngram_range) -> TfidfVectorizer:
    if analyzer == "word":
        return TfidfVectorizer(analyzer="word", ngram_range=ngram_range,
                               token_pattern=WORD_TOKEN_PATTERN,
                               max_features=TFIDF_MAX_FEATURES)
    return TfidfVectorizer(analyzer="char", ngram_range=ngram_range,
                           max_features=TFIDF_MAX_FEATURES)


def _l1_scores(X, y, seed: int) -> np.ndarray:
    """Mean absolute L1-LogisticRegression coefficients per feature.

    Uses ``l1_ratio=1.0`` with the ``saga`` solver — the sklearn >= 1.8 API
    for a pure-L1 penalty that supports both binary and multiclass (the old
    ``liblinear`` solver now rejects multiclass outright).
    """
    if np.unique(y).size < 2:
        return np.zeros(X.shape[1])
    clf = LogisticRegression(l1_ratio=1.0, solver="saga", C=L1_C,
                             random_state=seed, max_iter=L1_MAX_ITER)
    clf.fit(X, y)
    return np.abs(clf.coef_).mean(axis=0)


def _tree_scores(X, y, seed: int) -> np.ndarray:
    """RandomForest impurity importances per feature."""
    if np.unique(y).size < 2:
        return np.zeros(X.shape[1])
    clf = RandomForestClassifier(n_estimators=TREE_N_ESTIMATORS,
                                 max_depth=TREE_MAX_DEPTH,
                                 random_state=seed, n_jobs=1)
    clf.fit(X, y)
    return clf.feature_importances_


def _scores_for_target(X, y, method: str, seed: int) -> np.ndarray:
    """Per-feature scores for one target (binary or multiclass ``y``)."""
    if method == "chi2":
        scores, _ = chi2(X, y)
        return np.nan_to_num(scores)
    if method == "mutual_info":
        # Binarize term presence: MI on discrete presence is meaningful,
        # whereas MI treating continuous TF-IDF weights as discrete is not.
        return mutual_info_classif(X > 0, y, discrete_features=True, random_state=seed)
    if method == "anova":
        scores, _ = f_classif(X, y)
        return np.nan_to_num(scores)
    if method == "l1_logreg":
        return _l1_scores(X, y, seed)
    if method == "tree_importance":
        return _tree_scores(X, y, seed)
    raise ValueError(f"unknown term method '{method}'")


def _term_scores(X, y, label_space: LabelSpace, method: str, seed: int) -> np.ndarray:
    """Per-feature scores; multilabel averages one-vs-rest across labels.

    Labels with no positive (or no negative) documents are skipped so an
    empty one-vs-rest target never reaches a scorer.
    """
    if label_space.is_multilabel:
        Y = np.asarray(y)
        accum = np.zeros(X.shape[1])
        count = 0
        for col in range(Y.shape[1]):
            target = Y[:, col].ravel()
            n_pos = int(target.sum())
            if n_pos == 0 or n_pos == len(target):
                continue
            accum += _scores_for_target(X, target, method, seed)
            count += 1
        return accum / count if count else np.zeros(X.shape[1])

    yy = np.asarray(y).ravel()
    if np.unique(yy).size < 2:
        return np.zeros(X.shape[1])
    return _scores_for_target(X, yy, method, seed)


def _top_feature_scores(scores: np.ndarray, features: np.ndarray, top_k: int) -> list:
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(scores)[::-1][:top_k]
    return [FeatureScore(name=str(features[i]), score=float(scores[i])) for i in order]


def _cv_macro_f1(X, y, label_space: LabelSpace, seed: int) -> float:
    """3-fold cross-validated macro-F1 of a LogisticRegression on ``X``."""
    n = X.shape[0]
    if n < CV_FOLDS:
        return 0.0
    if label_space.is_multilabel:
        cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
        est = OneVsRestClassifier(LogisticRegression(max_iter=LOGREG_MAX_ITER))
        pred = cross_val_predict(est, X, np.asarray(y), cv=cv)
        return float(f1_score(np.asarray(y), pred, average="macro", zero_division=0))

    yy = np.asarray(y).ravel()
    _, counts = np.unique(yy, return_counts=True)
    if counts.size < 2:
        return 0.0
    if int(counts.min()) >= CV_FOLDS:
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    else:
        cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    pred = cross_val_predict(LogisticRegression(max_iter=LOGREG_MAX_ITER), X, yy, cv=cv)
    return float(f1_score(yy, pred, average="macro", zero_division=0))


# --------------------------------------------------------------------------- #
# Term-level public API
# --------------------------------------------------------------------------- #
def term_feature_selection(texts: Sequence[str], y, label_space: LabelSpace,
                           seed: int = 0) -> TermSelectionReport:
    """Recommend how many / which char TF-IDF features to keep.

    Ranks terms by five methods, sweeps a feature-count curve to find the
    saturation ``k``, compares a few n-gram settings by cross-val macro-F1,
    and lists near-duplicate (redundant) terms.
    """
    texts = [str(t) for t in texts]
    if not texts:
        raise ValueError("texts must be non-empty")

    vectorizer = _build_vectorizer(TFIDF_ANALYZER, TFIDF_NGRAM_RANGE)
    X = vectorizer.fit_transform(texts)
    features = vectorizer.get_feature_names_out()
    n_features = X.shape[1]

    chi2_scores = _term_scores(X, y, label_space, "chi2", seed)
    methods = {"chi2": _top_feature_scores(chi2_scores, features, TOP_TERMS)}
    for method in TERM_METHODS:
        if method == "chi2":
            continue
        scores = _term_scores(X, y, label_space, method, seed)
        methods[method] = _top_feature_scores(scores, features, TOP_TERMS)

    order = np.argsort(chi2_scores)[::-1]
    curve = _feature_count_curve(X, y, label_space, order, n_features, seed)
    recommended_max_features = _recommend_max_features(curve, n_features)

    ngram_scores, recommended_ngram = _ngram_comparison(texts, y, label_space, seed)
    redundant_pairs = _redundant_pairs(X, order, features, n_features)

    notes = [TERM_NOTE, WORD_NGRAM_NOTE]
    if label_space.is_multilabel:
        notes.append("多標籤：各方法以 one-vs-rest 逐標籤計分後平均。")

    return TermSelectionReport(
        methods=methods,
        feature_count_curve=curve,
        recommended_max_features=recommended_max_features,
        ngram_scores=ngram_scores,
        recommended_ngram=recommended_ngram,
        redundant_pairs=redundant_pairs,
        notes=notes,
    )


def _feature_count_curve(X, y, label_space, order, n_features, seed) -> list:
    """Macro-F1 of the top-``k`` chi² features for each ``k`` in the grid."""
    curve = []
    for k in DEFAULT_K_GRID:
        if k > n_features:
            continue
        cols = order[:k]
        f1 = _cv_macro_f1(X[:, cols], y, label_space, seed)
        curve.append({"k": int(k), "f1_macro": float(f1)})
    if not curve and n_features >= 2:
        f1 = _cv_macro_f1(X, y, label_space, seed)
        curve.append({"k": int(n_features), "f1_macro": float(f1)})
    return curve


def _recommend_max_features(curve: list, n_features: int) -> int:
    """Smallest ``k`` whose f1 is within ELBOW_TOLERANCE of the best (elbow)."""
    if not curve:
        return int(n_features)
    best = max(p["f1_macro"] for p in curve)
    for point in curve:  # curve is in ascending k order
        if point["f1_macro"] >= best - ELBOW_TOLERANCE:
            return int(point["k"])
    return int(curve[-1]["k"])


def _ngram_comparison(texts, y, label_space, seed) -> tuple:
    ngram_scores = {}
    for key, analyzer, ngram_range in NGRAM_CANDIDATES:
        vec = _build_vectorizer(analyzer, ngram_range)
        Xn = vec.fit_transform(texts)
        ngram_scores[key] = float(_cv_macro_f1(Xn, y, label_space, seed))
    recommended = max(ngram_scores, key=ngram_scores.get)
    return ngram_scores, recommended


def _redundant_pairs(X, order, features, n_features) -> list:
    """Near-duplicate term columns among the top chi² terms (cosine >= τ)."""
    top_idx = order[:min(REDUNDANCY_TOP_N, n_features)]
    if len(top_idx) < 2:
        return []
    sims = cosine_similarity(X[:, top_idx].T)
    pairs = []
    m = len(top_idx)
    for i in range(m):
        for j in range(i + 1, m):
            cos = float(sims[i, j])
            if cos >= REDUNDANCY_COSINE:
                pairs.append({
                    "term_a": str(features[top_idx[i]]),
                    "term_b": str(features[top_idx[j]]),
                    "cosine": cos,
                })
    pairs.sort(key=lambda p: -p["cosine"])
    return pairs[:MAX_REDUNDANT_PAIRS]


# --------------------------------------------------------------------------- #
# Metadata helpers
# --------------------------------------------------------------------------- #
def cramers_v(x, y) -> float:
    """Bias-corrected Cramér's V between two categorical vectors, in [0, 1].

    Constant or single-category inputs (a degenerate contingency table)
    return ``0.0``.
    """
    table = pd.crosstab(pd.Series(list(x)), pd.Series(list(y)))
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    chi2_stat = stats.chi2_contingency(table)[0]
    n = float(table.to_numpy().sum())
    if n <= 0:
        return 0.0
    phi2 = chi2_stat / n
    r, k = table.shape
    phi2corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rcorr = r - (r - 1) ** 2 / (n - 1)
    kcorr = k - (k - 1) ** 2 / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(phi2corr / denom))


def _label_targets(y, label_space: LabelSpace) -> list:
    """Binary/multiclass target(s) to score metadata relevance against."""
    if label_space.is_multilabel:
        Y = np.asarray(y)
        targets = [Y[:, c].ravel() for c in range(Y.shape[1])
                   if 0 < int(Y[:, c].sum()) < Y.shape[0]]
        return targets or [Y.sum(axis=1).ravel()]
    return [np.asarray(y).ravel()]


def _cat_chi2(col_values, target) -> float:
    table = pd.crosstab(pd.Series(list(col_values)), pd.Series(list(target)))
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    return float(stats.chi2_contingency(table)[0])


def _col_mutual_info(encoded_col: np.ndarray, target, discrete: bool, seed: int) -> float:
    mi = mutual_info_classif(encoded_col.reshape(-1, 1), target,
                             discrete_features=discrete, random_state=seed)
    return float(mi[0])


def _numeric_anova(col: np.ndarray, target) -> float:
    scores, _ = f_classif(col.reshape(-1, 1), target)
    return float(np.nan_to_num(scores)[0])


def _encode_metadata(df: pd.DataFrame, numeric_cols: list) -> tuple:
    """One-hot categoricals + raw numerics; return (matrix, owner-column list)."""
    frames = []
    owner = []
    for col in df.columns:
        if col in numeric_cols:
            frames.append(df[[col]].astype(float))
            owner.append(col)
        else:
            dummies = pd.get_dummies(df[col].astype(str), prefix=str(col))
            if dummies.shape[1] == 0:
                continue
            frames.append(dummies)
            owner.extend([col] * dummies.shape[1])
    if not frames:
        return np.zeros((len(df), 0)), owner
    encoded = pd.concat(frames, axis=1)
    return encoded.to_numpy(dtype=float), owner


def _tree_importance_per_column(df, numeric_cols, y, label_space, seed) -> dict:
    """Aggregate RandomForest importances back to original metadata columns."""
    X_enc, owner = _encode_metadata(df, numeric_cols)
    if X_enc.shape[1] == 0:
        return {col: 0.0 for col in df.columns}

    if label_space.is_multilabel:
        accum = np.zeros(X_enc.shape[1])
        targets = _label_targets(y, label_space)
        for target in targets:
            if np.unique(target).size < 2:
                continue
            clf = RandomForestClassifier(n_estimators=TREE_N_ESTIMATORS,
                                         max_depth=TREE_MAX_DEPTH,
                                         random_state=seed, n_jobs=1)
            clf.fit(X_enc, target)
            accum += clf.feature_importances_
        importances = accum / max(1, len(targets))
    else:
        yy = np.asarray(y).ravel()
        if np.unique(yy).size < 2:
            importances = np.zeros(X_enc.shape[1])
        else:
            clf = RandomForestClassifier(n_estimators=TREE_N_ESTIMATORS,
                                         max_depth=TREE_MAX_DEPTH,
                                         random_state=seed, n_jobs=1)
            clf.fit(X_enc, yy)
            importances = clf.feature_importances_

    per_col = {col: 0.0 for col in df.columns}
    for imp, col in zip(importances, owner):
        per_col[col] += float(imp)
    return per_col


def _compute_vif(df: pd.DataFrame, numeric_cols: list) -> dict:
    """VIF_j = 1/(1-R²_j) from regressing column j on the other numerics."""
    vif = {}
    if len(numeric_cols) < 2:
        for col in numeric_cols:
            vif[col] = 1.0  # single numeric column: VIF undefined, set to 1.0
        return vif
    X = df[numeric_cols].to_numpy(dtype=float)
    for i, col in enumerate(numeric_cols):
        target = X[:, i]
        others = np.delete(X, i, axis=1)
        if np.var(target) < _EPS:
            vif[col] = 1.0
            continue
        r2 = LinearRegression().fit(others, target).score(others, target)
        vif[col] = VIF_CAP if r2 >= 1.0 - _EPS else float(1.0 / (1.0 - r2))
    return vif


def _numeric_correlation_pairs(df: pd.DataFrame, numeric_cols: list) -> list:
    pairs = []
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            a, b = numeric_cols[i], numeric_cols[j]
            va = df[a].to_numpy(dtype=float)
            vb = df[b].to_numpy(dtype=float)
            if np.var(va) < _EPS or np.var(vb) < _EPS:
                continue
            r = float(np.corrcoef(va, vb)[0, 1])
            if np.isfinite(r) and abs(r) >= HIGH_CORRELATION:
                pairs.append({"col_a": str(a), "col_b": str(b), "pearson": r})
    pairs.sort(key=lambda p: -abs(p["pearson"]))
    return pairs[:MAX_CORR_PAIRS]


def _cramers_v_pairs(df: pd.DataFrame, categorical_cols: list) -> list:
    pairs = []
    for i in range(len(categorical_cols)):
        for j in range(i + 1, len(categorical_cols)):
            a, b = categorical_cols[i], categorical_cols[j]
            v = cramers_v(df[a], df[b])
            pairs.append({"col_a": str(a), "col_b": str(b), "cramers_v": float(v)})
    pairs.sort(key=lambda p: -p["cramers_v"])
    return pairs[:MAX_CORR_PAIRS]


def _column_relevance(df, numeric_cols, categorical_cols, y, label_space,
                      tree_per_col, seed) -> list:
    """Per-column relevance scores, averaged over one-vs-rest labels."""
    targets = _label_targets(y, label_space)
    relevance = []
    for col in df.columns:
        tree_imp = float(tree_per_col.get(col, 0.0))
        if col in numeric_cols:
            values = df[col].to_numpy(dtype=float)
            mi = float(np.mean([_col_mutual_info(values, t, False, seed) for t in targets]))
            anova = float(np.mean([_numeric_anova(values, t) for t in targets]))
            relevance.append(MetadataRelevance(
                column=str(col), dtype="numeric", chi2=None, mutual_info=mi,
                anova_f=anova, cramers_v=None, tree_importance=tree_imp))
        else:
            codes = pd.Categorical(df[col].astype(str)).codes.astype(float)
            mi = float(np.mean([_col_mutual_info(codes, t, True, seed) for t in targets]))
            chi2_val = float(np.mean([_cat_chi2(df[col], t) for t in targets]))
            cv_val = float(np.mean([cramers_v(df[col], t) for t in targets]))
            relevance.append(MetadataRelevance(
                column=str(col), dtype="categorical", chi2=chi2_val, mutual_info=mi,
                anova_f=None, cramers_v=cv_val, tree_importance=tree_imp))
    return relevance


def _recommend_keep_drop(relevance: list, correlation: CorrelationReport) -> tuple:
    """Conservative keep/drop: drop clearly-irrelevant OR redundant columns."""
    by_col = {r.column: r for r in relevance}
    score = {r.column: r.mutual_info + r.tree_importance for r in relevance}
    drop = set()
    notes = []

    # Redundancy: from each high-correlation pair drop the less-relevant column.
    for pair in correlation.numeric_correlation_pairs:
        a, b = pair["col_a"], pair["col_b"]
        loser = a if score.get(a, 0.0) <= score.get(b, 0.0) else b
        drop.add(loser)
        notes.append(f"{loser}：與 {a if loser == b else b} 高度線性相關"
                     f"（|pearson|={abs(pair['pearson']):.2f}），建議剔除其一。")
    for pair in correlation.cramers_v_pairs:
        if pair["cramers_v"] < HIGH_CORRELATION:
            continue
        a, b = pair["col_a"], pair["col_b"]
        loser = a if score.get(a, 0.0) <= score.get(b, 0.0) else b
        drop.add(loser)
        notes.append(f"{loser}：與另一類別欄位高度關聯"
                     f"（Cramér's V={pair['cramers_v']:.2f}），建議剔除其一。")

    # Low relevance: bottom-quantile mutual-info AND near-zero tree importance.
    mi_values = [r.mutual_info for r in relevance]
    mi_threshold = float(np.quantile(mi_values, RELEVANCE_LOW_QUANTILE)) if mi_values else 0.0
    for r in relevance:
        if (r.mutual_info <= max(mi_threshold, MI_NEAR_ZERO)
                and r.tree_importance <= TREE_NEAR_ZERO):
            drop.add(r.column)
            notes.append(f"{r.column}：與標籤幾乎無關（MI≈{r.mutual_info:.3f}，"
                         f"tree≈{r.tree_importance:.3f}），建議剔除。")

    keep = [r.column for r in relevance if r.column not in drop]
    drop_list = [c for c in by_col if c in drop]
    return keep, drop_list, notes


# --------------------------------------------------------------------------- #
# Metadata public API
# --------------------------------------------------------------------------- #
def metadata_feature_selection(metadata_df: pd.DataFrame, y, label_space: LabelSpace,
                               seed: int = 0) -> MetadataFeatureReport:
    """Rank structured metadata columns by relevance and flag redundancy."""
    if not isinstance(metadata_df, pd.DataFrame):
        raise TypeError(f"metadata_df must be a DataFrame, got {type(metadata_df).__name__}")
    if metadata_df.shape[1] == 0:
        raise ValueError("metadata_df has no columns")

    numeric_cols = [c for c in metadata_df.columns
                    if pd.api.types.is_numeric_dtype(metadata_df[c])]
    categorical_cols = [c for c in metadata_df.columns if c not in numeric_cols]

    tree_per_col = _tree_importance_per_column(metadata_df, numeric_cols, y,
                                               label_space, seed)
    relevance = _column_relevance(metadata_df, numeric_cols, categorical_cols, y,
                                  label_space, tree_per_col, seed)

    vif = _compute_vif(metadata_df, numeric_cols)
    high_vif_columns = [c for c, v in vif.items() if v >= HIGH_VIF]
    correlation = CorrelationReport(
        cramers_v_pairs=_cramers_v_pairs(metadata_df, categorical_cols),
        numeric_correlation_pairs=_numeric_correlation_pairs(metadata_df, numeric_cols),
        vif=vif,
        high_vif_columns=high_vif_columns,
    )

    keep, drop, drop_notes = _recommend_keep_drop(relevance, correlation)
    notes = ["保守剔除：僅剔除明顯無關或高度冗餘（相關/共線）的欄位。"]
    if len(numeric_cols) < 2 and numeric_cols:
        notes.append("僅一個數值欄位，VIF 無法定義，設為 1.0。")
    if high_vif_columns:
        notes.append("高 VIF（>= 10）欄位存在多重共線性，建議擇一保留。")
    notes.extend(drop_notes)

    return MetadataFeatureReport(
        relevance=relevance,
        correlation=correlation,
        recommended_keep=keep,
        recommended_drop=drop,
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# Top-level driver
# --------------------------------------------------------------------------- #
def run_feature_selection(texts: Sequence[str], raw_labels: Sequence,
                          config: RunConfig, metadata_df: Optional[pd.DataFrame] = None,
                          out_dir: Optional[str] = None) -> FeatureSelectionReport:
    """Run term (always) + metadata (when supplied) feature-selection analysis."""
    texts = [str(t) for t in texts]
    if not texts:
        raise ValueError("texts must be non-empty")
    if len(texts) != len(raw_labels):
        raise ValueError(
            f"texts and raw_labels differ in length: {len(texts)} vs {len(raw_labels)}")
    if metadata_df is not None and len(metadata_df) != len(texts):
        raise ValueError(
            f"metadata_df length {len(metadata_df)} does not match texts {len(texts)}")

    label_space, y = build_label_space(raw_labels, config.data.task_type,
                                       separator=config.data.label_separator)

    term = term_feature_selection(texts, y, label_space, seed=config.seed)
    metadata = None
    if metadata_df is not None and metadata_df.shape[1] > 0:
        metadata = metadata_feature_selection(metadata_df, y, label_space, seed=config.seed)

    recommendations = _assemble_recommendations(term, metadata)

    plots = {}
    if out_dir is not None:
        plots = _save_plots(term, metadata, metadata_df, out_dir)

    return FeatureSelectionReport(
        n_docs=len(texts),
        task_type=config.data.task_type,
        term=term,
        metadata=metadata,
        recommendations=recommendations,
        plots=plots,
    )


def _assemble_recommendations(term: TermSelectionReport,
                              metadata: Optional[MetadataFeatureReport]) -> list:
    recs = [
        f"TF-IDF 建議最多保留約 {term.recommended_max_features} 個字元特徵"
        f"（macro-F1 飽和點）。",
        f"建議 n-gram 設定：{term.recommended_ngram}"
        f"（交叉驗證 macro-F1={term.ngram_scores.get(term.recommended_ngram, 0.0):.4f}）。",
    ]
    if term.redundant_pairs:
        recs.append(f"發現 {len(term.redundant_pairs)} 組高度冗餘字元特徵"
                    f"（cosine >= {REDUNDANCY_COSINE}），可剔除其一。")
    recs.append(TERM_NOTE)
    if metadata is not None:
        if metadata.recommended_keep:
            recs.append("Metadata 建議保留欄位：" + "、".join(metadata.recommended_keep))
        if metadata.recommended_drop:
            recs.append("Metadata 建議剔除欄位：" + "、".join(metadata.recommended_drop))
        if metadata.correlation.high_vif_columns:
            recs.append("高共線性（VIF >= 10）欄位："
                        + "、".join(metadata.correlation.high_vif_columns))
    return recs


# --------------------------------------------------------------------------- #
# Plot helpers (lazy matplotlib, Agg forced)
# --------------------------------------------------------------------------- #
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.family"] = ["Noto Sans CJK TC", "Microsoft JhengHei",
                                   "PingFang TC", "sans-serif"]
    return plt


def _save_plots(term, metadata, metadata_df, out_dir: str) -> dict:
    plots_dir = os.path.join(os.fspath(out_dir), "feature_selection", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plots = {}
    curve_path = _save_curve_plot(term, os.path.join(plots_dir, "feature_count_curve.png"))
    if curve_path:
        plots["feature_count_curve"] = curve_path
    if metadata is not None and metadata_df is not None:
        heatmap_path = _save_cramers_heatmap(
            metadata, metadata_df, os.path.join(plots_dir, "cramers_v_heatmap.png"))
        if heatmap_path:
            plots["cramers_v_heatmap"] = heatmap_path
    return plots


def _save_curve_plot(term: TermSelectionReport, path: str) -> Optional[str]:
    if not term.feature_count_curve:
        return None
    plt = _plt()
    xs = [p["k"] for p in term.feature_count_curve]
    ys = [p["f1_macro"] for p in term.feature_count_curve]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, marker="o", color="#2a78d6")
    ax.axvline(term.recommended_max_features, color="#d1495b", linestyle="--",
               linewidth=1, label=f"建議 k={term.recommended_max_features}")
    ax.set_xlabel("保留特徵數 k")
    ax.set_ylabel("macro-F1 (3-fold CV)")
    ax.set_title("特徵數 vs macro-F1")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _save_cramers_heatmap(metadata, metadata_df, path: str) -> Optional[str]:
    cats = [r.column for r in metadata.relevance if r.dtype == "categorical"]
    if len(cats) < 1:
        return None
    plt = _plt()
    n = len(cats)
    matrix = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            v = cramers_v(metadata_df[cats[i]], metadata_df[cats[j]])
            matrix[i, j] = matrix[j, i] = v
    fig, ax = plt.subplots(figsize=(1.2 * n + 2, 1.2 * n + 1.5))
    im = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(n))
    ax.set_xticklabels(cats, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(cats)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    color="white", fontsize=8)
    fig.colorbar(im, ax=ax, label="Cramér's V")
    ax.set_title("類別欄位 Cramér's V")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
