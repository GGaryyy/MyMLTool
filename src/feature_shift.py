"""Feature-shift (data-drift) detection for runtime serving.

Fit a :class:`FeatureShiftDetector` on the training/reference data, then call
:meth:`FeatureShiftDetector.detect` on each batch of incoming server data to
decide whether its distribution has drifted away from training.

Drift signals:
- Numerical features: two-sample Kolmogorov-Smirnov test + Population
  Stability Index (PSI).
- Categorical features: Chi-square test of category counts + PSI.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

PSI_NO_SHIFT = 0.1
PSI_SHIFT = 0.25
ALPHA = 0.05
DEFAULT_BINS = 10
_EPSILON = 1e-6


@dataclass
class FeatureResult:
    """Per-feature drift result."""

    feature: str
    type: str  # "numerical" | "categorical"
    test: str  # "ks" | "chi2"
    statistic: float
    p_value: float
    psi: float
    drifted: bool


@dataclass
class DriftReport:
    """Aggregated drift result for one incoming batch."""

    drifted: bool
    drifted_features: list
    n_features: int
    results: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "drifted": self.drifted,
            "drifted_features": self.drifted_features,
            "n_features": self.n_features,
            "results": [r.__dict__ for r in self.results],
        }


def compute_psi(expected, actual, bins: int = DEFAULT_BINS) -> float:
    """Population Stability Index between an expected and an actual sample.

    Bins are derived from quantiles of ``expected`` so the reference is split
    into roughly equal-mass buckets. Empty buckets are floored to a small
    epsilon to keep the log finite.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if expected.size == 0 or actual.size == 0:
        raise ValueError("PSI requires non-empty expected and actual samples")

    quantiles = np.linspace(0, 100, bins + 1)
    edges = np.unique(np.percentile(expected, quantiles))
    if edges.size < 2:
        # Degenerate (near-constant) reference; no measurable shift by bins.
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    expected_counts = np.histogram(expected, bins=edges)[0]
    actual_counts = np.histogram(actual, bins=edges)[0]

    expected_pct = expected_counts / expected_counts.sum()
    actual_pct = actual_counts / actual_counts.sum()
    expected_pct = np.clip(expected_pct, _EPSILON, None)
    actual_pct = np.clip(actual_pct, _EPSILON, None)

    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def compute_categorical_psi(expected, actual) -> float:
    """PSI for a categorical feature, comparing per-category proportions."""
    expected_counts = pd.Series(expected).value_counts()
    actual_counts = pd.Series(actual).value_counts()
    if expected_counts.empty or actual_counts.empty:
        raise ValueError("Categorical PSI requires non-empty samples")

    categories = set(expected_counts.index) | set(actual_counts.index)
    expected_total = expected_counts.sum()
    actual_total = actual_counts.sum()

    psi = 0.0
    for category in categories:
        e_pct = max(expected_counts.get(category, 0) / expected_total, _EPSILON)
        a_pct = max(actual_counts.get(category, 0) / actual_total, _EPSILON)
        psi += (a_pct - e_pct) * np.log(a_pct / e_pct)
    return float(psi)


def ks_drift(reference, current, alpha: float = ALPHA) -> dict:
    """Two-sample KS test for a numerical feature."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    if reference.size == 0 or current.size == 0:
        raise ValueError("KS test requires non-empty samples")
    result = stats.ks_2samp(reference, current)
    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "drifted": bool(result.pvalue < alpha),
    }


def chi2_drift(reference, current, alpha: float = ALPHA) -> dict:
    """Chi-square test comparing category frequencies of a categorical feature.

    Builds a 2-row contingency table (reference vs current) over the union of
    observed categories. Zero columns are floored to epsilon so the test stays
    defined when a category is absent in one sample.
    """
    ref_counts = pd.Series(reference).value_counts()
    cur_counts = pd.Series(current).value_counts()
    categories = sorted(set(ref_counts.index) | set(cur_counts.index), key=str)
    if not categories:
        raise ValueError("Chi-square test requires at least one category")

    ref_row = np.array([ref_counts.get(c, 0) for c in categories], dtype=float)
    cur_row = np.array([cur_counts.get(c, 0) for c in categories], dtype=float)
    table = np.vstack([ref_row, cur_row])
    table = np.clip(table, _EPSILON, None)

    chi2, p_value, _, _ = stats.chi2_contingency(table)
    return {
        "statistic": float(chi2),
        "p_value": float(p_value),
        "drifted": bool(p_value < alpha),
    }


class FeatureShiftDetector:
    """Detect feature shift between reference data and incoming runtime data."""

    def __init__(
        self,
        reference_df: pd.DataFrame,
        categorical_features: Optional[list] = None,
        alpha: float = ALPHA,
        psi_threshold: float = PSI_SHIFT,
        bins: int = DEFAULT_BINS,
    ):
        if not isinstance(reference_df, pd.DataFrame):
            raise TypeError("reference_df must be a pandas DataFrame")
        if reference_df.empty:
            raise ValueError("reference_df must not be empty")

        self.reference_df = reference_df.copy()
        self.columns = list(reference_df.columns)
        self.alpha = alpha
        self.psi_threshold = psi_threshold
        self.bins = bins

        if categorical_features is None:
            categorical_features = [
                c
                for c in self.columns
                if not pd.api.types.is_numeric_dtype(reference_df[c])
            ]
        unknown = set(categorical_features) - set(self.columns)
        if unknown:
            raise ValueError(f"Unknown categorical features: {sorted(unknown)}")

        self.categorical_features = list(categorical_features)
        self.numerical_features = [
            c for c in self.columns if c not in self.categorical_features
        ]
        self._fitted = False

    def fit(self) -> "FeatureShiftDetector":
        """Cache reference views per feature. Returns self for chaining."""
        self._reference = {c: self.reference_df[c].dropna().values for c in self.columns}
        self._fitted = True
        return self

    def _check_incoming(self, incoming_df: pd.DataFrame) -> None:
        if not isinstance(incoming_df, pd.DataFrame):
            raise TypeError("incoming_df must be a pandas DataFrame")
        if incoming_df.empty:
            raise ValueError("incoming_df must not be empty")
        missing = set(self.columns) - set(incoming_df.columns)
        if missing:
            raise ValueError(f"Incoming data is missing columns: {sorted(missing)}")

    def detect(self, incoming_df: pd.DataFrame) -> DriftReport:
        """Compare an incoming batch against the reference and report drift."""
        if not self._fitted:
            self.fit()
        self._check_incoming(incoming_df)

        results = []
        for col in self.columns:
            ref = self._reference[col]
            cur = incoming_df[col].dropna().values

            if col in self.numerical_features:
                test = ks_drift(ref, cur, alpha=self.alpha)
                psi = compute_psi(ref, cur, bins=self.bins)
                ftype, test_name = "numerical", "ks"
            else:
                test = chi2_drift(ref, cur, alpha=self.alpha)
                psi = compute_categorical_psi(ref, cur)
                ftype, test_name = "categorical", "chi2"

            drifted = bool(test["drifted"] or psi >= self.psi_threshold)
            results.append(
                FeatureResult(
                    feature=col,
                    type=ftype,
                    test=test_name,
                    statistic=test["statistic"],
                    p_value=test["p_value"],
                    psi=psi,
                    drifted=drifted,
                )
            )

        drifted_features = [r.feature for r in results if r.drifted]
        return DriftReport(
            drifted=len(drifted_features) > 0,
            drifted_features=drifted_features,
            n_features=len(results),
            results=results,
        )
