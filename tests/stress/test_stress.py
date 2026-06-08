"""Stress / performance tests for the data-prep and feature-shift pipelines.

These tests build large synthetic frames inline (NOT the small conftest
fixtures) and assert the core pipeline stages finish within a generous time
budget and stay stable across many repeated detections.
"""

import time

import numpy as np
import pandas as pd
import pytest

from src.data_prep import (
    scale_features,
    split_features_target,
    split_train_test,
)
from src.feature_shift import FeatureShiftDetector

# --- Stress configuration ---------------------------------------------------
STRESS_SEED = 12345
N_ROWS = 200_000
N_NUMERIC = 12
TIME_BUDGET_S = 30.0  # generous per-stage budget
N_DETECT_BATCHES = 50
BATCH_ROWS = 20_000
REGIONS = ["north", "south", "east", "west"]


def _make_large_frame(seed, n_rows=N_ROWS, n_numeric=N_NUMERIC,
                      numeric_loc=0.0, region_probs=None):
    """Build a large frame: ``n_numeric`` numeric cols + 2 categorical cols.

    The last column is a 0/1 target so it works with ``split_features_target``.
    """
    if region_probs is None:
        region_probs = [0.4, 0.3, 0.2, 0.1]
    rng = np.random.default_rng(seed)

    data = {}
    for i in range(n_numeric):
        data[f"num_{i}"] = rng.normal(loc=numeric_loc + i, scale=1.0, size=n_rows)
    data["region"] = rng.choice(REGIONS, size=n_rows, p=region_probs)
    data["device"] = rng.choice(["mobile", "desktop"], size=n_rows, p=[0.6, 0.4])
    data["purchased"] = (data["num_0"] > numeric_loc).astype(int)
    return pd.DataFrame(data)


@pytest.fixture
def large_reference_frame():
    return _make_large_frame(STRESS_SEED)


@pytest.mark.stress
def test_split_and_scale_under_time_budget(large_reference_frame):
    """Feature/target split + train/test split + scaling complete in budget."""
    df = large_reference_frame
    # Drop categoricals so scaling sees pure numeric data (mirrors real usage
    # where categoricals are encoded upstream); keep the numeric target split.
    numeric_df = df.drop(columns=["region", "device"])

    start = time.perf_counter()
    X, y, feature_names = split_features_target(numeric_df)
    X_train, X_test, y_train, y_test = split_train_test(
        X, y, test_size=0.2, random_state=0
    )
    X_train_s, X_test_s, scaler = scale_features(X_train, X_test)
    elapsed = time.perf_counter() - start

    assert elapsed < TIME_BUDGET_S, f"splitting/scaling took {elapsed:.2f}s"

    # Shape / ratio correctness on the large frame.
    assert len(feature_names) == N_NUMERIC
    assert X.shape == (N_ROWS, N_NUMERIC)
    assert X_train.shape[0] + X_test.shape[0] == N_ROWS
    assert X_test.shape[0] == pytest.approx(N_ROWS * 0.2, rel=0.01)
    assert X_train_s.shape == X_train.shape
    assert X_test_s.shape == X_test.shape
    # StandardScaler fit on train => train columns ~ zero mean / unit std.
    assert np.allclose(X_train_s.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(X_train_s.std(axis=0), 1.0, atol=1e-6)


@pytest.mark.stress
def test_detector_fit_and_detect_under_time_budget(large_reference_frame):
    """fit() + detect() on a large drifted batch finish in budget and flag drift."""
    reference = large_reference_frame.drop(columns=["purchased"])
    drifted = _make_large_frame(
        STRESS_SEED + 1,
        numeric_loc=5.0,
        region_probs=[0.1, 0.2, 0.3, 0.4],
    ).drop(columns=["purchased"])

    detector = FeatureShiftDetector(reference)
    assert set(detector.categorical_features) == {"region", "device"}

    start = time.perf_counter()
    assert detector.fit() is detector
    report = detector.detect(drifted)
    elapsed = time.perf_counter() - start

    assert elapsed < TIME_BUDGET_S, f"fit+detect took {elapsed:.2f}s"
    assert report.n_features == N_NUMERIC + 2
    assert report.drifted is True
    assert len(report.drifted_features) > 0
    assert isinstance(report.to_dict(), dict)


@pytest.mark.stress
def test_detector_no_drift_on_same_distribution(large_reference_frame):
    """A large same-distribution batch must not be flagged as drifted."""
    reference = large_reference_frame.drop(columns=["purchased"])
    nodrift = _make_large_frame(STRESS_SEED + 99).drop(columns=["purchased"])

    detector = FeatureShiftDetector(reference).fit()
    report = detector.detect(nodrift)

    assert report.drifted is False
    assert report.drifted_features == []
    assert report.n_features == N_NUMERIC + 2


@pytest.mark.stress
def test_repeated_detect_is_stable(large_reference_frame):
    """Loop detect() across many batches: deterministic, valid report each time."""
    reference = large_reference_frame.drop(columns=["purchased"])
    detector = FeatureShiftDetector(reference).fit()
    expected_cols = list(reference.columns)

    # Deterministic per-batch sampling so identical-seed batches give identical
    # reports (stability check, no memory blowup across 50 iterations).
    first_signature = None
    start = time.perf_counter()
    for b in range(N_DETECT_BATCHES):
        batch = _make_large_frame(
            STRESS_SEED + 1000, n_rows=BATCH_ROWS
        ).drop(columns=["purchased"])

        report = detector.detect(batch)

        # Every iteration must yield a well-formed report.
        assert report.n_features == len(expected_cols)
        assert len(report.results) == len(expected_cols)
        assert [r.feature for r in report.results] == expected_cols
        for r in report.results:
            assert r.type in {"numerical", "categorical"}
            assert r.test in {"ks", "chi2"}
            assert np.isfinite(r.statistic)
            assert 0.0 <= r.p_value <= 1.0
            assert np.isfinite(r.psi)
            assert isinstance(r.drifted, bool)

        signature = (
            report.drifted,
            tuple(report.drifted_features),
            tuple(round(r.psi, 9) for r in report.results),
        )
        if first_signature is None:
            first_signature = signature
        else:
            assert signature == first_signature, f"batch {b} drifted result changed"

    elapsed = time.perf_counter() - start
    assert elapsed < TIME_BUDGET_S, f"{N_DETECT_BATCHES} detects took {elapsed:.2f}s"


@pytest.mark.stress
def test_detect_rejects_empty_and_missing_columns(large_reference_frame):
    """Edge cases under load: empty batch and missing reference columns raise."""
    reference = large_reference_frame.drop(columns=["purchased"])
    detector = FeatureShiftDetector(reference).fit()

    with pytest.raises(ValueError):
        detector.detect(pd.DataFrame(columns=reference.columns))

    incomplete = _make_large_frame(
        STRESS_SEED + 7, n_rows=BATCH_ROWS
    ).drop(columns=["purchased", "num_0"])
    with pytest.raises(ValueError):
        detector.detect(incomplete)
