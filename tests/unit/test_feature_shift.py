"""Unit tests for :mod:`src.feature_shift`.

Each function is exercised in isolation: PSI (numeric + categorical), the KS and
Chi-square drift tests, and FeatureShiftDetector construction / fit. Shared
synthetic frames come from the suite-wide conftest fixtures.
"""

import numpy as np
import pandas as pd
import pytest

from src.feature_shift import (
    ALPHA,
    DEFAULT_BINS,
    PSI_NO_SHIFT,
    PSI_SHIFT,
    FeatureShiftDetector,
    chi2_drift,
    compute_categorical_psi,
    compute_psi,
    ks_drift,
)

SEED = 12345


# --------------------------------------------------------------------------- #
# compute_psi
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_compute_psi_identical_samples_near_zero():
    rng = np.random.default_rng(SEED)
    sample = rng.normal(loc=50.0, scale=5.0, size=2000)
    psi = compute_psi(sample, sample.copy())
    assert isinstance(psi, float)
    assert psi < PSI_NO_SHIFT
    assert psi == pytest.approx(0.0, abs=1e-6)


@pytest.mark.unit
def test_compute_psi_shifted_samples_large():
    rng = np.random.default_rng(SEED)
    expected = rng.normal(loc=50.0, scale=5.0, size=2000)
    actual = rng.normal(loc=80.0, scale=5.0, size=2000)
    psi = compute_psi(expected, actual)
    assert isinstance(psi, float)
    assert psi >= PSI_SHIFT


@pytest.mark.unit
def test_compute_psi_respects_bins_argument():
    rng = np.random.default_rng(SEED)
    expected = rng.normal(size=1000)
    actual = rng.normal(size=1000)
    # Both call paths must succeed and return floats for differing bin counts.
    assert isinstance(compute_psi(expected, actual, bins=5), float)
    assert isinstance(compute_psi(expected, actual, bins=DEFAULT_BINS), float)


@pytest.mark.unit
def test_compute_psi_empty_expected_raises():
    with pytest.raises(ValueError):
        compute_psi([], [1.0, 2.0, 3.0])


@pytest.mark.unit
def test_compute_psi_empty_actual_raises():
    with pytest.raises(ValueError):
        compute_psi([1.0, 2.0, 3.0], [])


# --------------------------------------------------------------------------- #
# compute_categorical_psi
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_compute_categorical_psi_identical_near_zero():
    sample = ["north"] * 50 + ["south"] * 30 + ["east"] * 20
    psi = compute_categorical_psi(sample, list(sample))
    assert isinstance(psi, float)
    assert psi == pytest.approx(0.0, abs=1e-6)


@pytest.mark.unit
def test_compute_categorical_psi_changed_mix_positive():
    expected = ["north"] * 50 + ["south"] * 30 + ["east"] * 20
    actual = ["north"] * 10 + ["south"] * 20 + ["east"] * 70
    psi = compute_categorical_psi(expected, actual)
    assert isinstance(psi, float)
    assert psi > 0.0


@pytest.mark.unit
def test_compute_categorical_psi_empty_raises():
    with pytest.raises(ValueError):
        compute_categorical_psi([], ["a", "b"])
    with pytest.raises(ValueError):
        compute_categorical_psi(["a", "b"], [])


# --------------------------------------------------------------------------- #
# ks_drift
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_ks_drift_same_distribution_not_drifted(feature_reference_df, feature_nodrift_df):
    result = ks_drift(feature_reference_df["age"], feature_nodrift_df["age"])
    assert set(result) == {"statistic", "p_value", "drifted"}
    assert isinstance(result["statistic"], float)
    assert isinstance(result["p_value"], float)
    assert isinstance(result["drifted"], bool)
    assert result["p_value"] >= ALPHA
    assert result["drifted"] is False


@pytest.mark.unit
def test_ks_drift_shifted_distribution_drifted(feature_reference_df, feature_drifted_df):
    result = ks_drift(feature_reference_df["age"], feature_drifted_df["age"])
    assert result["p_value"] < ALPHA
    assert result["drifted"] is True
    assert result["statistic"] > 0.0


@pytest.mark.unit
def test_ks_drift_empty_raises():
    with pytest.raises(ValueError):
        ks_drift([], [1.0, 2.0])
    with pytest.raises(ValueError):
        ks_drift([1.0, 2.0], [])


# --------------------------------------------------------------------------- #
# chi2_drift
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_chi2_drift_same_mix_not_drifted(feature_reference_df, feature_nodrift_df):
    result = chi2_drift(feature_reference_df["region"], feature_nodrift_df["region"])
    assert set(result) == {"statistic", "p_value", "drifted"}
    assert isinstance(result["statistic"], float)
    assert isinstance(result["p_value"], float)
    assert isinstance(result["drifted"], bool)
    assert result["p_value"] >= ALPHA
    assert result["drifted"] is False


@pytest.mark.unit
def test_chi2_drift_changed_mix_drifted(feature_reference_df, feature_drifted_df):
    result = chi2_drift(feature_reference_df["region"], feature_drifted_df["region"])
    assert result["p_value"] < ALPHA
    assert result["drifted"] is True


@pytest.mark.unit
def test_chi2_drift_handles_category_absent_in_one_sample():
    # "east" exists only in the reference; the test must remain defined.
    reference = ["north"] * 40 + ["south"] * 30 + ["east"] * 30
    current = ["north"] * 50 + ["south"] * 50
    result = chi2_drift(reference, current)
    assert isinstance(result["statistic"], float)
    assert isinstance(result["p_value"], float)
    assert np.isfinite(result["statistic"])
    assert result["drifted"] is True


# --------------------------------------------------------------------------- #
# FeatureShiftDetector construction
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_detector_auto_detects_feature_types(feature_reference_df):
    detector = FeatureShiftDetector(feature_reference_df)
    assert detector.categorical_features == ["region"]
    assert set(detector.numerical_features) == {"age", "salary"}


@pytest.mark.unit
def test_detector_construction_stores_params(feature_reference_df):
    detector = FeatureShiftDetector(
        feature_reference_df, alpha=0.01, psi_threshold=0.3, bins=8
    )
    assert detector.alpha == 0.01
    assert detector.psi_threshold == 0.3
    assert detector.bins == 8
    assert detector.columns == list(feature_reference_df.columns)


@pytest.mark.unit
def test_detector_explicit_categorical_features(feature_reference_df):
    detector = FeatureShiftDetector(
        feature_reference_df, categorical_features=["region"]
    )
    assert detector.categorical_features == ["region"]
    assert "region" not in detector.numerical_features


@pytest.mark.unit
def test_detector_non_dataframe_raises_typeerror():
    with pytest.raises(TypeError):
        FeatureShiftDetector({"age": [1, 2, 3]})


@pytest.mark.unit
def test_detector_empty_dataframe_raises_valueerror():
    with pytest.raises(ValueError):
        FeatureShiftDetector(pd.DataFrame())


@pytest.mark.unit
def test_detector_unknown_categorical_feature_raises_valueerror(feature_reference_df):
    with pytest.raises(ValueError):
        FeatureShiftDetector(
            feature_reference_df, categorical_features=["does_not_exist"]
        )


# --------------------------------------------------------------------------- #
# FeatureShiftDetector.fit
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_detector_fit_returns_self(feature_reference_df):
    detector = FeatureShiftDetector(feature_reference_df)
    returned = detector.fit()
    assert returned is detector
    assert detector._fitted is True


# --------------------------------------------------------------------------- #
# Defensive edge branches
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_compute_psi_constant_reference_returns_zero():
    """A near-constant reference yields fewer than two bin edges -> PSI 0."""
    constant = np.full(100, 5.0)
    assert compute_psi(constant, np.full(50, 5.0)) == 0.0


@pytest.mark.unit
def test_chi2_drift_empty_samples_raises():
    """No categories on either side is an error, not a silent pass."""
    with pytest.raises(ValueError):
        chi2_drift(pd.Series([], dtype=object), pd.Series([], dtype=object))


@pytest.mark.unit
def test_detect_auto_fits_when_not_fitted(feature_reference_df, feature_nodrift_df):
    """Calling detect() before fit() triggers an automatic fit."""
    detector = FeatureShiftDetector(feature_reference_df)
    assert detector._fitted is False
    report = detector.detect(feature_nodrift_df)
    assert detector._fitted is True
    assert report.n_features == feature_reference_df.shape[1]
