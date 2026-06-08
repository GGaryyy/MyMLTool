"""Integration tests spanning data_prep and feature_shift.

These tests exercise the full happy-path pipeline: prepare data from a CSV,
build a drift detector on the prepared feature columns, run it against
no-drift and drifted batches, and train a real sklearn model on the prepared
(scaled) output. Error/edge cases for the cross-module flow are covered too.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.data_prep import prepare_data, DEFAULT_TEST_SIZE
from src.feature_shift import FeatureShiftDetector


@pytest.mark.integration
def test_prepare_data_feeds_drift_detector_nodrift_and_drift(
    reference_csv, feature_reference_df, feature_nodrift_df, feature_drifted_df
):
    """prepare_data runs the pipeline; a typed feature frame drives the detector.

    The detector consumes a properly typed feature DataFrame (numeric dtypes
    preserved), which is how it is fed at serving time. ``prepare_data`` is
    exercised here to confirm the pipeline produces a usable split.
    """
    prepared = prepare_data(reference_csv, target_col="purchased")
    assert prepared.feature_names == ["age", "salary", "region"]
    assert prepared.X_train.shape[0] > 0

    detector = FeatureShiftDetector(feature_reference_df).fit()

    nodrift_report = detector.detect(feature_nodrift_df)
    drift_report = detector.detect(feature_drifted_df)

    assert nodrift_report.drifted is False
    assert nodrift_report.drifted_features == []
    assert nodrift_report.n_features == 3

    assert drift_report.drifted is True
    # The drifted fixture shifts age, salary and the region mix.
    assert set(drift_report.drifted_features) == {"age", "salary", "region"}
    assert drift_report.n_features == 3


@pytest.mark.integration
def test_prepare_data_scaled_trains_logistic_regression(reference_df, csv_factory):
    """A real sklearn model fits and predicts on scaled prepare_data output.

    Scaling and model training operate on numeric features only (the original
    repo's assumption), so the categorical ``region`` column is excluded.
    """
    numeric_df = reference_df.drop(columns=["region"])
    numeric_csv = csv_factory(numeric_df, "numeric.csv")

    prepared = prepare_data(
        numeric_csv, target_col="purchased", scale=True, random_state=0
    )

    assert prepared.scaler is not None
    # Scaling fit on train -> per-column mean ~0, std ~1 on the train split.
    assert np.allclose(prepared.X_train.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(prepared.X_train.std(axis=0), 1.0, atol=1e-9)

    model = LogisticRegression(random_state=0, max_iter=1000)
    model.fit(prepared.X_train, prepared.y_train.astype(int))

    preds = model.predict(prepared.X_test)
    assert preds.shape == (prepared.X_test.shape[0],)
    assert set(np.unique(preds)).issubset({0, 1})

    # Target is salary > threshold, so a salary-aware model must be accurate.
    accuracy = (preds == prepared.y_test.astype(int)).mean()
    assert accuracy > 0.9


@pytest.mark.integration
def test_prepare_data_split_ratio_matches_detector_sample(
    reference_csv, feature_reference_df, feature_nodrift_df
):
    """Train/test split ratio is honoured and the detector consumes the split."""
    prepared = prepare_data(reference_csv, target_col="purchased")

    total = prepared.X_train.shape[0] + prepared.X_test.shape[0]
    expected_test = round(total * DEFAULT_TEST_SIZE)
    assert prepared.X_test.shape[0] == expected_test
    assert prepared.y_train.shape[0] == prepared.X_train.shape[0]

    detector = FeatureShiftDetector(feature_reference_df).fit()
    report = detector.detect(feature_nodrift_df)
    assert report.drifted is False


@pytest.mark.integration
def test_detector_from_prepared_data_rejects_missing_columns(reference_csv):
    """Detector built on prepared features rejects an incoming frame missing them."""
    prepared = prepare_data(reference_csv, target_col="purchased")
    ref_features = pd.DataFrame(prepared.X_train, columns=prepared.feature_names)
    detector = FeatureShiftDetector(ref_features).fit()

    incoming = pd.DataFrame({"age": [40.0, 41.0], "salary": [60000.0, 61000.0]})
    with pytest.raises(ValueError):
        detector.detect(incoming)


@pytest.mark.integration
def test_detector_from_prepared_data_rejects_empty_batch(reference_csv):
    """Detector built on prepared features rejects an empty incoming batch."""
    prepared = prepare_data(reference_csv, target_col="purchased")
    ref_features = pd.DataFrame(prepared.X_train, columns=prepared.feature_names)
    detector = FeatureShiftDetector(ref_features).fit()

    empty = pd.DataFrame(columns=prepared.feature_names)
    with pytest.raises(ValueError):
        detector.detect(empty)
