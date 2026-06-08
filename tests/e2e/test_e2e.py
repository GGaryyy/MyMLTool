"""End-to-end monitoring workflow tests.

Simulates the realistic flow a user would run in production:

1. A training CSV is written to disk and fed through ``prepare_data`` to obtain
   train/test splits (and feature names).
2. The training feature frame becomes the "reference" distribution for a
   :class:`FeatureShiftDetector`.
3. Two runtime server batches are scored against that reference:
   - a no-drift batch from the same distribution -> ``drifted`` is False,
   - a shifted batch -> ``drifted`` is True and the right features are flagged.
4. The resulting :class:`DriftReport` must serialize to JSON.
"""

import json
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from src.data_prep import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    prepare_data,
)
from src.feature_shift import DriftReport, FeatureShiftDetector, PSI_SHIFT


@pytest.mark.e2e
def test_full_monitoring_workflow_no_drift_then_drift(
    reference_df,
    feature_nodrift_df,
    feature_drifted_df,
    csv_factory,
):
    """Train via prepare_data, then monitor a clean batch and a drifted batch."""
    # --- Stage 1: training data prep from a CSV on disk ---------------------
    train_csv = csv_factory(reference_df, "train.csv")
    prepared = prepare_data(
        train_csv,
        target_col=-1,
        test_size=DEFAULT_TEST_SIZE,
        random_state=DEFAULT_RANDOM_STATE,
        scale=False,
    )

    n_total = len(reference_df)
    expected_test = round(n_total * DEFAULT_TEST_SIZE)
    expected_train = n_total - expected_test
    assert prepared.X_train.shape[0] == expected_train
    assert prepared.X_test.shape[0] == expected_test
    assert prepared.y_train.shape[0] == expected_train
    assert prepared.y_test.shape[0] == expected_test
    # last column was the target, so three features remain
    assert prepared.feature_names == ["age", "salary", "region"]
    assert prepared.X_train.shape[1] == len(prepared.feature_names)
    assert prepared.scaler is None

    # --- Stage 2: build the reference distribution from training features ---
    reference_features = reference_df[prepared.feature_names]
    detector = FeatureShiftDetector(reference_features).fit()
    assert set(detector.numerical_features) == {"age", "salary"}
    assert detector.categorical_features == ["region"]

    # --- Stage 3a: runtime batch with NO drift -----------------------------
    clean_report = detector.detect(feature_nodrift_df)
    assert isinstance(clean_report, DriftReport)
    assert clean_report.n_features == len(prepared.feature_names)
    assert clean_report.drifted is False
    assert clean_report.drifted_features == []

    # --- Stage 3b: runtime batch WITH drift --------------------------------
    drift_report = detector.detect(feature_drifted_df)
    assert isinstance(drift_report, DriftReport)
    assert drift_report.drifted is True
    # every feature was shifted in the drifted frame
    assert set(drift_report.drifted_features) == {"age", "salary", "region"}

    # per-feature results carry the right test/type pairing
    by_feature = {r.feature: r for r in drift_report.results}
    assert by_feature["age"].type == "numerical"
    assert by_feature["age"].test == "ks"
    assert by_feature["salary"].type == "numerical"
    assert by_feature["region"].type == "categorical"
    assert by_feature["region"].test == "chi2"

    # --- Stage 4: the report must be JSON-serializable ----------------------
    payload = drift_report.to_dict()
    assert payload["drifted"] is True
    assert payload["n_features"] == len(prepared.feature_names)
    assert len(payload["results"]) == len(prepared.feature_names)

    serialized = json.dumps(payload)
    assert isinstance(serialized, str)
    # round-trips cleanly
    roundtrip = json.loads(serialized)
    assert roundtrip["drifted_features"] == drift_report.drifted_features


@pytest.mark.e2e
def test_scaled_pipeline_then_streaming_batches(reference_df, csv_factory):
    """Scaled prepare_data run, then stream several runtime batches through one detector."""
    # Scaling operates on numeric features only; exclude the categorical column.
    numeric_df = reference_df.drop(columns=["region"])
    train_csv = csv_factory(numeric_df, "scaled_train.csv")
    prepared = prepare_data(train_csv, scale=True, random_state=DEFAULT_RANDOM_STATE)

    # scaling fit on train: train columns are ~zero-mean / unit-variance
    assert prepared.scaler is not None
    assert prepared.X_train.shape[1] == 2
    assert np.allclose(prepared.X_train.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(prepared.X_train.std(axis=0), 1.0, atol=1e-6)

    # The detector still monitors the full typed feature frame (incl. region).
    reference_features = reference_df[["age", "salary", "region"]]
    detector = FeatureShiftDetector(reference_features).fit()

    # stream deterministic batches drawn from the SAME generating distribution
    # as the reference (matches conftest's _make_frame parameters).
    rng = np.random.default_rng(7)
    reports = []
    for _ in range(3):
        batch = pd.DataFrame(
            {
                "age": rng.normal(40.0, 5.0, 500),
                "salary": rng.normal(60000.0, 10000.0, 500),
                "region": rng.choice(["north", "south", "east"], 500, p=[0.5, 0.3, 0.2]),
            }
        )
        report = detector.detect(batch)
        reports.append(report)
        # each streamed report is independently JSON-serializable
        json.dumps(report.to_dict())

    # Same-distribution batches must not show a real distribution shift: PSI (the
    # magnitude-based stability metric) stays below the shift threshold for every
    # feature. The KS p-value test has an inherent ~5% per-feature false-positive
    # rate, so we assert no PERSISTENT drift (no feature flagged in every batch)
    # rather than zero flags, which would be statistically unsound.
    for report in reports:
        for result in report.results:
            assert result.psi < PSI_SHIFT
    flag_counts = Counter(
        result.feature
        for report in reports
        for result in report.results
        if result.drifted
    )
    assert all(count < len(reports) for count in flag_counts.values())


@pytest.mark.e2e
def test_runtime_batch_missing_column_is_rejected(reference_df):
    """A malformed runtime batch (missing a reference column) must be rejected."""
    reference_features = reference_df[["age", "salary", "region"]]
    detector = FeatureShiftDetector(reference_features).fit()

    bad_batch = reference_features.drop(columns=["salary"])
    with pytest.raises(ValueError):
        detector.detect(bad_batch)

    # an empty runtime batch is also rejected
    empty_batch = pd.DataFrame(columns=["age", "salary", "region"])
    with pytest.raises(ValueError):
        detector.detect(empty_batch)
