"""Unit tests for src.data_prep — every public function in isolation."""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.data_prep import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    PreparedData,
    load_dataset,
    prepare_data,
    scale_features,
    split_features_target,
    split_train_test,
)


# --------------------------------------------------------------------------- #
# load_dataset
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_load_dataset_happy_path(reference_csv, reference_df):
    df = load_dataset(reference_csv)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == reference_df.shape
    assert list(df.columns) == list(reference_df.columns)


@pytest.mark.unit
def test_load_dataset_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_dataset("/nonexistent/path/does_not_exist.csv")


@pytest.mark.unit
def test_load_dataset_empty_file_raises_value_error(csv_factory):
    empty_path = csv_factory(pd.DataFrame(), "empty.csv")
    with pytest.raises(ValueError):
        load_dataset(empty_path)


# --------------------------------------------------------------------------- #
# split_features_target
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_split_features_target_default_last_column(reference_df):
    X, y, feature_names = split_features_target(reference_df)
    assert feature_names == ["age", "salary", "region"]
    assert X.shape == (len(reference_df), 3)
    assert y.shape == (len(reference_df),)
    np.testing.assert_array_equal(y, reference_df["purchased"].values)


@pytest.mark.unit
def test_split_features_target_by_name(reference_df):
    X, y, feature_names = split_features_target(reference_df, target_col="salary")
    assert "salary" not in feature_names
    assert feature_names == ["age", "region", "purchased"]
    np.testing.assert_array_equal(y, reference_df["salary"].values)


@pytest.mark.unit
def test_split_features_target_by_int(reference_df):
    X, y, feature_names = split_features_target(reference_df, target_col=0)
    assert feature_names == ["salary", "region", "purchased"]
    np.testing.assert_array_equal(y, reference_df["age"].values)


@pytest.mark.unit
def test_split_features_target_single_column_raises(reference_df):
    one_col = reference_df[["age"]]
    with pytest.raises(ValueError):
        split_features_target(one_col)


@pytest.mark.unit
def test_split_features_target_unknown_name_raises(reference_df):
    with pytest.raises(ValueError):
        split_features_target(reference_df, target_col="not_a_column")


# --------------------------------------------------------------------------- #
# split_train_test
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_split_train_test_default_ratio_shapes(reference_df):
    X, y, _ = split_features_target(reference_df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    n = len(reference_df)
    expected_test = int(round(n * DEFAULT_TEST_SIZE))
    assert len(X_test) == expected_test
    assert len(X_train) == n - expected_test
    assert len(y_test) == expected_test
    assert len(y_train) == n - expected_test


@pytest.mark.unit
def test_split_train_test_custom_test_size(reference_df):
    X, y, _ = split_features_target(reference_df)
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.5)
    n = len(reference_df)
    assert len(X_test) == int(round(n * 0.5))
    assert len(X_train) + len(X_test) == n
    assert len(y_train) + len(y_test) == n


@pytest.mark.unit
def test_split_train_test_deterministic_with_random_state(reference_df):
    X, y, _ = split_features_target(reference_df)
    a = split_train_test(X, y, random_state=DEFAULT_RANDOM_STATE)
    b = split_train_test(X, y, random_state=DEFAULT_RANDOM_STATE)
    for arr_a, arr_b in zip(a, b):
        np.testing.assert_array_equal(arr_a, arr_b)


# --------------------------------------------------------------------------- #
# scale_features
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_scale_features_train_mean_zero_std_one(reference_df):
    # Use only the numeric columns for scaling.
    X_num = reference_df[["age", "salary"]].values
    y = reference_df["purchased"].values
    X_train, X_test, _, _ = split_train_test(X_num, y)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    np.testing.assert_allclose(X_train_scaled.mean(axis=0), 0.0, atol=1e-9)
    np.testing.assert_allclose(X_train_scaled.std(axis=0), 1.0, atol=1e-9)


@pytest.mark.unit
def test_scale_features_returns_fitted_scaler_and_uses_train_stats(reference_df):
    X_num = reference_df[["age", "salary"]].values
    y = reference_df["purchased"].values
    X_train, X_test, _, _ = split_train_test(X_num, y)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    assert isinstance(scaler, StandardScaler)
    # Test set is transformed by the train-fitted scaler, not refit on test.
    expected_test = scaler.transform(X_test)
    np.testing.assert_allclose(X_test_scaled, expected_test)
    # The test mean is generally not zero (proves it wasn't refit on test).
    assert X_train_scaled.shape == X_train.shape
    assert X_test_scaled.shape == X_test.shape


# --------------------------------------------------------------------------- #
# prepare_data
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_prepare_data_returns_prepared_data_with_all_fields(reference_csv, reference_df):
    result = prepare_data(reference_csv)
    assert isinstance(result, PreparedData)
    n = len(reference_df)
    assert len(result.X_train) + len(result.X_test) == n
    assert len(result.y_train) + len(result.y_test) == n
    assert result.feature_names == ["age", "salary", "region"]
    assert result.scaler is None


@pytest.mark.unit
def test_prepare_data_scale_false_scaler_none(reference_csv):
    result = prepare_data(reference_csv, scale=False)
    assert result.scaler is None


@pytest.mark.unit
def test_prepare_data_scale_true_scaler_set_and_arrays_scaled(reference_df, csv_factory):
    # Drop categorical so StandardScaler receives only numeric data.
    numeric_df = reference_df[["age", "salary", "purchased"]]
    numeric_csv = csv_factory(numeric_df, "numeric.csv")
    result = prepare_data(numeric_csv, scale=True)
    assert isinstance(result.scaler, StandardScaler)
    np.testing.assert_allclose(result.X_train.mean(axis=0), 0.0, atol=1e-9)
    np.testing.assert_allclose(result.X_train.std(axis=0), 1.0, atol=1e-9)


@pytest.mark.unit
def test_prepare_data_deterministic(reference_csv):
    a = prepare_data(reference_csv, random_state=DEFAULT_RANDOM_STATE)
    b = prepare_data(reference_csv, random_state=DEFAULT_RANDOM_STATE)
    np.testing.assert_array_equal(a.X_train, b.X_train)
    np.testing.assert_array_equal(a.y_test, b.y_test)
