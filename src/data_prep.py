"""Reusable data-preparation pipeline.

Extracts the load -> split-features/target -> train/test-split -> optional
feature-scaling steps that were duplicated across every ML script in
``Classification/`` and ``Regression/``.
"""

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 0


@dataclass
class PreparedData:
    """Container for the output of :func:`prepare_data`."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list
    scaler: Optional[StandardScaler] = None


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame.

    Raises ``FileNotFoundError`` if the path does not exist and ``ValueError``
    if the file contains no rows.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Dataset is empty: {csv_path}") from exc

    if df.empty:
        raise ValueError(f"Dataset has no rows: {csv_path}")
    return df


def split_features_target(df: pd.DataFrame, target_col: Union[int, str] = -1):
    """Split a DataFrame into features ``X`` and target ``y``.

    By default the last column is the target, matching the repo's existing
    ``iloc[:, :-1]`` / ``iloc[:, -1]`` convention. ``target_col`` may be a
    column name or an integer position.
    """
    if df.shape[1] < 2:
        raise ValueError("DataFrame needs at least one feature and one target column")

    if isinstance(target_col, str):
        if target_col not in df.columns:
            raise ValueError(f"Target column not found: {target_col}")
        target_name = target_col
    else:
        target_name = df.columns[target_col]

    feature_names = [c for c in df.columns if c != target_name]
    X = df[feature_names].values
    y = df[target_name].values
    return X, y, feature_names


def split_train_test(
    X,
    y,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
):
    """Split arrays into train/test sets, keeping the repo's default ratios."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def scale_features(X_train, X_test):
    """Standard-scale features: fit on train, transform both.

    Returns ``(X_train_scaled, X_test_scaled, scaler)``. Mirrors the
    ``StandardScaler`` usage in ``Classification/logistic_regression.py``.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def prepare_data(
    csv_path: str,
    target_col: Union[int, str] = -1,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    scale: bool = False,
) -> PreparedData:
    """Run the full data-prep pipeline end to end.

    Loads the CSV, splits features/target, splits train/test and optionally
    scales features. Returns a :class:`PreparedData` bundle.
    """
    df = load_dataset(csv_path)
    X, y, feature_names = split_features_target(df, target_col=target_col)
    X_train, X_test, y_train, y_test = split_train_test(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = None
    if scale:
        X_train, X_test, scaler = scale_features(X_train, X_test)

    return PreparedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        scaler=scaler,
    )
