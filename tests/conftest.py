"""Shared pytest fixtures for the MyMLTool test suite.

Fixtures provide synthetic reference / no-drift / drifted dataframes plus a
temp-CSV factory so every test category can build inputs without touching the
real repo data.
"""

import numpy as np
import pandas as pd
import pytest

RANDOM_SEED = 42
N_ROWS = 600


def _make_frame(seed: int, age_loc: float, salary_loc: float, region_probs):
    """Build a frame with two numeric features, one categorical, one target."""
    rng = np.random.default_rng(seed)
    age = rng.normal(loc=age_loc, scale=5.0, size=N_ROWS)
    salary = rng.normal(loc=salary_loc, scale=10000.0, size=N_ROWS)
    region = rng.choice(["north", "south", "east"], size=N_ROWS, p=region_probs)
    target = (salary > salary_loc).astype(int)
    return pd.DataFrame(
        {"age": age, "salary": salary, "region": region, "purchased": target}
    )


@pytest.fixture
def reference_df():
    """Baseline training/reference distribution (includes the target column)."""
    return _make_frame(RANDOM_SEED, age_loc=40.0, salary_loc=60000.0,
                       region_probs=[0.5, 0.3, 0.2])


@pytest.fixture
def nodrift_df():
    """Fresh sample drawn from the SAME distribution as the reference."""
    return _make_frame(RANDOM_SEED + 1, age_loc=40.0, salary_loc=60000.0,
                       region_probs=[0.5, 0.3, 0.2])


@pytest.fixture
def drifted_df():
    """Sample with clearly shifted numeric means and categorical mix."""
    return _make_frame(RANDOM_SEED + 2, age_loc=65.0, salary_loc=95000.0,
                       region_probs=[0.1, 0.2, 0.7])


@pytest.fixture
def feature_reference_df(reference_df):
    """Reference frame WITHOUT the target column, for the drift detector."""
    return reference_df.drop(columns=["purchased"])


@pytest.fixture
def feature_nodrift_df(nodrift_df):
    return nodrift_df.drop(columns=["purchased"])


@pytest.fixture
def feature_drifted_df(drifted_df):
    return drifted_df.drop(columns=["purchased"])


@pytest.fixture
def csv_factory(tmp_path):
    """Return a function that writes a DataFrame to a temp CSV and returns its path."""
    def _write(df, name="data.csv"):
        path = tmp_path / name
        df.to_csv(path, index=False)
        return str(path)
    return _write


@pytest.fixture
def reference_csv(reference_df, csv_factory):
    """Path to a CSV of the reference frame (target in the last column)."""
    return csv_factory(reference_df, "reference.csv")
