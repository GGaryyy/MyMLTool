# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`docs/examples/usage_demo.ipynb`** — executed Jupyter notebook demonstrating
  end-to-end usage of the data-prep pipeline (`load_dataset` → `split_features_target`
  → `split_train_test` → `scale_features` → `prepare_data`, plus training a model)
  and the `FeatureShiftDetector` (no-drift vs drifted runtime batches, per-feature
  `DriftReport`, and the low-level `ks_drift` / `compute_psi` / `chi2_drift` helpers).

## [1.0.0] - 2026-06-08

### Added

- **`src/data_prep.py`** — reusable data-prep pipeline that extracts the
  load-CSV → split features/target (last column) → `train_test_split(test_size=0.2, random_state=0)`
  → optional `StandardScaler` block that was previously duplicated across every
  script in `Classification/` and `Regression/`. Public API: `load_dataset`,
  `split_features_target`, `split_train_test`, `scale_features`, and `prepare_data`
  (returns a `PreparedData` dataclass). Constants `DEFAULT_TEST_SIZE = 0.2` and
  `DEFAULT_RANDOM_STATE = 0`.
- **`src/feature_shift.py`** — `FeatureShiftDetector` for detecting feature shift /
  data drift in a runtime server's incoming data versus the training reference.
  Numerical features use the Kolmogorov–Smirnov test (`scipy.stats.ks_2samp`) plus
  PSI; categorical features use the Chi-square test (`scipy.stats.chi2_contingency`)
  plus categorical PSI. `detect()` returns a `DriftReport` with per-feature
  statistic / p-value / PSI / drifted flag, an overall verdict, and the list of
  drifted features. Constants `PSI_NO_SHIFT = 0.1`, `PSI_SHIFT = 0.25`,
  `ALPHA = 0.05`, `DEFAULT_BINS = 10`.
- **Full 5-category test suite** under `tests/{unit,integration,stress,e2e,security}/`
  with shared fixtures in `tests/conftest.py`: 71 tests passing with 100% line
  coverage of `src/` (unit 41, integration 5, stress 5, e2e 3, security 17).
- **pytest / coverage configuration** for running the suite with `pytest-cov`
  line-coverage reporting.
- **Project structure** — `src/`, `tests/`, `docs/`, and `output/` directories
  established per the project conventions.
- **`requirements.txt`** pinning the project dependencies (numpy, pandas,
  scikit-learn, scipy, pytest, pytest-cov, pip-audit).
- **`.gitignore`** excluding `.venv/`, `output/`, and cache directories.
- **Documentation and reports** — test report, security report, and code-review
  report, plus system-flow (`docs/flow/`) and development-workflow
  (`docs/workflow/`) documentation.

### Notes

- Existing ML scripts in `Classification/` and `Regression/` were left unchanged;
  this release only adds the new reusable modules, tests, configuration, and docs.

[Unreleased]: https://keepachangelog.com/en/1.1.0/
[1.0.0]: https://keepachangelog.com/en/1.1.0/
