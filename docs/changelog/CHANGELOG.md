# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`src/nlp/` — 中文文本 (繁中) NLP text-classification analysis & selection pipeline.**
  A pre-built, data-agnostic pipeline (runs on synthetic Chinese-text data until real
  data arrives) that produces (a) a text EDA report and (b) a pluggable model
  benchmark for algorithm selection. PyTorch-based; GPU-aware for RTX 4070
  (Ada), RTX 5070 Ti (Blackwell) and GB10 (Grace-Blackwell/ARM64).
  - **Core:** `config` (YAML→validated dataclasses, `task_type` multiclass/multilabel,
    `metadata_cols`), `labels` (`LabelSpace`), `metrics` (task-dispatched:
    accuracy/balanced-acc/macro-micro-weighted F1/PR-AUC; multilabel subset-acc/
    hamming/per-label), `segment` (spaCy **char** segmentation — avoids
    jieba/pkuseg), `synth` (deterministic synthetic Chinese text with injected defects),
    `datasets` (stratified train/val/test), `harness` (`run_benchmark`, per-model
    error isolation), `report` (md/json/PNG writers), `cli`
    (`diagnose`/`eda`/`feature-select`/`benchmark`/`download-models`).
  - **device:** `detect_device` / `assert_wheel_compatible` — resolves device &
    precision (bf16 on Blackwell/Ada), and cross-checks GPU compute capability
    against the torch wheel's compiled arch list (CUDA major/minor + PTX rules)
    to pre-empt "no kernel image" failures on Blackwell/GB10.
  - **Models (pluggable `TextClassifier`):** TF-IDF + {LogReg, LinearSVM, NB, Tree},
    TF-IDF/SVD + LightGBM, TextCNN, BiLSTM+Attention, Chinese-BERT fine-tune
    (default `google-bert/bert-base-chinese`, offline path, bf16, truncation),
    frozen sentence-embedding + linear head, SetFit (optional).
  - **Analysis:** `difficulty` (separability, learning curve), `label_quality`
    (self-implemented confident learning — avoids AGPL cleanlab), `keywords`
    (χ²/MI per class), `pii` (Taiwan PII + normalization scan), `feature_selection`
    (term-level χ²/MI/ANOVA/L1/tree + feature-count curve + n-gram + redundancy;
    metadata relevance + Cramér's V / Pearson / VIF correlation + keep/drop).
- **Feature Selection recommendation report** (`analysis/feature_selection.py`,
  `feature-select` CLI, `configs/feature_select.example.yaml`) for text + structured
  metadata columns.
- **Docker delivery** — `Dockerfile` (CUDA 12.8 base for 4070/5070 Ti, Noto CJK
  fonts, non-root, GB10/ARM64 build-arg预留), `docker-compose.yml`,
  `scripts/download_models.py` (offline model pre-download), `.dockerignore`.
- **Config examples** — `configs/{eda,benchmark,feature_select}.example.yaml`.
- **Docs** — `docs/plan_nlp_pipeline.md`, `docs/nlp/{INSTALL,LICENSES,DEPLOYMENT}.md`
  (license inventory with origin-country column, torch install matrix, deployment
  guide), dated test/security/code-review reports (2026-07-21).
- **Tests** — NLP suite across `tests/{unit,integration,stress,e2e,security}/`:
  **427 passing, 2 environment-gated skips, 91% line coverage**. New pytest markers
  `gpu`, `slow`, `network`.
- **`requirements-nlp.txt`** — heavy NLP/DL deps (torch≥2.7, transformers, spaCy,
  sentence-transformers, setfit, lightgbm) split from the CI-light `requirements.txt`
  (which gains pyyaml, matplotlib), with licenses and the torch CUDA index-url matrix.

### Security

- China-origin packages (jieba, pkuseg) and models (hfl/*) excluded; AGPL cleanlab
  replaced by a self-implemented confident-learning; GPL ckip-transformers isolated
  from the default image (opt-in). YAML via `yaml.safe_load`; torch checkpoints via
  `weights_only=True`; PII masked in all generated reports.

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
