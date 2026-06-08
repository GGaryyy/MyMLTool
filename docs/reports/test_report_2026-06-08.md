# Test Report — 2026-06-08

**Project:** MyMLTool
**Scope:** New modules `src/data_prep.py` and `src/feature_shift.py` and their full test suite. No existing ML scripts under `Classification/` or `Regression/` were modified.
**Framework:** pytest 9.0.3 on Python 3.12.3 (linux)
**Coverage tooling:** pytest-cov / coverage 7.14.1
**Wall time:** 26.20s

## Test Execution Summary

| Metric | Count |
|--------|-------|
| Total | 71 |
| Passed | 71 |
| Failed | 0 |
| Skipped | 0 |

All tests passed at completion.

## Coverage

Line coverage was measured. Branch coverage was not separately instrumented in this run.

| Module | Statements | Missed | Line Coverage |
|--------|-----------:|-------:|--------------:|
| src/data_prep.py | — | 0 | 100% |
| src/feature_shift.py | — | 0 | 100% |
| src/__init__.py | — | 0 | 100% |
| **TOTAL** | **180** | **0** | **100%** |

Notes:
- **Line coverage: 100%** (180 statements, 0 missed).
- **Branch coverage:** not separately measured in this run.
- **Function coverage:** not separately reported; every public function/method has at least one unit test (`load_dataset`, `split_features_target`, `split_train_test`, `scale_features`, `prepare_data`; `FeatureShiftDetector.fit` / `.detect`).

## Per-Category Breakdown

| Category | Directory | Count | Result |
|----------|-----------|------:|--------|
| Unit | `tests/unit/` | 41 | All passed |
| Integration | `tests/integration/` | 5 | All passed |
| Stress | `tests/stress/` | 5 | All passed |
| E2E | `tests/e2e/` | 3 | All passed |
| Security | `tests/security/` | 17 | All passed |
| **Total** | | **71** | **All passed** |

## Stress Test Metrics

Stress tests exercise large frames (up to 200k rows x ~14 columns) and repeated detector invocations. Each test ran within its time budget; no budget was exceeded.

| Test | Workload | Measured Duration |
|------|----------|------------------:|
| test_repeated_detect_is_stable | 50 sequential `detect()` calls over a 200k-row x ~14-col frame | 11.27s |
| detector `fit()` + `detect()` under time budget | single fit + detect on large frame | 0.41s |
| split + scale | `prepare_data` pipeline (split + StandardScaler) | 0.08s |

**Throughput (repeated detect):** ~50 batches / 11.27s ≈ **4.4 `detect()` calls/sec** on a 200k-row frame.

**Latency:** The single slowest `detect()`-class call observed was the **0.41s** fit+detect; the repeated-detect loop averaged ~0.225s per call (11.27s / 50). Fine-grained latency percentiles (**p50 / p95 / p99**) were **not separately instrumented** in this run — only the measured slowest-call durations and aggregate throughput above are reported.

**Error rate:** 0% (no errors or budget overruns across all stress tests).

## Failed Test Details

No tests failed at completion. During development, **4 integration/e2e tests** initially failed. All root causes were on the **test side**; the `src/` modules were correct and were **not** modified to make these pass.

| # | Category | Initial Failure | Root Cause (test-side) | Fix (test-side) |
|---|----------|-----------------|------------------------|-----------------|
| 1 | integration/e2e | Detector reference rebuild error | Test rebuilt the `FeatureShiftDetector` reference from a mixed-dtype `object` numpy array, losing per-column dtypes | Rebuilt the reference from a properly typed `DataFrame` so numeric/categorical columns retained their dtypes |
| 2 | integration/e2e | Model fit error | Test trained `LogisticRegression` on a string column | Excluded/encoded the string column before fitting; trained only on numeric features |
| 3 | integration/e2e | Scaler fit error | Test fit `StandardScaler` on a string column | Restricted scaling to numeric features before fitting the scaler |
| 4 | integration/e2e | Assertion error on KS false-positives | Test asserted **zero** KS false-positives, which is statistically unrealistic at ALPHA=0.05 | Relaxed the assertion to tolerate the expected small false-positive rate under the chosen significance level |

**Resolution:** All 4 issues were fixed in the tests. After the fixes the full suite passed (71/71) with no remaining failures. No source changes were required.
