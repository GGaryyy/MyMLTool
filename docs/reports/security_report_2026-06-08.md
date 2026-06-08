# Security Report — 2026-06-08

**Scope:** `src/data_prep.py`, `src/feature_shift.py`, `src/__init__.py` (newly added modules; no existing ML scripts modified)
**Tooling:** pip-audit 2.10.0 (dependency scan), regex-based SAST + secret scan over `src/*.py`, security test suite (17 tests) on pytest 9.0.3 / Python 3.12.3 (linux)
**Date:** 2026-06-08

---

## 1. Executive Summary

| Field | Result |
|-------|--------|
| Overall verdict | **PASS** |
| Total findings | **0** |
| Critical | 0 |
| High | 0 |
| Medium | 0 |
| Low | 0 |
| Informational | 0 |

No security findings of any severity were identified. The dependency scan reported no known vulnerabilities, the SAST and secret scans found no dangerous patterns or hardcoded credentials, and all 17 security tests pass. No remediation is required.

This is an **offline ML utility library**: it has no network endpoints, no authentication layer, no database, no session/cookie handling, and performs no deserialization of untrusted input. Several OWASP web-application categories are therefore Not Applicable (N/A) and are justified as such below.

---

## 2. OWASP Top 10 (2021) Check Results

| # | Category | Status | Justification |
|---|----------|--------|---------------|
| A01 | Broken Access Control | N/A | No authorization model, users, roles, or protected resources. The library exposes only pure data-prep / drift-detection functions. |
| A02 | Cryptographic Failures | N/A | No secrets, credentials, PII handling, or cryptographic operations. No data is transmitted or stored encrypted/decrypted. |
| A03 | Injection | **PASS** | No SQL/OS/command construction. SAST confirms no `eval`/`exec`/`os.system`/`shell=True`/`__import__`. Security tests verify CSV formula/script-like cells (`=SUM(1+1)`, `__import__('os').system(...)`) are read as plain string values and never executed. |
| A04 | Insecure Design | **PASS** | Inputs are validated: malformed/empty/single-column CSVs raise specific `ValueError`/`FileNotFoundError`/pandas errors; `FeatureShiftDetector` rejects missing/renamed/empty/non-DataFrame inputs with `ValueError`/`TypeError`. No bare `except`. |
| A05 | Security Misconfiguration | **PASS** | No framework/server config. `.gitignore` excludes `.venv/`, `output/`, and caches. No debug endpoints or exposed config. |
| A06 | Vulnerable & Outdated Components | **PASS** | pip-audit 2.10.0 reports "No known vulnerabilities found" across all installed and transitive dependencies (see §3). |
| A07 | Identification & Authentication Failures | N/A | No authentication, login, session, or identity management in the library. |
| A08 | Software & Data Integrity Failures | **PASS** | No insecure deserialization: SAST confirms no `pickle.load(s)`, no unsafe `yaml.load`. The library does not load untrusted serialized objects; it only reads CSV via pandas as plain data. |
| A09 | Security Logging & Monitoring Failures | N/A | Library raises explicit, specific exceptions to the caller rather than logging/monitoring. No silent failure paths (no bare `except`); error handling is the caller's responsibility. |
| A10 | Server-Side Request Forgery (SSRF) | N/A | No outbound network requests. The library reads only local CSV files supplied by the caller. |

**Summary:** 5 PASS, 5 N/A, 0 FAIL.

---

## 3. Dependency Vulnerability Scan

**Tool:** pip-audit 2.10.0
**Result:** No known vulnerabilities found across all installed dependencies (42 packages incl. transitive). Raw output: `output/pip_audit.json`.

| Package | Version | Known CVEs | Fix Version | Severity |
|---------|---------|------------|-------------|----------|
| numpy | 2.4.6 | none | n/a | n/a |
| pandas | 3.0.3 | none | n/a | n/a |
| scikit-learn | 1.9.0 | none | n/a | n/a |
| scipy | 1.17.1 | none | n/a | n/a |
| pytest | 9.0.3 | none | n/a | n/a |
| pytest-cov | 7.1.0 | none | n/a | n/a |
| coverage | 7.14.1 | none | n/a | n/a |
| pip-audit | 2.10.0 | none | n/a | n/a |
| requests | 2.34.2 | none | n/a | n/a |
| urllib3 | 2.7.0 | none | n/a | n/a |
| certifi | 2026.5.20 | none | n/a | n/a |

All remaining transitive dependencies (joblib, threadpoolctl, packaging, python-dateutil, six, idna, charset-normalizer, etc.) were scanned and reported `vulns: []`. The `fixes` list in the pip-audit output is empty, confirming no remediation is pending.

---

## 4. SAST Findings

**Result:** 0 findings.

The static scan inspects the source text of every `src/*.py` module (excluding `__init__.py`) for dangerous code-execution, deserialization, and command-injection patterns. None were present.

| Pattern Checked | Regex Intent | Found |
|-----------------|--------------|-------|
| `eval(` | Arbitrary code evaluation | No |
| `exec(` | Arbitrary code execution | No |
| `os.system(` | Shell command execution | No |
| `subprocess shell=True` | Shell injection surface | No |
| `pickle.load` / `pickle.loads` | Insecure deserialization | No |
| `yaml.load` (without `Loader=`) | Unsafe YAML deserialization | No |
| `__import__(` | Dynamic import / code execution | No |

These assertions are enforced by the security test suite: `test_no_dangerous_code_execution_patterns` (parametrized per source file) confirms none of the above patterns appear, and `test_src_files_discovered` guards against the scan silently matching nothing. Additionally, `test_csv_formula_cell_read_as_plain_value` and `test_prepare_data_does_not_execute_csv_content` prove that CSV cell contents resembling code/formulas are returned verbatim as data and never evaluated.

---

## 5. Secret Leak Detection

**Result:** 0 secrets found.

The scan checks every `src/*.py` module against secret-detection heuristics. No matches were found.

| Pattern Checked | Intent | Found |
|-----------------|--------|-------|
| `api_key` / `secret` / `password` / `token` / `private_key` assignment to a non-trivial string literal | Hardcoded credentials | No |
| `AKIA[0-9A-Z]{16}` | AWS access key ID | No |
| `-----BEGIN ... PRIVATE KEY-----` | Embedded private key block | No |
| `bearer <token>` | Hardcoded bearer token | No |

| File | Type of Secret | Status |
|------|----------------|--------|
| `src/data_prep.py` | — | Clean (no secrets) |
| `src/feature_shift.py` | — | Clean (no secrets) |
| `src/__init__.py` | — | Clean (no secrets) |

**Repository hygiene:** No `.env` file or secrets file is present in the repository. `.gitignore` excludes `.venv/`, `output/` (program-generated artifacts including scan output), `__pycache__/`, `.coverage`, and `.pytest_cache/`, so credentials and generated artifacts cannot be accidentally committed. Enforced by `test_no_hardcoded_secrets` (parametrized per source file).

---

## 6. Remediation Actions

**None required.** No findings of any severity were identified across dependency scanning, SAST, secret detection, or the OWASP Top 10 review.

For completeness, four test-side failures observed *during development* (rebuilding the detector reference from a mixed-dtype object numpy array, training LogisticRegression / StandardScaler on a string column, and asserting zero KS false-positives) were defects in the test code, not the `src/` modules; they were corrected in the tests. No source changes were needed and they have no security impact.

---

## 7. Test Evidence

| Metric | Value |
|--------|-------|
| Security tests | 17 passed, 0 failed |
| Total suite | 71 passed, 0 failed, 0 skipped (26.20s) |
| Line coverage | `src/data_prep.py` 100%, `src/feature_shift.py` 100%, `src/__init__.py` 100%, TOTAL 100% (180 statements, 0 missed) |
| Artifacts | `output/pytest_full.txt`, `output/pip_audit.json`, `output/category_counts.txt` |
