# Security Report — 公文 NLP Pipeline

**Date:** 2026-07-21
**Scope:** `src/nlp/` sub-package, CLI, Docker delivery
**Context:** Government use — compliance constraints (no China-origin packages, license tracking, offline delivery) apply.

## Executive Summary

| Severity | Count | Status |
|----------|-------|--------|
| Critical | 0 | — |
| High | 0 | — |
| Medium | dev-tooling only (see §Dependency Scan) | Not in delivered image |
| Low | 0 in pipeline source | — |

**Verdict: PASS.** No secrets, no unsafe deserialization of untrusted input, no prohibited-origin imports, PII masked in all generated artifacts. Dependency-scan findings are confined to the developer/notebook toolchain (jupyter, mistune, tornado, setuptools) which is **not** part of the delivered Docker image.

## OWASP Top 10 (adapted for an offline analysis tool)

| Item | Result | Detail |
|------|--------|--------|
| A01 Broken Access Control | N/A | No auth surface; local CLI/batch tool |
| A02 Cryptographic Failures | PASS | No secrets stored/transmitted; no crypto claims |
| A03 Injection | PASS | No SQL/shell; YAML parsed with `yaml.safe_load` (asserted in `test_config_uses_safe_yaml_load`) |
| A04 Insecure Design | PASS | Wheel/device hard-gate before GPU work; model failures isolated in harness |
| A05 Security Misconfiguration | PASS | Docker runs non-root; `.dockerignore` excludes data/models/output; `HF_HUB_OFFLINE` for air-gap |
| A06 Vulnerable Components | SEE §Dependency Scan | Findings only in non-delivered dev tooling |
| A07 Auth Failures | N/A | No authentication |
| A08 Integrity / Deserialization | PASS | Torch checkpoints load with `weights_only=True` (asserted); no `pickle`/`eval`/`yaml.load` of untrusted input |
| A09 Logging Failures | PASS | `logging` used, not prints (harness); no sensitive data logged |
| A10 SSRF | N/A | No outbound requests except opt-in model download (documented) |

## Dependency Vulnerability Scan (`pip-audit`)

Findings in the WSL dev venv, all in **developer / notebook tooling — NOT pipeline runtime deps and NOT installed in the delivered image**:

| Package | Advisory | Fix | In delivered image? |
|---------|----------|-----|---------------------|
| jupyter-server 2.19.0 | PYSEC-2026-366 | 2.20.0 | No |
| jupyterlab 4.5.8 | GHSA-vmhf-c436-hxj4 | 4.5.9 | No |
| mistune 3.2.1 | PYSEC-2026-2210…2652 (multiple) | 3.3.0 | No |
| msgpack 1.1.2 | GHSA-6v7p-g79w-8964 | 1.2.1 | No (transitive, dev) |
| tornado 6.5.6 | GHSA-pw6j-qg29-8w7f | 6.5.7 | No |
| setuptools 78.1.0 | PYSEC-2025-49, PYSEC-2026-3447 | 83.0.0 | Build-time only |

- **Pipeline runtime stack** (numpy, pandas, scikit-learn, scipy, transformers, sentence-transformers, spaCy, matplotlib, pyyaml): **no advisories reported**.
- `torch 2.11.0+cu128`: not auditable via PyPI (local CUDA wheel); sourced from the official PyTorch cu128 index.
- **Remediation for delivery:** the Docker image installs only `requirements.txt` + the NLP stack (no jupyter). Pin `setuptools>=83` in the image build; developers should update jupyter/mistune/tornado in their local venvs.

## SAST / Static Checks

| Check | Result |
|-------|--------|
| Hard-coded secrets (regex: api_key/secret/password/token, AWS keys, private keys) | PASS — none in `src/nlp/**` |
| Prohibited-origin imports (`jieba`, `pkuseg`, `cleanlab`) | PASS — none imported (`test_no_prohibited_or_unsafe_imports`) |
| Unsafe YAML (`yaml.load`) | PASS — only `yaml.safe_load` |
| Unsafe torch load (`weights_only=False`) | PASS — `weights_only=True` enforced |
| Bare `except` | PASS — specific exceptions throughout |

## Secret Leak Detection

No API keys, tokens, passwords, or private keys in source or config. Config examples use placeholder paths only. `.env` handling not required (offline tool, no credentials).

## PII Handling (government 個資 compliance)

- `analysis/pii.py` scans for Taiwan national IDs, mobile/landline numbers, emails, addresses; the EDA report **counts** PII but **masks** every snippet (`test_pii_masked_in_eda_report`, `test_pii_scan_masks_snippets`).
- Raw national IDs never appear verbatim in `eda_report.md` / `.json` (asserted).
- Recommendation surfaced to users when PII detected: de-identify before model training.

## License Compliance

Full inventory in [docs/nlp/LICENSES.md](../nlp/LICENSES.md). Key controls:
- China-origin packages/models (jieba, pkuseg, hfl/*) excluded.
- AGPL (cleanlab) excluded — confident-learning self-implemented.
- GPL (ckip-transformers, 台灣中研院) opt-in only, **not** in the default Docker image.
- All shipped deps are BSD / MIT / Apache-2.0.

## Findings & Remediation

| ID | Severity | Finding | Action |
|----|----------|---------|--------|
| SEC-01 | Medium | Dev-tooling CVEs (jupyter/mistune/tornado/setuptools) in local venv | Not shipped; pin `setuptools>=83` in image; update dev venv |
| SEC-02 | Low/Info | `torch` cu128 wheel not PyPI-auditable | Sourced from official PyTorch index; documented in INSTALL.md |

No Critical or High findings. Task may proceed.
