# Code Review Report — 公文 NLP Pipeline

**Date:** 2026-07-21
**Reviewer:** orchestrator synthesis over multi-agent implementation
**Scope:** new `src/nlp/` sub-package (config, labels, metrics, segment, synth, device, eda, datasets, harness, report, cli; models/*; analysis/*), Docker delivery, tests across 5 categories.

## Review Scope

| Area | Files | ~LOC |
|------|-------|------|
| Core | config, labels, metrics, segment, synth, device, datasets, eda, harness, report, cli | ~1900 |
| Models | base, registry, tfidf_linear, tfidf_gbm, _torch_base, textcnn, bilstm_attn, bert_finetune, sent_embed, setfit_clf | ~1300 |
| Analysis | difficulty, label_quality, keywords, pii, feature_selection | ~1400 |
| Tests | unit/integration/stress/e2e/security | ~3000 |
| Delivery | Dockerfile, docker-compose, scripts/download_models, docs/nlp/* | — |

## Code Quality Assessment

| Dimension | Rating | Notes |
|-----------|--------|-------|
| Readability | Strong | Module docstrings, UPPER_CASE constants blocks, small functions, English identifiers |
| Maintainability | Strong | dataclass result containers with `to_dict()`; single pluggable `TextClassifier` interface |
| Complexity | Controlled | Shared `TorchTextClassifier` base removes DL duplication; harness never branches per family |
| Error handling | Strong | Specific exceptions, no bare except; harness isolates per-model failures |
| Consistency | Strong | Mirrors `src/data_prep.py` conventions (last-column label, dataclasses, absolute imports) |

## Design Pattern Compliance

- **Pluggable strategy**: `TextClassifier` ABC + `registry.py` static map; config selects models by string. Multiclass↔multilabel decided solely by `LabelSpace.is_multilabel` — no per-family branching (verified in harness).
- **Lazy heavy imports**: torch/transformers/spaCy/lightgbm imported inside functions or torch-only modules loaded lazily via registry — pure-Python core stays importable and CI-light (confirmed: `--cov=src` fast suite unaffected).
- **Separation**: computation (eda/analysis/harness) vs presentation (report.py writers) cleanly split; all artifacts to `output/nlp/`.
- **Compliance by construction**: char-level segmentation avoids jieba/pkuseg; confident-learning self-implemented to avoid AGPL; GPL isolated from default image.

## Performance Considerations

| Item | Assessment |
|------|------------|
| EDA near-duplicate detection | Bucketed by length to avoid O(n²) on large corpora (documented heuristic) |
| Baseline TF-IDF | char (1,2) grams, `max_features=50000`, sublinear tf — standard, fast |
| DL device/precision | bf16 autocast on Blackwell/Ada; CPU fallback fp32; wheel hard-gate pre-empts cryptic kernel errors |
| BERT long docs | Truncated at `max_length` with a FitReport note; sliding-window flagged as future work |
| feature_selection VIF/Cramér's V | hand-computed (no statsmodels dep), guarded for constant/tiny columns |

## Findings

| ID | File | Severity | Description | Recommendation | Status |
|----|------|----------|-------------|----------------|--------|
| CR-01 | models/setfit_clf.py | Medium | SetFit 1.1.3 incompatible with transformers 5.x | Documented; pin transformers<5 to use SetFit (optional model) | Noted in requirements-nlp.txt |
| CR-02 | models/tfidf_gbm.py | Low | 0% test coverage in WSL (LightGBM needs libgomp) | Runs in Docker (Ubuntu base has libgomp); code exercised via missing-dep test | Accepted |
| CR-03 | report.py plots | Low | CJK glyphs render as tofu without a CJK font | Dockerfile installs `fonts-noto-cjk`; graceful (never fails) | Resolved for delivery |
| CR-04 | bert_finetune.py | Info | `ignore_mismatched_sizes=True` on head load | Correct for fine-tuning from base backbones; documented inline | Accepted |
| CR-05 | models/*_torch | Info | char-level tokenization for TextCNN/BiLSTM (not word) | Deliberate: zero segmenter dependency, strong for Chinese; documented | Accepted |

No High/Critical findings.

## Test Adequacy

427 passing tests, 91% line coverage; positive + negative paths, determinism checks, multilabel paths, offline BERT via tiny local checkpoint, device logic via faked torch across Ada/Blackwell/GB10. Skips are environment-gated and documented.

## Overall Verdict

**Approved with Notes.** The pipeline is coherent, well-tested, and compliant with the government constraints (no China-origin packages, license tracking, offline Docker delivery, PII masking). Notes CR-01 (SetFit/transformers pin) and CR-02 (LightGBM runtime) are environment/optional-feature caveats, both documented with clear remediation, and do not block delivery.
