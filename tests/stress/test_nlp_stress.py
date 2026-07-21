"""Stress tests: larger corpora through EDA and the baseline benchmark.

Bounded so CI stays reasonable (a few thousand docs, CPU, baseline models
only). Asserts completion within a generous time budget and correct output
shapes rather than tight performance numbers.
"""

import time

import numpy as np
import pytest

from src.nlp.config import config_from_dict
from src.nlp.eda import run_eda
from src.nlp.labels import build_label_space
from src.nlp.synth import generate_synthetic_gov_docs, texts_and_labels

pytestmark = pytest.mark.stress

LARGE_N = 3000
EDA_TIME_BUDGET_S = 120.0


def test_eda_on_large_corpus_completes():
    df = generate_synthetic_gov_docs("imbalanced", n_docs=LARGE_N, seed=0)
    texts, raw = texts_and_labels(df)
    config = config_from_dict({
        "data": {"task_type": "multiclass"},
        "segment": {"engine": "char"},
    })
    start = time.perf_counter()
    report = run_eda(texts, raw, config)
    elapsed = time.perf_counter() - start
    assert report.n_docs == LARGE_N
    assert report.vocab.vocab_size > 0
    assert elapsed < EDA_TIME_BUDGET_S


def test_baseline_benchmark_scales():
    df = generate_synthetic_gov_docs("balanced", n_docs=1500, seed=1)
    texts, raw = texts_and_labels(df)
    space, y = build_label_space(raw, "multiclass")

    from src.nlp.config import DeviceConfig, ModelConfig
    from src.nlp.models.tfidf_linear import TfidfLinearClassifier

    model = TfidfLinearClassifier(variant="logreg")
    model.build(space, ModelConfig(name="tfidf_logreg"), DeviceConfig(device="cpu"))
    start = time.perf_counter()
    model.fit(texts[:1200], y[:1200])
    proba = model.predict_proba(texts[1200:])
    elapsed = time.perf_counter() - start
    assert proba.shape == (300, space.n_classes)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert elapsed < 90.0


def test_many_classes_wide_label_space():
    # 50 synthetic classes to stress metric aggregation / confusion matrix size
    rng = np.random.default_rng(0)
    texts = [f"主旨：案件 {i} 類別測試內容 {rng.integers(0, 999)}" for i in range(1000)]
    raw = [f"類別{i % 50}" for i in range(1000)]
    space, y = build_label_space(raw, "multiclass")
    assert space.n_classes == 50
    config = config_from_dict({"data": {"task_type": "multiclass"}, "segment": {"engine": "char"}})
    report = run_eda(texts, raw, config)
    assert len(report.balance.counts) == 50
