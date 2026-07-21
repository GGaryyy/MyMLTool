"""Unit tests for the TF-IDF + SVD + LightGBM baseline."""

import sys

import numpy as np
import pytest

from src.nlp.config import DeviceConfig, ModelConfig, TASK_MULTICLASS, TASK_MULTILABEL
from src.nlp.labels import build_label_space
from src.nlp.metrics import compute_metrics
from src.nlp.synth import generate_synthetic_gov_docs, texts_and_labels

pytestmark = pytest.mark.unit

# lightgbm needs the native OpenMP runtime (libgomp); a missing runtime raises
# OSError, which importorskip does not catch — skip the whole module either way.
try:
    import lightgbm  # noqa: F401
except (ImportError, OSError) as exc:  # pragma: no cover - environment gate
    pytest.skip(f"lightgbm unavailable: {exc}", allow_module_level=True)


def _dataset(mode=TASK_MULTICLASS, n=90, seed=0):
    synth_mode = "multilabel" if mode == TASK_MULTILABEL else "balanced"
    df = generate_synthetic_gov_docs(synth_mode, n_docs=n, seed=seed)
    texts, raw = texts_and_labels(df)
    space, y = build_label_space(raw, mode)
    cut = int(n * 0.75)
    return texts[:cut], y[:cut], texts[cut:], y[cut:], space


def test_gbm_multiclass_fit_predict():
    from src.nlp.models.tfidf_gbm import TfidfGbmClassifier

    texts_tr, y_tr, texts_te, y_te, space = _dataset()
    model = TfidfGbmClassifier()
    model.build(space, ModelConfig(name="tfidf_lightgbm", params={"n_estimators": 60}),
                DeviceConfig(device="cpu"))
    report = model.fit(texts_tr, y_tr, texts_te, y_te)
    assert report.family == "baseline"
    proba = model.predict_proba(texts_te)
    assert proba.shape == (len(texts_te), space.n_classes)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert compute_metrics(y_te, model.predict(texts_te), space)["f1_macro"] > 0.3


def test_gbm_svd_components_clamped_note():
    from src.nlp.models.tfidf_gbm import TfidfGbmClassifier

    texts_tr, y_tr, _, _, space = _dataset(n=60)
    model = TfidfGbmClassifier()
    model.build(space, ModelConfig(name="tfidf_lightgbm",
                                   params={"svd_components": 10000, "n_estimators": 40}),
                DeviceConfig(device="cpu"))
    report = model.fit(texts_tr, y_tr)
    # far more components requested than samples/features -> clamp recorded
    assert any("svd" in str(k).lower() or "svd" in str(v).lower()
               for k, v in report.notes.items())


def test_gbm_multilabel_smoke():
    from src.nlp.models.tfidf_gbm import TfidfGbmClassifier

    texts_tr, y_tr, texts_te, y_te, space = _dataset(mode=TASK_MULTILABEL, n=64)
    model = TfidfGbmClassifier()
    model.build(space, ModelConfig(name="tfidf_lightgbm", params={"n_estimators": 40}),
                DeviceConfig(device="cpu"))
    model.fit(texts_tr, y_tr)
    pred = model.predict(texts_te)
    assert pred.shape == (len(texts_te), space.n_classes)
    assert set(np.unique(pred)).issubset({0, 1})


def test_gbm_missing_dependency(monkeypatch):
    from src.nlp.models.tfidf_gbm import TfidfGbmClassifier

    _, _, _, _, space = _dataset(n=40)
    model = TfidfGbmClassifier()
    monkeypatch.setitem(sys.modules, "lightgbm", None)
    with pytest.raises(ImportError):
        model.build(space, ModelConfig(name="tfidf_lightgbm"), DeviceConfig(device="cpu"))


def test_gbm_save_load_roundtrip(tmp_path):
    from src.nlp.models.tfidf_gbm import TfidfGbmClassifier

    texts_tr, y_tr, texts_te, _, space = _dataset(n=70)
    mc = ModelConfig(name="tfidf_lightgbm", params={"n_estimators": 40})
    model = TfidfGbmClassifier()
    model.build(space, mc, DeviceConfig(device="cpu"))
    model.fit(texts_tr, y_tr)
    before = model.predict_proba(texts_te)

    path = str(tmp_path / "gbm.joblib")
    model.save(path)
    restored = TfidfGbmClassifier()
    restored.build(space, mc, DeviceConfig(device="cpu"))
    restored.load(path)
    np.testing.assert_allclose(before, restored.predict_proba(texts_te), atol=1e-6)
