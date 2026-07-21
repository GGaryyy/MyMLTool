"""Unit tests for src.nlp.models.tfidf_linear — all four variants."""

import numpy as np
import pytest

from src.nlp.config import DeviceConfig, ModelConfig
from src.nlp.labels import build_label_space
from src.nlp.models.base import FAMILY_BASELINE, FitReport
from src.nlp.models.tfidf_linear import VARIANTS, TfidfLinearClassifier
from src.nlp.synth import generate_synthetic_gov_docs, texts_and_labels

pytestmark = pytest.mark.unit

N_MC_DOCS = 80
N_ML_DOCS = 80
N_CLASSES = 6


@pytest.fixture(scope="module")
def mc_data():
    df = generate_synthetic_gov_docs(mode="balanced", n_docs=N_MC_DOCS, seed=7,
                                     inject_defects=False)
    texts, labels = texts_and_labels(df)
    space, y = build_label_space(labels, "multiclass")
    assert space.n_classes == N_CLASSES  # all six topics present at n=80/seed=7
    return texts, y, space


@pytest.fixture(scope="module")
def ml_data():
    df = generate_synthetic_gov_docs(mode="multilabel", n_docs=N_ML_DOCS, seed=5,
                                     inject_defects=False)
    texts, labels = texts_and_labels(df)
    space, y = build_label_space(labels, "multilabel")
    assert space.n_classes == N_CLASSES
    return texts, y, space


def _built_model(variant, space, **config_kwargs):
    model = TfidfLinearClassifier(variant=variant)
    model.build(space, ModelConfig(name=model.name, **config_kwargs), DeviceConfig())
    return model


# --------------------------------------------------------------------------- #
# multiclass: every variant
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("variant", VARIANTS)
def test_multiclass_fit_predict_proba(variant, mc_data):
    texts, y, space = mc_data
    model = _built_model(variant, space)
    assert model.name == f"tfidf_{variant}"
    assert model.family == FAMILY_BASELINE

    report = model.fit(texts, y)
    assert isinstance(report, FitReport)
    assert report.n_epochs == 1
    assert report.device == "cpu"
    assert report.precision == "fp32"
    assert report.train_seconds > 0.0
    assert len(report.history) == 1
    # synthetic topics are easily separable, so train F1 must clear 0.5
    assert report.history[0]["train_f1_macro"] > 0.5

    pred = model.predict(texts)
    assert pred.shape == (len(texts),)
    proba = model.predict_proba(texts)
    assert proba.shape == (len(texts), N_CLASSES)
    assert np.all(proba >= 0.0)
    assert np.all(proba <= 1.0)
    # rows must sum to 1 (softmax for linearsvm, native proba otherwise)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_val_history_recorded_when_val_given(mc_data):
    texts, y, space = mc_data
    model = _built_model("logreg", space)
    report = model.fit(texts[:60], y[:60], val_texts=texts[60:], val_y=y[60:])
    assert "val_f1_macro" in report.history[0]
    assert 0.0 <= report.history[0]["val_f1_macro"] <= 1.0


# --------------------------------------------------------------------------- #
# multilabel: every variant
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("variant", VARIANTS)
def test_multilabel_fit_predict_no_exception(variant, ml_data):
    texts, y, space = ml_data
    model = _built_model(variant, space)
    model.fit(texts, y)

    proba = model.predict_proba(texts)
    assert proba.shape == (len(texts), N_CLASSES)
    assert np.all(proba >= 0.0)
    assert np.all(proba <= 1.0)

    pred = model.predict(texts)
    assert pred.shape == (len(texts), N_CLASSES)
    assert set(np.unique(pred)) <= {0, 1}


# --------------------------------------------------------------------------- #
# class_weight handling
# --------------------------------------------------------------------------- #
def test_logreg_class_weight_balanced_runs(mc_data):
    texts, y, space = mc_data
    model = _built_model("logreg", space, class_weight="balanced")
    report = model.fit(texts, y)
    assert "class_weight" not in report.notes


def test_nb_class_weight_balanced_warns_and_notes(mc_data):
    texts, y, space = mc_data
    model = TfidfLinearClassifier(variant="nb")
    with pytest.warns(UserWarning, match="class_weight"):
        model.build(space, ModelConfig(name="tfidf_nb", class_weight="balanced"),
                    DeviceConfig())
    report = model.fit(texts, y)
    assert report.notes["class_weight"] == "unsupported for nb; ignored"


# --------------------------------------------------------------------------- #
# lifecycle errors
# --------------------------------------------------------------------------- #
def test_fit_before_build_raises_runtime_error():
    model = TfidfLinearClassifier()
    with pytest.raises(RuntimeError, match="build"):
        model.fit(["一份文件"], np.array([0]))


def test_predict_before_build_raises_runtime_error():
    model = TfidfLinearClassifier()
    with pytest.raises(RuntimeError, match="build"):
        model.predict(["一份文件"])


def test_unknown_variant_raises_value_error():
    with pytest.raises(ValueError, match="variant"):
        TfidfLinearClassifier(variant="xgboost")


# --------------------------------------------------------------------------- #
# save / load
# --------------------------------------------------------------------------- #
def test_save_load_round_trip_logreg(tmp_path, mc_data):
    texts, y, space = mc_data
    model = _built_model("logreg", space)
    model.fit(texts, y)
    path = str(tmp_path / "logreg.joblib")
    model.save(path)

    restored = TfidfLinearClassifier(variant="logreg")
    restored.build(space, ModelConfig(name="tfidf_logreg"), DeviceConfig())
    restored.load(path)
    assert np.array_equal(restored.predict(texts), model.predict(texts))
    np.testing.assert_allclose(restored.predict_proba(texts), model.predict_proba(texts))


def test_load_rejects_variant_mismatch(tmp_path, mc_data):
    texts, y, space = mc_data
    model = _built_model("nb", space)
    model.fit(texts, y)
    path = str(tmp_path / "nb.joblib")
    model.save(path)

    other = TfidfLinearClassifier(variant="logreg")
    other.build(space, ModelConfig(name="tfidf_logreg"), DeviceConfig())
    with pytest.raises(ValueError, match="variant"):
        other.load(path)
