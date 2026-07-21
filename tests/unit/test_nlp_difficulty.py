"""Unit tests for src.nlp.analysis.difficulty — vectors, separability, curves."""

import json
import os
import sys

import numpy as np
import pytest

from src.nlp.analysis.difficulty import (
    DEFAULT_SVD_COMPONENTS,
    learning_curve_report,
    project_2d,
    run_difficulty_analysis,
    separability_report,
    vectorize_texts,
)
from src.nlp.config import DataConfig, RunConfig, TASK_MULTICLASS, TASK_MULTILABEL
from src.nlp.labels import LabelSpace, build_label_space
from src.nlp.synth import generate_synthetic_gov_docs, texts_and_labels

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def mc_data():
    df = generate_synthetic_gov_docs(mode="balanced", n_docs=120, seed=7,
                                     inject_defects=False)
    return texts_and_labels(df)


@pytest.fixture(scope="module")
def ml_data():
    df = generate_synthetic_gov_docs(mode="multilabel", n_docs=60, seed=7,
                                     inject_defects=False)
    return texts_and_labels(df)


@pytest.fixture(scope="module")
def mc_vectors(mc_data):
    texts, labels = mc_data
    space, y = build_label_space(labels, TASK_MULTICLASS)
    X = vectorize_texts(texts, seed=0)
    return X, y, space


# --------------------------------------------------------------------------- #
# vectorize_texts
# --------------------------------------------------------------------------- #
def test_vectorize_tfidf_svd_shape_and_dtype(mc_data):
    texts, _ = mc_data
    X = vectorize_texts(texts, method="tfidf_svd", seed=0)
    assert X.shape == (120, DEFAULT_SVD_COMPONENTS)
    assert X.dtype == np.float32


def test_vectorize_clamps_components_on_tiny_corpus():
    df = generate_synthetic_gov_docs(mode="balanced", n_docs=10, seed=1,
                                     inject_defects=False)
    texts, _ = texts_and_labels(df)
    X = vectorize_texts(texts, n_components=500, seed=0)
    assert X.shape[0] == 10
    assert 2 <= X.shape[1] <= 9  # clamped to n_samples - 1 at most
    assert X.dtype == np.float32


def test_vectorize_invalid_method_raises(mc_data):
    texts, _ = mc_data
    with pytest.raises(ValueError, match="method must be one of"):
        vectorize_texts(texts, method="bogus")


def test_vectorize_empty_texts_raises():
    with pytest.raises(ValueError, match="non-empty"):
        vectorize_texts([])


def test_sent_embed_missing_dependency_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(ImportError, match="requirements-nlp.txt"):
        vectorize_texts(["文件甲", "文件乙"], method="sent_embed")


# --------------------------------------------------------------------------- #
# project_2d
# --------------------------------------------------------------------------- #
def test_project_2d_pca_shape(mc_vectors):
    X, _, _ = mc_vectors
    coords = project_2d(X, method="pca", seed=0)
    assert coords.shape == (X.shape[0], 2)


def test_project_2d_tsne_small_n(mc_vectors):
    X, _, _ = mc_vectors
    coords = project_2d(X[:30], method="tsne", seed=0)
    assert coords.shape == (30, 2)


def test_project_2d_invalid_method_raises(mc_vectors):
    X, _, _ = mc_vectors
    with pytest.raises(ValueError, match="method must be one of"):
        project_2d(X, method="umap")


# --------------------------------------------------------------------------- #
# separability_report
# --------------------------------------------------------------------------- #
def test_separability_multiclass_balanced(mc_vectors):
    X, y, space = mc_vectors
    report = separability_report(X, y, space, seed=0)
    assert report.silhouette is not None
    assert report.linear_probe_f1_macro > 0.5
    assert report.n_samples == 120
    assert report.n_classes == 6


def test_separability_single_class_edge(mc_data):
    texts, _ = mc_data
    X = vectorize_texts(texts[:12], seed=0)
    y = np.zeros(12, dtype=np.int64)
    space = LabelSpace(classes=["人事"], is_multilabel=False)
    report = separability_report(X, y, space, seed=0)
    assert report.silhouette is None
    assert any("silhouette" in note for note in report.notes)
    assert report.linear_probe_f1_macro == 0.0


def test_separability_multilabel(ml_data):
    texts, labels = ml_data
    space, Y = build_label_space(labels, TASK_MULTILABEL)
    X = vectorize_texts(texts, seed=0)
    report = separability_report(X, Y, space, seed=0)
    assert report.silhouette is None
    assert any("multilabel" in note for note in report.notes)
    assert 0.0 <= report.linear_probe_f1_macro <= 1.0


# --------------------------------------------------------------------------- #
# learning_curve_report
# --------------------------------------------------------------------------- #
def test_learning_curve_multiclass(mc_data):
    texts, labels = mc_data
    space, y = build_label_space(labels, TASK_MULTICLASS)
    report = learning_curve_report(texts, y, space, seed=0)

    assert len(report.points) == 5
    n_trains = [p.n_train for p in report.points]
    assert n_trains == sorted(n_trains)
    assert all(b > a for a, b in zip(n_trains, n_trains[1:]))
    assert report.points[-1].f1_macro >= report.points[0].f1_macro - 0.05
    assert isinstance(report.saturating, bool)


@pytest.mark.parametrize("bad_fractions", [(0.0, 0.5), (0.5, 1.5)])
def test_learning_curve_fraction_validation(mc_data, bad_fractions):
    texts, labels = mc_data
    space, y = build_label_space(labels, TASK_MULTICLASS)
    with pytest.raises(ValueError, match="fractions"):
        learning_curve_report(texts, y, space, fractions=bad_fractions)


# --------------------------------------------------------------------------- #
# run_difficulty_analysis
# --------------------------------------------------------------------------- #
def test_run_difficulty_analysis_writes_plots(tmp_path, mc_data):
    texts, labels = mc_data
    config = RunConfig(data=DataConfig(task_type=TASK_MULTICLASS))
    report = run_difficulty_analysis(texts, labels, config, out_dir=str(tmp_path))

    assert report.projection_plot is not None
    assert report.learning_curve_plot is not None
    assert os.path.isfile(report.projection_plot)
    assert os.path.isfile(report.learning_curve_plot)
    pngs = list((tmp_path / "difficulty" / "plots").glob("*.png"))
    assert len(pngs) == 2
    assert report.vector_method == "tfidf_svd"
    json.dumps(report.to_dict())


def test_run_difficulty_analysis_multilabel_smoke(tmp_path, ml_data):
    texts, labels = ml_data
    config = RunConfig(data=DataConfig(task_type=TASK_MULTILABEL))
    report = run_difficulty_analysis(texts, labels, config, out_dir=str(tmp_path))

    assert report.separability.silhouette is None
    pngs = list((tmp_path / "difficulty" / "plots").glob("*.png"))
    assert len(pngs) == 2
    json.dumps(report.to_dict())


def test_run_difficulty_analysis_without_out_dir(mc_data):
    texts, labels = mc_data
    config = RunConfig(data=DataConfig(task_type=TASK_MULTICLASS))
    report = run_difficulty_analysis(texts[:40], labels[:40], config, out_dir=None)
    assert report.projection_plot is None
    assert report.learning_curve_plot is None
    json.dumps(report.to_dict())
