"""Unit tests for src.nlp.analysis.feature_selection and its report writer."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.nlp.analysis.feature_selection import (
    DEFAULT_K_GRID,
    FeatureScore,
    cramers_v,
    metadata_feature_selection,
    run_feature_selection,
    term_feature_selection,
)
from src.nlp.config import config_from_dict
from src.nlp.labels import build_label_space
from src.nlp.synth import generate_synthetic_gov_docs, texts_and_labels

pytestmark = pytest.mark.unit

REL_CAT = "來源機關"      # perfectly correlated with the label
IRREL_CAT = "隨機類別"    # random, irrelevant
NUM_A = "數值A"           # base numeric
NUM_B = "數值B"           # near-duplicate of NUM_A (collinear -> high VIF)
NUM_C = "數值C"           # random numeric


# --------------------------------------------------------------------------- #
# Fixtures / builders (local only — conftest untouched)
# --------------------------------------------------------------------------- #
def _make_dataset(mode: str, n_docs: int, seed: int):
    """Texts + labels from synth, plus a hand-built metadata DataFrame."""
    df = generate_synthetic_gov_docs(mode=mode, n_docs=n_docs, seed=seed)
    texts, labels = texts_and_labels(df)
    rng = np.random.default_rng(seed)

    source = [f"機關_{lab}" for lab in labels]           # perfect label correlation
    noise = rng.choice(["甲", "乙", "丙", "丁"], size=len(labels))
    base = rng.normal(size=len(labels))
    dup = base + rng.normal(0.0, 1e-3, size=len(labels))  # collinear with base
    rand = rng.normal(size=len(labels))

    metadata = pd.DataFrame({
        REL_CAT: source,
        IRREL_CAT: noise,
        NUM_A: base,
        NUM_B: dup,
        NUM_C: rand,
    })
    return texts, labels, metadata


def _multiclass_config():
    return config_from_dict({
        "data": {"task_type": "multiclass"},
        "segment": {"engine": "char"},
    })


# --------------------------------------------------------------------------- #
# term_feature_selection
# --------------------------------------------------------------------------- #
def test_term_feature_selection_multiclass_structure():
    texts, labels, _ = _make_dataset("balanced", 120, seed=0)
    label_space, y = build_label_space(labels, "multiclass")
    report = term_feature_selection(texts, y, label_space, seed=0)

    assert set(report.methods) == {"chi2", "mutual_info", "anova",
                                    "l1_logreg", "tree_importance"}
    for method, scored in report.methods.items():
        assert scored, f"{method} produced no ranked terms"
        assert all(isinstance(f, FeatureScore) for f in scored)
        vals = [f.score for f in scored]
        assert vals == sorted(vals, reverse=True), f"{method} not sorted desc"

    curve = report.feature_count_curve
    assert curve
    ks = [p["k"] for p in curve]
    assert ks == sorted(ks) and len(set(ks)) == len(ks)
    assert all(0.0 <= p["f1_macro"] <= 1.0 for p in curve)

    assert report.recommended_max_features in DEFAULT_K_GRID
    assert report.recommended_max_features <= max(ks)

    assert report.ngram_scores
    assert report.recommended_ngram in report.ngram_scores

    for pair in report.redundant_pairs:
        assert set(pair) == {"term_a", "term_b", "cosine"}
        assert pair["cosine"] >= 0.9


def test_term_feature_selection_multilabel_runs():
    texts, labels, _ = _make_dataset("multilabel", 80, seed=1)
    label_space, y = build_label_space(labels, "multilabel", "|")
    assert label_space.is_multilabel
    report = term_feature_selection(texts, y, label_space, seed=1)

    assert set(report.methods) == {"chi2", "mutual_info", "anova",
                                    "l1_logreg", "tree_importance"}
    assert report.feature_count_curve
    assert report.recommended_ngram in report.ngram_scores


# --------------------------------------------------------------------------- #
# metadata_feature_selection
# --------------------------------------------------------------------------- #
def test_metadata_relevance_and_correlation():
    texts, labels, metadata = _make_dataset("balanced", 120, seed=0)
    label_space, y = build_label_space(labels, "multiclass")
    report = metadata_feature_selection(metadata, y, label_space, seed=0)

    by_col = {r.column: r for r in report.relevance}
    rel = by_col[REL_CAT]
    irrel = by_col[IRREL_CAT]

    # The perfectly-correlated categorical dominates on both learned signals.
    assert rel.mutual_info == max(r.mutual_info for r in report.relevance)
    assert rel.tree_importance == max(r.tree_importance for r in report.relevance)
    assert rel.mutual_info > irrel.mutual_info
    assert 0.0 <= rel.cramers_v <= 1.0
    assert rel.cramers_v > 0.9  # near-perfect association

    # Collinear numeric pair is caught by Pearson AND VIF.
    corr = report.correlation
    num_pair = [p for p in corr.numeric_correlation_pairs
                if {p["col_a"], p["col_b"]} == {NUM_A, NUM_B}]
    assert num_pair and abs(num_pair[0]["pearson"]) >= 0.8
    assert NUM_A in corr.high_vif_columns or NUM_B in corr.high_vif_columns

    # Keep the relevant column; drop at least an irrelevant/redundant one.
    assert REL_CAT in report.recommended_keep
    assert set(report.recommended_drop) & {IRREL_CAT, NUM_A, NUM_B, NUM_C}


def test_metadata_constant_column_guarded():
    texts, labels, metadata = _make_dataset("balanced", 100, seed=3)
    metadata = metadata.copy()
    metadata["常數欄"] = 7.0
    label_space, y = build_label_space(labels, "multiclass")
    report = metadata_feature_selection(metadata, y, label_space, seed=3)

    const = next(r for r in report.relevance if r.column == "常數欄")
    assert const.mutual_info == pytest.approx(0.0, abs=1e-6)


def test_metadata_feature_selection_multilabel_runs():
    texts, labels, metadata = _make_dataset("multilabel", 80, seed=2)
    label_space, y = build_label_space(labels, "multilabel", "|")
    report = metadata_feature_selection(metadata, y, label_space, seed=2)
    assert REL_CAT in report.recommended_keep


# --------------------------------------------------------------------------- #
# cramers_v helper
# --------------------------------------------------------------------------- #
def test_cramers_v_perfect_independent_and_constant():
    perfect_x = ["a", "a", "b", "b", "c", "c"] * 4
    perfect_y = list(perfect_x)
    assert cramers_v(perfect_x, perfect_y) > 0.95

    rng = np.random.default_rng(0)
    ind_x = rng.choice(["p", "q", "r", "s"], size=400)
    ind_y = rng.choice(["p", "q", "r", "s"], size=400)
    assert cramers_v(ind_x, ind_y) < 0.3

    assert cramers_v(["z"] * 20, list(rng.choice(["a", "b"], size=20))) == 0.0


# --------------------------------------------------------------------------- #
# run_feature_selection
# --------------------------------------------------------------------------- #
def test_run_feature_selection_with_metadata(tmp_path):
    texts, labels, metadata = _make_dataset("balanced", 120, seed=0)
    config = _multiclass_config()
    report = run_feature_selection(texts, labels, config,
                                   metadata_df=metadata, out_dir=str(tmp_path))

    assert report.metadata is not None
    assert report.recommendations
    assert report.n_docs == 120

    payload = json.dumps(report.to_dict(), ensure_ascii=False)
    assert isinstance(payload, str) and payload

    assert report.plots
    for path in report.plots.values():
        assert Path(path).is_file()
        assert Path(path).stat().st_size > 0


def test_run_feature_selection_without_metadata():
    texts, labels, _ = _make_dataset("balanced", 100, seed=1)
    config = _multiclass_config()
    report = run_feature_selection(texts, labels, config, metadata_df=None)
    assert report.metadata is None
    assert report.term is not None
    assert report.recommendations


def test_run_feature_selection_validation_errors():
    texts, labels, metadata = _make_dataset("balanced", 60, seed=4)
    config = _multiclass_config()

    with pytest.raises(ValueError):
        run_feature_selection(texts, labels[:-1], config)
    with pytest.raises(ValueError):
        run_feature_selection([], [], config)
    with pytest.raises(ValueError):
        run_feature_selection(texts, labels, config, metadata_df=metadata.iloc[:-1])


# --------------------------------------------------------------------------- #
# write_feature_selection_report
# --------------------------------------------------------------------------- #
def test_write_feature_selection_report_with_metadata(tmp_path):
    from src.nlp.report import write_feature_selection_report

    texts, labels, metadata = _make_dataset("balanced", 100, seed=0)
    config = _multiclass_config()
    report = run_feature_selection(texts, labels, config,
                                   metadata_df=metadata, out_dir=str(tmp_path))
    written = write_feature_selection_report(report, str(tmp_path))

    md_path = Path(written["markdown"])
    json_path = Path(written["json"])
    assert md_path == tmp_path / "feature_selection" / "feature_selection_report.md"
    assert md_path.is_file() and json_path.is_file()

    parsed = json.loads(json_path.read_text(encoding="utf-8"))
    assert parsed["n_docs"] == 100
    assert parsed["metadata"] is not None

    md = md_path.read_text(encoding="utf-8")
    assert "總體建議" in md
    assert "Metadata 特徵" in md


def test_write_feature_selection_report_without_metadata(tmp_path):
    from src.nlp.report import write_feature_selection_report

    texts, labels, _ = _make_dataset("balanced", 80, seed=1)
    config = _multiclass_config()
    report = run_feature_selection(texts, labels, config, metadata_df=None)
    written = write_feature_selection_report(report, str(tmp_path))

    md = Path(written["markdown"]).read_text(encoding="utf-8")
    assert "總體建議" in md
    assert "Metadata 特徵" not in md


def test_report_module_regression_guard():
    import src.nlp.report as report_mod

    assert callable(report_mod.write_eda_report)
    assert callable(report_mod.write_benchmark_report)
    assert callable(report_mod.write_feature_selection_report)
