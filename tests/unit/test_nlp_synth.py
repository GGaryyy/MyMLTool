"""Unit tests for src.nlp.synth — determinism, modes, defect injection."""

import pandas as pd
import pytest

from src.nlp.synth import (
    MODES,
    TOPICS,
    generate_synthetic_gov_docs,
    texts_and_labels,
)

pytestmark = pytest.mark.unit

ID_PATTERN = r"A\d{9}"
PHONE_PATTERN = r"\d{2}-\d{4}-\d{4}"
MOBILE_PATTERN = r"09\d{2}-\d{3}-\d{3}"


# --------------------------------------------------------------------------- #
# determinism and frame shape
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", MODES)
def test_same_seed_gives_identical_frames(mode):
    df_a = generate_synthetic_gov_docs(mode=mode, n_docs=120, seed=3)
    df_b = generate_synthetic_gov_docs(mode=mode, n_docs=120, seed=3)
    pd.testing.assert_frame_equal(df_a, df_b)


def test_different_seed_gives_different_frames():
    df_a = generate_synthetic_gov_docs(n_docs=120, seed=0)
    df_b = generate_synthetic_gov_docs(n_docs=120, seed=1)
    assert not df_a.equals(df_b)


def test_column_order_is_text_then_label():
    df = generate_synthetic_gov_docs(n_docs=30, seed=0)
    assert list(df.columns) == ["text", "label"]
    assert len(df) == 30


# --------------------------------------------------------------------------- #
# class distributions per mode
# --------------------------------------------------------------------------- #
def test_balanced_mode_roughly_uniform_at_600():
    df = generate_synthetic_gov_docs(mode="balanced", n_docs=600, seed=0)
    counts = df["label"].value_counts()
    assert set(counts.index) <= set(TOPICS)
    assert counts.max() / counts.min() < 2.5


def test_imbalanced_mode_long_tail_at_600():
    df = generate_synthetic_gov_docs(mode="imbalanced", n_docs=600, seed=0)
    counts = df["label"].value_counts()
    assert set(counts.index) <= set(TOPICS)
    assert counts.max() > 3 * counts.min()


def test_multilabel_mode_labels_are_valid_sorted_topic_combos():
    df = generate_synthetic_gov_docs(mode="multilabel", n_docs=300, seed=0)
    for label in df["label"]:
        parts = label.split("|")
        assert 1 <= len(parts) <= 3
        assert len(set(parts)) == len(parts)
        assert parts == sorted(parts)
        assert all(part in TOPICS for part in parts)
    # 1-3 topics per doc implies multi-topic rows actually occur.
    assert (df["label"].str.contains(r"\|", regex=True)).any()


# --------------------------------------------------------------------------- #
# defect injection ON
# --------------------------------------------------------------------------- #
@pytest.fixture
def defect_df():
    return generate_synthetic_gov_docs(mode="balanced", n_docs=300, seed=0,
                                       inject_defects=True)


def test_defects_include_exact_duplicates(defect_df):
    assert defect_df["text"].duplicated().any()
    duplicated = defect_df[defect_df.duplicated(keep=False)]
    assert len(duplicated) >= 2  # at least one full-row duplicate pair


def test_defects_include_label_conflicts(defect_df):
    labels_per_text = defect_df.groupby("text")["label"].nunique()
    assert (labels_per_text > 1).any()


def test_defects_include_near_empty_texts(defect_df):
    texts = set(defect_df["text"])
    assert "函" in texts
    assert "主旨：" in texts


def test_defects_include_fake_pii(defect_df):
    text = defect_df["text"]
    assert text.str.contains(ID_PATTERN, regex=True).any()
    assert text.str.contains(PHONE_PATTERN, regex=True).any()
    assert text.str.contains(MOBILE_PATTERN, regex=True).any()


def test_defects_include_fullwidth_and_ocr_noise(defect_df):
    text = defect_df["text"]
    assert text.str.contains("１２３ＡＢ", regex=False).any()
    assert text.str.contains("〇", regex=False).any()
    assert text.str.contains("ｌ", regex=False).any()


# --------------------------------------------------------------------------- #
# defect injection OFF
# --------------------------------------------------------------------------- #
def test_clean_generation_has_no_defects():
    df = generate_synthetic_gov_docs(mode="balanced", n_docs=300, seed=0,
                                     inject_defects=False)
    # Unique serial numbers make base texts collision-free.
    assert not df["text"].duplicated().any()
    labels_per_text = df.groupby("text")["label"].nunique()
    assert not (labels_per_text > 1).any()
    assert not df["text"].str.contains(ID_PATTERN, regex=True).any()
    assert df["text"].str.len().min() > 20  # no near-empty stubs


# --------------------------------------------------------------------------- #
# lengths
# --------------------------------------------------------------------------- #
def test_some_docs_exceed_600_chars():
    df = generate_synthetic_gov_docs(mode="balanced", n_docs=300, seed=0,
                                     inject_defects=False)
    lengths = df["text"].str.len()
    assert (lengths > 600).any()
    assert lengths.nunique() > 10  # 說明 clause count varies doc lengths


# --------------------------------------------------------------------------- #
# validation and helpers
# --------------------------------------------------------------------------- #
def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="mode must be one of"):
        generate_synthetic_gov_docs(mode="bogus")


@pytest.mark.parametrize("n_docs", [0, -5])
def test_invalid_n_docs_raises(n_docs):
    with pytest.raises(ValueError, match="n_docs"):
        generate_synthetic_gov_docs(n_docs=n_docs)


def test_texts_and_labels_returns_matching_string_lists():
    df = generate_synthetic_gov_docs(mode="multilabel", n_docs=40, seed=0)
    texts, labels = texts_and_labels(df)
    assert isinstance(texts, list) and isinstance(labels, list)
    assert len(texts) == len(labels) == 40
    assert all(isinstance(t, str) for t in texts)
    assert all(isinstance(lab, str) for lab in labels)


def test_texts_and_labels_missing_columns_raises():
    with pytest.raises(ValueError, match="'text' and 'label'"):
        texts_and_labels(pd.DataFrame({"a": [1]}))
