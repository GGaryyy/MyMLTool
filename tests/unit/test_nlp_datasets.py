"""Unit tests for src.nlp.datasets — CSV loading, splitting, stratification."""

import numpy as np
import pandas as pd
import pytest

from src.nlp.config import config_from_dict
from src.nlp.datasets import TextDataset, load_text_dataset, split_text_dataset
from src.nlp.synth import TOPICS, generate_synthetic_gov_docs, texts_and_labels

pytestmark = pytest.mark.unit

N_DOCS = 100


def _write_csv(tmp_path, df, name="docs.csv"):
    path = tmp_path / name
    df.to_csv(path, index=False)
    return str(path)


def _make_config(csv_path, task_type="multiclass", text_col="text", label_col="label",
                 test_size=0.2, val_size=0.1, seed=0):
    return config_from_dict({
        "data": {
            "csv_path": csv_path,
            "text_col": text_col,
            "label_col": label_col,
            "task_type": task_type,
            "test_size": test_size,
            "val_size": val_size,
        },
        "seed": seed,
    })


@pytest.fixture
def balanced_csv(tmp_path):
    df = generate_synthetic_gov_docs(mode="balanced", n_docs=N_DOCS, seed=0,
                                     inject_defects=False)
    return _write_csv(tmp_path, df)


# --------------------------------------------------------------------------- #
# CSV round trip and split sizes
# --------------------------------------------------------------------------- #
def test_csv_round_trip_loads_and_splits(balanced_csv):
    ds = load_text_dataset(_make_config(balanced_csv))
    assert isinstance(ds, TextDataset)
    assert len(ds.texts_train) == 70
    assert len(ds.texts_val) == 10
    assert len(ds.texts_test) == 20
    assert ds.y_train.shape == (70,)
    assert ds.y_val.shape == (10,)
    assert ds.y_test.shape == (20,)
    assert ds.text_col == "text"
    assert ds.raw_label_col == "label"
    assert set(ds.label_space.classes) <= set(TOPICS)
    assert not ds.label_space.is_multilabel
    assert all(isinstance(t, str) for t in ds.texts_train)


def test_positional_columns_text_first_label_last(balanced_csv):
    ds = load_text_dataset(_make_config(balanced_csv, text_col=0, label_col=-1))
    assert ds.text_col == "text"
    assert ds.raw_label_col == "label"


def test_text_column_coerced_to_stripped_str(tmp_path):
    df = pd.DataFrame({
        "text": ["  前後有空白  "] + [f"一般文件{i}" for i in range(8)] + [123],
        "label": ["人事", "預算"] * 5,
    })
    path = _write_csv(tmp_path, df)
    ds = load_text_dataset(_make_config(path, val_size=0.0))
    all_texts = ds.texts_train + ds.texts_val + ds.texts_test
    assert "前後有空白" in all_texts
    assert "123" in all_texts


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #
def test_same_seed_gives_identical_split(balanced_csv):
    cfg = _make_config(balanced_csv, seed=13)
    ds_a = load_text_dataset(cfg)
    ds_b = load_text_dataset(cfg)
    assert ds_a.texts_train == ds_b.texts_train
    assert ds_a.texts_val == ds_b.texts_val
    assert ds_a.texts_test == ds_b.texts_test
    assert np.array_equal(ds_a.y_train, ds_b.y_train)
    assert np.array_equal(ds_a.y_val, ds_b.y_val)
    assert np.array_equal(ds_a.y_test, ds_b.y_test)


def test_different_seed_gives_different_split(balanced_csv):
    ds_a = load_text_dataset(_make_config(balanced_csv, seed=0))
    ds_b = load_text_dataset(_make_config(balanced_csv, seed=1))
    assert ds_a.texts_train != ds_b.texts_train


# --------------------------------------------------------------------------- #
# stratification
# --------------------------------------------------------------------------- #
def test_stratified_split_keeps_every_class_everywhere():
    texts = [f"文件內容第{i}號" for i in range(120)]
    labels = [TOPICS[i % len(TOPICS)] for i in range(120)]  # exactly 20 per class
    ds = split_text_dataset(texts, labels, task_type="multiclass", seed=0)
    n_classes = len(TOPICS)
    assert len(ds.texts_train) == 84
    assert len(ds.texts_val) == 12
    assert len(ds.texts_test) == 24
    assert set(np.unique(ds.y_train)) == set(range(n_classes))
    assert set(np.unique(ds.y_val)) == set(range(n_classes))
    assert set(np.unique(ds.y_test)) == set(range(n_classes))


def test_single_member_class_warns_and_still_splits():
    texts = [f"第{i}份文件" for i in range(100)]
    labels = ["人事"] * 50 + ["預算"] * 49 + ["罕見類"]
    with pytest.warns(UserWarning, match="non-stratified"):
        ds = split_text_dataset(texts, labels, task_type="multiclass", seed=0)
    assert len(ds.texts_train) == 70
    assert len(ds.texts_val) == 10
    assert len(ds.texts_test) == 20
    assert "罕見類" in ds.label_space.classes  # label space built on ALL labels


# --------------------------------------------------------------------------- #
# val_size = 0
# --------------------------------------------------------------------------- #
def test_val_size_zero_gives_empty_val_multiclass():
    texts = [f"第{i}份文件" for i in range(100)]
    labels = [TOPICS[i % 5] for i in range(100)]
    ds = split_text_dataset(texts, labels, task_type="multiclass", val_size=0.0, seed=0)
    assert ds.texts_val == []
    assert ds.y_val.shape == (0,)
    assert len(ds.texts_train) == 80
    assert len(ds.texts_test) == 20


def test_val_size_zero_gives_empty_val_multilabel():
    df = generate_synthetic_gov_docs(mode="multilabel", n_docs=60, seed=2,
                                     inject_defects=False)
    texts, labels = texts_and_labels(df)
    ds = split_text_dataset(texts, labels, task_type="multilabel", val_size=0.0, seed=0)
    assert ds.texts_val == []
    assert ds.y_val.shape == (0, ds.label_space.n_classes)


# --------------------------------------------------------------------------- #
# multilabel path
# --------------------------------------------------------------------------- #
def test_multilabel_split_shapes_and_no_unseen_class_crash():
    df = generate_synthetic_gov_docs(mode="multilabel", n_docs=100, seed=2,
                                     inject_defects=False)
    texts, labels = texts_and_labels(df)
    ds = split_text_dataset(texts, labels, task_type="multilabel", seed=0)
    n_classes = ds.label_space.n_classes
    assert ds.label_space.is_multilabel
    assert ds.y_train.shape == (70, n_classes)
    assert ds.y_val.shape == (10, n_classes)
    assert ds.y_test.shape == (20, n_classes)
    assert set(np.unique(ds.y_train)) <= {0, 1}


# --------------------------------------------------------------------------- #
# errors
# --------------------------------------------------------------------------- #
def test_missing_file_raises_file_not_found(tmp_path):
    cfg = _make_config(str(tmp_path / "nope.csv"))
    with pytest.raises(FileNotFoundError):
        load_text_dataset(cfg)


def test_empty_csv_raises_value_error(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_text_dataset(_make_config(str(path)))


def test_header_only_csv_raises_value_error(tmp_path):
    path = tmp_path / "header.csv"
    path.write_text("text,label\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no rows"):
        load_text_dataset(_make_config(str(path)))


def test_bad_column_name_raises(balanced_csv):
    with pytest.raises(ValueError, match="not found"):
        load_text_dataset(_make_config(balanced_csv, label_col="不存在"))


def test_column_index_out_of_range_raises(balanced_csv):
    with pytest.raises(ValueError, match="out of range"):
        load_text_dataset(_make_config(balanced_csv, label_col=7))


def test_text_and_label_same_column_raises(balanced_csv):
    with pytest.raises(ValueError, match="same column"):
        load_text_dataset(_make_config(balanced_csv, text_col="label", label_col="label"))


def test_nan_labels_raise_with_row_numbers(tmp_path):
    df = pd.DataFrame({"text": ["甲文", "乙文", "丙文"],
                       "label": ["人事", None, "預算"]})
    path = _write_csv(tmp_path, df, "nan.csv")
    with pytest.raises(ValueError, match=r"0-based row\(s\) \[1\]"):
        load_text_dataset(_make_config(path))


def test_nan_labels_list_only_first_ten_rows(tmp_path):
    df = pd.DataFrame({
        "text": [f"文{i}" for i in range(15)],
        "label": [None] * 12 + ["人事", "預算", "採購"],
    })
    path = _write_csv(tmp_path, df, "many_nan.csv")
    with pytest.raises(ValueError, match="12 NaN") as excinfo:
        load_text_dataset(_make_config(path))
    message = str(excinfo.value)
    assert "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]" in message
    assert "..." in message


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="length"):
        split_text_dataset(["甲", "乙"], ["人事"], task_type="multiclass")
