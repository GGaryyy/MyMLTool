"""Unit tests for src.nlp.labels — LabelSpace, parsing, distributions."""

import numpy as np
import pytest

from src.nlp.config import TASK_MULTICLASS, TASK_MULTILABEL
from src.nlp.labels import (
    LabelSpace,
    build_label_space,
    class_distribution,
    parse_multilabel,
)

pytestmark = pytest.mark.unit

MULTILABEL_RAW = ["資訊安全|人事", "預算", "人事|預算"]
# sorted() by codepoint: 人事 (U+4EBA) < 資訊安全 (U+8CC7) < 預算 (U+9810)
MULTILABEL_CLASSES = ["人事", "資訊安全", "預算"]


# --------------------------------------------------------------------------- #
# build_label_space — multiclass
# --------------------------------------------------------------------------- #
def test_build_label_space_multiclass_sorted_classes():
    space, y = build_label_space(["預算", "人事", "預算", "法規"], TASK_MULTICLASS)
    assert space.classes == sorted({"預算", "人事", "法規"})
    assert space.is_multilabel is False
    assert space.n_classes == 3
    assert y.dtype == np.int64
    assert y.shape == (4,)


def test_build_label_space_multiclass_deterministic_across_row_order():
    space_a, _ = build_label_space(["c", "a", "b"], TASK_MULTICLASS)
    space_b, _ = build_label_space(["b", "c", "a"], TASK_MULTICLASS)
    assert space_a.classes == space_b.classes == ["a", "b", "c"]


def test_multiclass_encode_decode_round_trip():
    raw = ["b", "a", "c", "a"]
    space, y = build_label_space(raw, TASK_MULTICLASS)
    assert space.decode(y) == raw


def test_multiclass_encode_unknown_class_raises():
    space, _ = build_label_space(["a", "b"], TASK_MULTICLASS)
    with pytest.raises(ValueError, match="Unknown class"):
        space.encode(["c"])


def test_numeric_labels_coerced_to_str():
    space, y = build_label_space([10, 2, 2], TASK_MULTICLASS)
    # String sort, not numeric: "10" < "2".
    assert space.classes == ["10", "2"]
    assert all(isinstance(c, str) for c in space.classes)
    np.testing.assert_array_equal(y, [0, 1, 1])
    assert space.decode(y) == ["10", "2", "2"]


def test_build_label_space_bad_task_type_raises():
    with pytest.raises(ValueError, match="task_type must be one of"):
        build_label_space(["a"], "binary")


def test_build_label_space_empty_input_raises():
    with pytest.raises(ValueError, match="zero labels"):
        build_label_space([], TASK_MULTICLASS)
    with pytest.raises(ValueError, match="zero labels"):
        build_label_space([], TASK_MULTILABEL)


# --------------------------------------------------------------------------- #
# parse_multilabel
# --------------------------------------------------------------------------- #
def test_parse_multilabel_splits_on_separator():
    assert parse_multilabel(["人事|預算", "採購"]) == [["人事", "預算"], ["採購"]]


def test_parse_multilabel_strips_whitespace():
    assert parse_multilabel([" 人事 | 預算 "]) == [["人事", "預算"]]


def test_parse_multilabel_drops_empty_parts():
    assert parse_multilabel(["人事||預算", "|採購"]) == [["人事", "預算"], ["採購"]]


def test_parse_multilabel_custom_separator():
    assert parse_multilabel(["a;b"], separator=";") == [["a", "b"]]


def test_parse_multilabel_all_empty_row_raises():
    with pytest.raises(ValueError, match="no labels"):
        parse_multilabel(["人事", "||"])


def test_parse_multilabel_empty_separator_raises():
    with pytest.raises(ValueError, match="separator"):
        parse_multilabel(["a|b"], separator="")


# --------------------------------------------------------------------------- #
# build_label_space — multilabel
# --------------------------------------------------------------------------- #
def test_build_label_space_multilabel_indicator_matrix():
    space, y = build_label_space(MULTILABEL_RAW, TASK_MULTILABEL)
    assert space.is_multilabel is True
    assert space.classes == MULTILABEL_CLASSES
    np.testing.assert_array_equal(y, [[1, 1, 0], [0, 0, 1], [1, 0, 1]])


def test_multilabel_decode_round_trip():
    space, y = build_label_space(MULTILABEL_RAW, TASK_MULTILABEL)
    # decode() lists names in class-index order.
    assert space.decode(y) == [["人事", "資訊安全"], ["預算"], ["人事", "預算"]]


def test_multilabel_encode_unknown_class_raises():
    space, _ = build_label_space(MULTILABEL_RAW, TASK_MULTILABEL)
    with pytest.raises(ValueError, match="Unknown class"):
        space.encode([["人事", "幽靈類別"]])


def test_multilabel_encode_rejects_plain_string_rows():
    space, _ = build_label_space(MULTILABEL_RAW, TASK_MULTILABEL)
    with pytest.raises(ValueError, match="label-lists"):
        space.encode(["人事"])


# --------------------------------------------------------------------------- #
# LabelSpace construction guards
# --------------------------------------------------------------------------- #
def test_label_space_zero_classes_raises():
    with pytest.raises(ValueError, match="at least one class"):
        LabelSpace(classes=[], is_multilabel=False)


def test_label_space_duplicate_classes_raises():
    with pytest.raises(ValueError, match="unique"):
        LabelSpace(classes=["人事", "人事"], is_multilabel=True)


# --------------------------------------------------------------------------- #
# decode shape errors
# --------------------------------------------------------------------------- #
def test_multiclass_decode_rejects_2d_input():
    space = LabelSpace(classes=["a", "b"], is_multilabel=False)
    with pytest.raises(ValueError, match="1-D"):
        space.decode(np.array([[0, 1]]))


def test_multiclass_decode_out_of_range_index_raises():
    space = LabelSpace(classes=["a", "b"], is_multilabel=False)
    with pytest.raises(ValueError, match="out of range"):
        space.decode(np.array([0, 7]))


def test_multilabel_decode_rejects_1d_input():
    space = LabelSpace(classes=["a", "b", "c"], is_multilabel=True)
    with pytest.raises(ValueError, match="indicator matrix"):
        space.decode(np.array([0, 1, 0]))


def test_multilabel_decode_rejects_wrong_width():
    space = LabelSpace(classes=["a", "b", "c"], is_multilabel=True)
    with pytest.raises(ValueError, match="indicator matrix"):
        space.decode(np.zeros((2, 2), dtype=np.int64))


# --------------------------------------------------------------------------- #
# class_distribution
# --------------------------------------------------------------------------- #
def test_class_distribution_multiclass():
    space, y = build_label_space(["b", "a", "b", "c", "b"], TASK_MULTICLASS)
    assert class_distribution(space, y) == {"a": 1, "b": 3, "c": 1}


def test_class_distribution_multilabel():
    space, y = build_label_space(MULTILABEL_RAW, TASK_MULTILABEL)
    assert class_distribution(space, y) == {"人事": 2, "資訊安全": 1, "預算": 2}
