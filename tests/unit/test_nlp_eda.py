"""Unit tests for src.nlp.eda — run_eda statistics on synthetic Chinese text data."""

import json

import pytest

from src.nlp.config import config_from_dict
from src.nlp.eda import (
    BERT_TOKEN_LIMIT,
    SHORT_TEXT_CHARS,
    TextEdaReport,
    run_eda,
)
from src.nlp.synth import generate_synthetic_gov_docs, texts_and_labels

pytestmark = pytest.mark.unit


def _config(task_type="multiclass"):
    return config_from_dict({
        "data": {"task_type": task_type},
        "segment": {"engine": "char"},
    })


def _imbalanced_data():
    df = generate_synthetic_gov_docs(mode="imbalanced", n_docs=300, seed=0)
    return texts_and_labels(df)


@pytest.fixture(scope="module")
def imbalanced_report():
    texts, labels = _imbalanced_data()
    return run_eda(texts, labels, _config())


# --------------------------------------------------------------------------- #
# report shape and class balance
# --------------------------------------------------------------------------- #
def test_report_type_and_basic_fields(imbalanced_report):
    assert isinstance(imbalanced_report, TextEdaReport)
    assert imbalanced_report.n_docs == 300
    assert imbalanced_report.task_type == "multiclass"
    assert imbalanced_report.leakage is None


def test_imbalance_detected_and_minority_flagged(imbalanced_report):
    balance = imbalanced_report.balance
    assert balance.imbalance_ratio > 1
    assert balance.minority_classes
    assert 0 < balance.entropy <= 1
    assert sum(balance.counts.values()) == 300


def test_single_class_entropy_is_one():
    report = run_eda(
        ["第一份文件內容說明如下", "第二份文件內容說明如下"],
        ["人事", "人事"],
        _config(),
    )
    assert report.balance.entropy == 1.0
    assert report.balance.imbalance_ratio == 1.0
    assert report.balance.minority_classes == []


# --------------------------------------------------------------------------- #
# quality defects surface
# --------------------------------------------------------------------------- #
def test_synth_defects_surface_in_quality(imbalanced_report):
    quality = imbalanced_report.quality
    assert quality.exact_duplicate_docs > 0
    assert quality.exact_duplicate_groups > 0
    assert quality.label_conflicts > 0
    assert quality.conflict_examples
    assert all(isinstance(i, int) for group in quality.conflict_examples for i in group)
    assert quality.short_texts >= 2  # the "函" / "主旨：" stubs
    assert quality.empty_texts == 0


def test_empty_and_short_texts_counted():
    report = run_eda(
        ["", "這是一份長度正常的文件內容說明", "短文"],
        ["甲", "乙", "甲"],
        _config(),
    )
    assert report.quality.empty_texts == 1
    assert report.quality.short_texts == 2  # empty doc is also < SHORT_TEXT_CHARS
    assert SHORT_TEXT_CHARS == 10


def test_near_duplicates_detected_but_exact_dups_excluded():
    base = "資通安全管理法弱點掃描端點防護資安事件通報社交工程演練資訊資產盤點"
    near_a = base + "一"
    near_b = base + "二"
    report = run_eda(
        [near_a, near_b, "完全不同的另一份文件內容僅供對照使用"],
        ["資訊安全", "資訊安全", "人事"],
        _config(),
    )
    assert report.quality.near_duplicate_pairs == 1

    exact = run_eda(
        [base, base, "完全不同的另一份文件內容僅供對照使用"],
        ["資訊安全", "資訊安全", "人事"],
        _config(),
    )
    assert exact.quality.near_duplicate_pairs == 0
    assert exact.quality.exact_duplicate_docs == 2


def test_pii_and_normalization_surface(imbalanced_report):
    assert imbalanced_report.pii.docs_with_pii > 0
    assert imbalanced_report.normalization.docs_with_fullwidth > 0
    assert sum(imbalanced_report.normalization.ocr_suspect_counts.values()) > 0


# --------------------------------------------------------------------------- #
# length and vocabulary statistics
# --------------------------------------------------------------------------- #
def test_over_512_ratio_positive_with_char_segmenter(imbalanced_report):
    length = imbalanced_report.length
    assert length.over_512_ratio > 0
    assert length.token_max > BERT_TOKEN_LIMIT
    # char segmenter: token count is close to (and never above) char count
    assert length.token_max <= length.max
    assert length.max > 600
    assert len(length.char_lengths) == 300


def test_top_tokens_nonempty_and_sorted_desc(imbalanced_report):
    vocab = imbalanced_report.vocab
    assert vocab.top_tokens
    counts = [count for _, count in vocab.top_tokens]
    assert counts == sorted(counts, reverse=True)
    assert vocab.top_bigrams and vocab.top_trigrams
    assert vocab.vocab_size > 0
    assert 0 < vocab.ttr <= 1
    assert 0 <= vocab.hapax_ratio <= 1


def test_warnings_nonempty_and_mention_class_weight(imbalanced_report):
    warnings_list = imbalanced_report.warnings
    assert warnings_list
    assert any("class_weight" in w for w in warnings_list)
    assert any("f1_macro" in w for w in warnings_list)


# --------------------------------------------------------------------------- #
# multilabel mode
# --------------------------------------------------------------------------- #
def test_multilabel_per_label_counts_sum_at_least_n_docs():
    df = generate_synthetic_gov_docs(mode="multilabel", n_docs=200, seed=0)
    texts, labels = texts_and_labels(df)
    report = run_eda(texts, labels, _config(task_type="multilabel"))
    assert report.task_type == "multilabel"
    assert sum(report.balance.counts.values()) >= 200


# --------------------------------------------------------------------------- #
# split leakage
# --------------------------------------------------------------------------- #
def test_leakage_detects_shared_exact_text():
    texts = [
        "共用的文件內容甲種文件全文",
        "訓練集專屬的文件內容一",
        "共用的文件內容甲種文件全文",
        "測試集專屬的文件內容二",
    ]
    labels = ["人事", "預算", "人事", "預算"]
    splits = {"train": [0, 1], "test": [2, 3]}
    report = run_eda(texts, labels, _config(), splits=splits)
    assert report.leakage is not None
    assert report.leakage.pairs["train~test"] == 1
    example = report.leakage.leaked_examples[0]
    assert example["pair"] == "train~test"
    assert example["row_a"] == 0
    assert example["row_b"] == 2


def test_leakage_invalid_split_index_raises():
    with pytest.raises(ValueError, match="out of range"):
        run_eda(
            ["文件一內容", "文件二內容"],
            ["人事", "預算"],
            _config(),
            splits={"train": [0], "test": [99]},
        )


# --------------------------------------------------------------------------- #
# input validation
# --------------------------------------------------------------------------- #
def test_mismatched_lengths_raise_value_error():
    with pytest.raises(ValueError, match="mismatch"):
        run_eda(["只有一份文件"], ["人事", "預算"], _config())


def test_empty_inputs_raise_value_error():
    with pytest.raises(ValueError, match="empty"):
        run_eda([], [], _config())


def test_non_str_text_raises_type_error():
    with pytest.raises(TypeError):
        run_eda([12345], ["人事"], _config())


def test_bad_config_type_raises_type_error():
    with pytest.raises(TypeError, match="RunConfig"):
        run_eda(["文件內容"], ["人事"], {"data": {}})


# --------------------------------------------------------------------------- #
# determinism and serializability
# --------------------------------------------------------------------------- #
def test_same_input_gives_identical_to_dict():
    texts, labels = _imbalanced_data()
    report_a = run_eda(texts, labels, _config())
    report_b = run_eda(texts, labels, _config())
    assert report_a.to_dict() == report_b.to_dict()


def test_to_dict_json_serializable(imbalanced_report):
    payload = json.dumps(imbalanced_report.to_dict(), ensure_ascii=False)
    parsed = json.loads(payload)
    assert parsed["n_docs"] == 300
    assert parsed["leakage"] is None
