"""Unit tests for src.nlp.analysis.keywords — chi2/mi keyword extraction."""

import json

import pytest

from src.nlp.analysis.keywords import class_keywords
from src.nlp.config import TASK_MULTICLASS, TASK_MULTILABEL
from src.nlp.synth import TOPICS, generate_synthetic_gov_docs, texts_and_labels

pytestmark = pytest.mark.unit

SECURITY_MARKERS = ("資安", "資通", "弱點", "端點")


@pytest.fixture(scope="module")
def kw_corpus():
    df = generate_synthetic_gov_docs(mode="balanced", n_docs=120, seed=11,
                                     inject_defects=False)
    return texts_and_labels(df)


@pytest.fixture(scope="module")
def chi2_report(kw_corpus):
    texts, labels = kw_corpus
    return class_keywords(texts, labels, TASK_MULTICLASS)


# --------------------------------------------------------------------------- #
# chi2 on balanced multiclass
# --------------------------------------------------------------------------- #
def test_chi2_covers_all_topics(chi2_report):
    assert chi2_report.method == "chi2"
    assert {entry.class_name for entry in chi2_report.per_class} == set(TOPICS)


def test_chi2_finds_security_vocab(chi2_report):
    entry = next(e for e in chi2_report.per_class if e.class_name == "資訊安全")
    assert entry.keywords
    joined = "".join(gram for gram, _ in entry.keywords)
    assert any(marker in joined for marker in SECURITY_MARKERS)


def test_chi2_scores_sorted_descending(chi2_report):
    for entry in chi2_report.per_class:
        scores = [score for _, score in entry.keywords]
        assert scores == sorted(scores, reverse=True)


def test_top_k_is_respected(kw_corpus, chi2_report):
    texts, labels = kw_corpus
    for entry in chi2_report.per_class:
        assert len(entry.keywords) <= 20
    small = class_keywords(texts, labels, TASK_MULTICLASS, top_k=5)
    for entry in small.per_class:
        assert len(entry.keywords) <= 5


def test_report_to_dict_is_json_serializable(chi2_report):
    json.dumps(chi2_report.to_dict())


# --------------------------------------------------------------------------- #
# mi method
# --------------------------------------------------------------------------- #
def test_mi_method_runs(kw_corpus):
    texts, labels = kw_corpus
    report = class_keywords(texts[:60], labels[:60], TASK_MULTICLASS,
                            top_k=10, method="mi", min_df=3)
    assert report.method == "mi"
    assert any(entry.keywords for entry in report.per_class)
    for entry in report.per_class:
        assert len(entry.keywords) <= 10
        scores = [score for _, score in entry.keywords]
        assert scores == sorted(scores, reverse=True)


# --------------------------------------------------------------------------- #
# validation and edge cases
# --------------------------------------------------------------------------- #
def test_invalid_method_raises(kw_corpus):
    texts, labels = kw_corpus
    with pytest.raises(ValueError, match="method must be one of"):
        class_keywords(texts, labels, TASK_MULTICLASS, method="tfidf")


def test_rare_class_yields_empty_keyword_entry(kw_corpus):
    texts, labels = kw_corpus
    texts = list(texts[:40]) + ["這是一份非常罕見的特殊測試文件內容。"]
    labels = list(labels[:40]) + ["稀有類"]
    report = class_keywords(texts, labels, TASK_MULTICLASS)

    entry = next(e for e in report.per_class if e.class_name == "稀有類")
    assert entry.keywords == []
    assert any(e.keywords for e in report.per_class if e.class_name != "稀有類")


# --------------------------------------------------------------------------- #
# multilabel smoke
# --------------------------------------------------------------------------- #
def test_multilabel_smoke():
    df = generate_synthetic_gov_docs(mode="multilabel", n_docs=60, seed=11,
                                     inject_defects=False)
    texts, labels = texts_and_labels(df)
    report = class_keywords(texts, labels, TASK_MULTILABEL)

    assert {entry.class_name for entry in report.per_class} <= set(TOPICS)
    assert any(entry.keywords for entry in report.per_class)
    json.dumps(report.to_dict())
